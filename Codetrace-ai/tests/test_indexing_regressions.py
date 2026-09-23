"""
Regression tests for the call-graph / indexing correctness fixes:

- incremental re-index keeps calls from unchanged files (prune_files + restore)
- deleted files are recognised as indexed by path alone
- git_diff passes the revision before "--"
- symbol IDs resolve from relative / forward-slash / bare forms
- nested .gitignore patterns stay scoped to their own directory
- callee resolution is order-independent and conservative
- JS calls inside `const x = f()` belong to the enclosing function
"""

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from src.core.agents.tools import analyze_impact_impl, get_symbol_relations_impl, git_diff_impl
from src.core.graph.builder import CodeGraph
from src.core.graph.orchestrator import GraphOrchestrator
from src.core.parser.parser import CodeParser
from src.ignore import read_gitignore


# Incremental re-index


def test_reindexing_a_file_keeps_calls_from_unchanged_files(tmp_path):
    g = CodeGraph(db_dir=tmp_path)
    g.add_nodes_batch([
        ("a.py:caller", "function", "a.py"),
        ("b.py:target", "function", "b.py"),
        ("b.py:renamed_away", "function", "b.py"),
    ])
    g.add_edges_batch([("a.py:caller", "b.py:target"), ("a.py:caller", "b.py:renamed_away")])
    g.persist_to_db()

    # b.py changes: prune it, re-parse only b.py (renamed_away no longer exists).
    inbound = g.prune_files(["b.py"])
    g.add_nodes_batch([("b.py:target", "function", "b.py")])
    restored = g.restore_inbound_edges(inbound)
    g.persist_to_db()
    g.load_from_db()

    assert g.get_callers("b.py:target") == ["a.py:caller"]
    assert restored == 1
    # The edge to the removed symbol must not come back as a phantom node.
    assert "b.py:renamed_away" not in g.direct_graph


def test_prune_does_not_return_edges_between_pruned_files(tmp_path):
    g = CodeGraph(db_dir=tmp_path)
    g.add_nodes_batch([("a.py:f", "function", "a.py"), ("b.py:g", "function", "b.py")])
    g.add_edges_batch([("a.py:f", "b.py:g")])
    g.persist_to_db()

    # Both files re-parsed in the same batch: a.py's call is rebuilt by parsing.
    assert g.prune_files(["a.py", "b.py"]) == []


def test_deleted_files_are_recognised_as_indexed_without_existing_on_disk(tmp_path):
    orch = GraphOrchestrator()
    gone = [str(tmp_path / "gone.py"), str(tmp_path / "Dockerfile"), str(tmp_path / "notes.txt")]
    supported = [f for f, _ in orch.iter_supported_files(gone)]
    assert supported == [gone[0], gone[1]]


# git_diff


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_diff_head_reports_uncommitted_changes(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "f.txt").write_text("one\n")
    git("add", "f.txt")
    git("commit", "-q", "-m", "init")
    (tmp_path / "f.txt").write_text("two\n")
    monkeypatch.chdir(tmp_path)

    out = git_diff_impl("HEAD")
    assert "+two" in out and "-one" in out
    assert git_diff_impl("--staged") == "No changes found."
    assert git_diff_impl("--output=/tmp/x").startswith("Blocked")


# Symbol ID resolution


@pytest.fixture
def abs_graph():
    root = "E:\\proj" if Path("E:\\").drive else "/proj"
    sep = "\\" if Path("E:\\").drive else "/"
    f = lambda rel: root + sep + rel.replace("/", sep)
    g = CodeGraph()
    g.add_nodes_batch([
        (f"{f('src/b.py')}:Service.run", "function", f("src/b.py")),
        (f"{f('src/ab.py')}:Service.run", "function", f("src/ab.py")),
        (f"{f('src/c.py')}:helper", "function", f("src/c.py")),
    ])
    g.add_edges_batch([(f"{f('src/c.py')}:helper", f"{f('src/b.py')}:Service.run")])
    return g, f


def test_symbol_id_resolves_from_relative_and_forward_slash_forms(abs_graph):
    g, f = abs_graph
    want = f"{f('src/b.py')}:Service.run"
    assert g.resolve_symbol_id(want) == [want]
    assert g.resolve_symbol_id("src/b.py:Service.run") == [want]
    assert g.resolve_symbol_id("./src/b.py:Service.run") == [want]
    assert g.resolve_symbol_id("b.py:Service.run") == [want]  # never matches ab.py
    assert g.resolve_symbol_id("helper") == [f"{f('src/c.py')}:helper"]
    assert g.resolve_symbol_id("src/b.py:Nope.run") == []


def test_graph_tools_accept_relative_ids_and_flag_ambiguity(abs_graph):
    g, _ = abs_graph
    assert "helper" in get_symbol_relations_impl(g, "src/b.py:Service.run")
    assert "Total affected symbols: 1" in analyze_impact_impl(g, "src/b.py:Service.run")
    assert "ambiguous" in get_symbol_relations_impl(g, "Service.run")
    assert "not found" in analyze_impact_impl(g, "missing.py:nothing")


# .gitignore scoping


def test_nested_gitignore_patterns_stay_in_their_directory(tmp_path):
    (tmp_path / "packages" / "web").mkdir(parents=True)
    (tmp_path / "packages" / "web" / ".gitignore").write_text("*.js\n!keep.js\n/generated\n")
    (tmp_path / ".gitignore").write_text("/scratch\n")
    spec = read_gitignore(str(tmp_path))

    assert spec.match_file("packages/web/bundle.js")
    assert spec.match_file("packages/web/deep/bundle.js")
    assert not spec.match_file("packages/web/keep.js")
    assert spec.match_file("packages/web/generated/x.py")
    assert not spec.match_file("src/app.js")          # was ignored repo-wide
    assert not spec.match_file("src/generated/x.py")
    assert spec.match_file("scratch/out.py")
    assert not spec.match_file("src/scratch/util.py")  # root '/scratch' is anchored


# Callee resolution


def _sym(name, qualified=None):
    return {"name": name, "qualified_name": qualified or name, "type": "function"}


def _call(caller, callee):
    return {"caller": caller, "callee": callee}


def test_callee_resolution_is_order_independent_and_conservative():
    symbols = [
        ("a.py", _sym("run", "Service.run")), ("a.py", _sym("helper", "Service.helper")),
        ("a.py", _sym("helper")),
        ("b.py", _sym("save")), ("c.py", _sym("save")),
        ("d.py", _sym("only_here")),
        ("e.js", _sym("util")),
    ]
    calls = [
        ("a.py", _call("Service.run", "helper")),   # own class wins over module helper
        ("a.py", _call("helper", "save")),          # two candidates -> unresolved
        ("a.py", _call("helper", "only_here")),     # unique in language -> resolved
        ("a.py", _call("helper", "util")),          # only exists in JS -> unresolved
        ("a.py", _call("helper", "append")),        # unknown -> unresolved
    ]
    expected = {
        ("a.py:Service.run", "a.py:Service.helper"),
        ("a.py:helper", "save"),
        ("a.py:helper", "d.py:only_here"),
        ("a.py:helper", "util"),
        ("a.py:helper", "append"),
    }

    for seed in range(5):
        s, c = symbols[:], calls[:]
        random.Random(seed).shuffle(s)
        random.Random(seed).shuffle(c)
        _, edges = GraphOrchestrator().build_batch(s, c)
        assert set(edges) == expected


def test_callee_resolves_into_unchanged_files_already_in_graph():
    orch = GraphOrchestrator()
    orch.graph.add_nodes_batch([("lib.py:shared", "function", "lib.py")])
    _, edges = orch.build_batch([("app.py", _sym("main"))], [("app.py", _call("main", "shared"))])
    assert edges == [("app.py:main", "lib.py:shared")]


# JS caller attribution


def test_js_call_in_plain_declarator_belongs_to_enclosing_function():
    js = (
        "function outer() {\n"
        "  const x = compute();\n"
        "}\n"
        "const handler = () => { save(); };\n"
    )
    _, calls = CodeParser().extract_symbols_and_calls(js, "javascript")
    pairs = {(c["caller"], c["callee"]) for c in calls}
    assert ("outer", "compute") in pairs
    assert ("handler", "save") in pairs
    assert not any(caller == "x" for caller, _ in pairs)
