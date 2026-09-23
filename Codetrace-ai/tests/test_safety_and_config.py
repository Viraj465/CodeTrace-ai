"""
Regression tests for the write-safety, config, MCP-registration, snapshot
lookup, output-normalizer and remote-indexing fixes.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from src.cli.config_helpers import register_mcp
from src.cli.project_helpers import clone_repo, repo_cache_dir
from src.config_io import write_private_json
from src.core.agents.output_normalizer import OutputNormalizer
from src.core.agents.tools import clear_pending_writes, get_pending_writes, propose_write_impl
from src.core.database.sync_manager import SyncManager
from src.core.graph.orchestrator import GraphOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Writes stay inside the project


def test_proposed_writes_outside_the_project_are_blocked(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    clear_pending_writes()

    assert propose_write_impl("../outside.txt", "x").startswith("Blocked")
    assert propose_write_impl(str(tmp_path / "sibling.txt"), "x").startswith("Blocked")
    assert propose_write_impl("src/ok.py", "print(1)\n").startswith("Proposed")
    assert [Path(w["file_path"]).name for w in get_pending_writes()] == ["ok.py"]
    clear_pending_writes()


# Snapshot lookup


def test_snapshot_lookup_matches_on_directory_boundary(tmp_path):
    sync = SyncManager(db_dir=str(tmp_path / ".codetrace"))
    root = tmp_path / "proj"
    for rel in ("src/data.py", "src/my_file.py", "src/myXfile.py", "a/util.py", "b/util.py"):
        sync.upsert_file_snapshot(str(root / rel), f"# {rel}\n", file_hash="h")

    assert sync.get_file_snapshot("a.py") is None                      # not data.py
    assert sync.get_file_snapshot(str(Path("src/data.py")))["content"] == "# src/data.py\n"
    # '_' is a LIKE wildcard; it must not match myXfile.py.
    assert sync.get_file_snapshot("my_file.py")["content"] == "# src/my_file.py\n"
    assert sync.get_file_snapshot("util.py") is None                   # ambiguous
    assert sync.get_file_snapshot(str(Path("b/util.py")))["content"] == "# b/util.py\n"


# Output normalizer


def test_normalizer_leaves_code_comments_alone():
    raw = "Here:\n```python\nimport os\n# load config\nx = 1\n```\n# Heading\ntext"
    out = OutputNormalizer().normalize(raw)
    assert "import os\n# load config\nx = 1" in out
    assert "\n\n## Heading\n\ntext" in out


# Source decoding


def test_non_utf8_source_is_still_parsed(tmp_path):
    src = tmp_path / "legacy.py"
    src.write_bytes("def caf\xe9_menu():\n    return 'na\xefve'\n".encode("latin-1"))
    result = GraphOrchestrator().extract_from_file(str(src))
    assert [s["name"] for s in result["symbols"]] == ["café_menu"]


# Private config writes


def test_config_is_written_privately_and_atomically(tmp_path):
    path = tmp_path / "config.json"
    write_private_json(path, {"api_key": "sk-test"})
    write_private_json(path, {"api_key": "sk-new"})

    assert json.loads(path.read_text()) == {"api_key": "sk-new"}
    assert [p.name for p in tmp_path.iterdir()] == ["config.json"]  # no temp leftovers
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


# MCP registration


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


SERVER = {"codetrace": {"command": "python", "args": ["/x/codetrace_mcp/server.py", "--project", "/p"]}}


def test_mcp_registration_is_project_scoped_and_merges(tmp_path, fake_home):
    project = tmp_path / "proj"
    (project / ".cursor").mkdir(parents=True)
    (project / ".cursor" / "mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))

    register_mcp(SERVER, workspace_dir=project)

    cursor = json.loads((project / ".cursor" / "mcp.json").read_text())
    assert set(cursor["mcpServers"]) == {"other", "codetrace"}
    assert "codetrace" in json.loads((project / ".mcp.json").read_text())["mcpServers"]
    vscode = json.loads((project / ".vscode" / "mcp.json").read_text())
    assert vscode["servers"]["codetrace"]["type"] == "stdio"
    assert not (fake_home / ".cursor").exists()  # nothing global written


def test_mcp_registration_never_clobbers_unparseable_config(tmp_path, fake_home):
    project = tmp_path / "proj"
    (project / ".vscode").mkdir(parents=True)
    jsonc = '{\n  // my servers\n  "servers": {"mine": {"type": "stdio", "command": "y"}}\n}\n'
    (project / ".vscode" / "mcp.json").write_text(jsonc)

    results = register_mcp(SERVER, workspace_dir=project)

    assert (project / ".vscode" / "mcp.json").read_text() == jsonc
    assert any("VS Code" in r and "untouched" in r for r in results)


def test_mcp_registration_removes_only_our_legacy_global_entry(tmp_path, fake_home):
    legacy = fake_home / ".cursor" / "mcp.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"mcpServers": {**SERVER, "keep": {"command": "z"}}}))

    register_mcp(SERVER, workspace_dir=tmp_path / "proj")

    assert json.loads(legacy.read_text())["mcpServers"] == {"keep": {"command": "z"}}


# `codetrace mcp` keeps stdout clean


def test_mcp_command_writes_nothing_to_stdout(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", "from src.cli.main import app; app()", "mcp", str(tmp_path), "--port", "9"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=300,
    )
    assert proc.returncode == 1               # not indexed
    assert proc.stdout == ""                  # stdout is reserved for JSON-RPC
    assert "No .codetrace directory" in proc.stderr
    assert "--port/--host are ignored" in proc.stderr


# Remote repositories are kept and updated in place


def test_repo_cache_dir_is_stable_and_contained(fake_home):
    base = fake_home / ".codetrace" / "repos"
    assert repo_cache_dir("https://github.com/user/repo.git") == base / "github.com" / "user" / "repo"
    assert repo_cache_dir("git@gitlab.com:grp/sub/proj.git", "dev") == base / "gitlab.com" / "grp" / "sub" / "proj@dev"
    evil = repo_cache_dir("https://host/../../etc.git", "../x")
    assert evil.resolve().is_relative_to(base.resolve())


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_reindexing_a_url_updates_the_clone_and_keeps_the_index(tmp_path):
    def git(cwd, *args):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q")
    git(origin, "config", "user.email", "t@example.com")
    git(origin, "config", "user.name", "t")
    (origin / "a.py").write_text("v1\n")
    git(origin, "add", ".")
    git(origin, "commit", "-q", "-m", "one")

    dest = tmp_path / "cache" / "repo"
    clone_repo(origin.as_uri(), dest=dest)
    (dest / ".codetrace").mkdir()
    (dest / ".codetrace" / "marker").write_text("index")

    (origin / "a.py").write_text("v2\n")
    git(origin, "commit", "-q", "-am", "two")
    assert clone_repo(origin.as_uri(), dest=dest) == dest

    assert (dest / "a.py").read_text() == "v2\n"
    assert (dest / ".codetrace" / "marker").read_text() == "index"


def test_clone_rejects_option_like_branch(tmp_path):
    with pytest.raises(RuntimeError):
        clone_repo("https://example.com/r.git", branch="--upload-pack=touch x", dest=tmp_path / "r")
