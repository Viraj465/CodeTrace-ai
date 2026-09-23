"""
Codetrace Agent Tools: Shared core logic + OpenAI-compatible tool schemas.

The _impl functions contain the actual logic and can be called by
both the httpx-based agent (CLI) and the MCP server (IDE integration).
The LangChain @tool decorator has been replaced with plain JSON schemas.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

# No LangChain here — the tool schemas further down are just plain dicts.

# Relative imports so IDEs resolve them correctly and don't flag missing imports.
from ..graph.builder import CodeGraph
from ...backend.vector_store import VectorStore
from ..database.sync_manager import SyncManager




def search_codebase_impl(vector_store: VectorStore, query: str) -> str:
    """Search the indexed codebase for symbols semantically related to the query."""
    docs = vector_store.hybrid_search(query)
    if not docs:
        hint = inspect_index_impl(query=query, limit=20)
        return (
            "No semantic symbol results found.\n"
            "Use indexed file coverage below to refine your query:\n\n"
            f"{hint}"
        )

    blocks = []
    for i, doc in enumerate(docs, 1):
        file_path = doc.metadata.get("file_path", "unknown")
        symbol    = doc.metadata.get("qualified_name", doc.metadata.get("symbol_name", "unknown"))
        sym_type  = doc.metadata.get("type", "unknown")
        code      = doc.page_content

        blocks.append(
            f"--- Result {i} ---\n"
            f"File: {file_path}\n"
            f"Symbol: {symbol} ({sym_type})\n"
            f"Code:\n{code}\n"
        )
    return "\n".join(blocks)


def _resolve_symbol_or_error(graph: CodeGraph, symbol_id: str) -> tuple[str | None, str | None]:
    """
    Resolve a supplied symbol ID to one stored node ID.
    Returns (node_id, None) on success, or (None, message) when the ID is
    unknown or ambiguous so the tool can tell the model exactly what to retry.
    """
    matches = graph.resolve_symbol_id(symbol_id)
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, (
            f"Symbol '{symbol_id}' not found in the graph. Use the "
            "'filepath:qualified_name' form (e.g. 'src/app.py:Service.run'), "
            "or search_codebase to find the exact symbol."
        )
    shown = "\n".join(f"  - {m}" for m in matches[:15])
    more = f"\n  - ... and {len(matches) - 15} more" if len(matches) > 15 else ""
    return None, (
        f"Symbol '{symbol_id}' is ambiguous — {len(matches)} matches. "
        f"Retry with one of these IDs:\n{shown}{more}"
    )


def get_symbol_relations_impl(graph: CodeGraph, symbol_id: str) -> str:
    """Get structural relationships (callers + dependencies) of a symbol."""
    symbol_id, error = _resolve_symbol_or_error(graph, symbol_id)
    if error:
        return error

    callers      = graph.get_callers(symbol_id)
    dependencies = graph.get_dependencies(symbol_id)

    if not callers and not dependencies:
        return f"Symbol '{symbol_id}' is in the graph but has no callers or dependencies."

    lines = [f"Relations for: {symbol_id}\n"]
    if callers:
        lines.append("Called by:")
        for c in callers[:15]:
            lines.append(f"  ← {c}")
    if dependencies:
        lines.append("Calls / depends on:")
        for d in dependencies[:15]:
            lines.append(f"  → {d}")
    return "\n".join(lines)


def read_file_impl(file_path: str, max_lines: int = 200) -> str:
    """Read file contents from the index DB snapshot (no direct filesystem reads)."""
    db_dir = Path.cwd() / ".codetrace"
    if not db_dir.exists():
        return "Index DB not found. Run 'codetrace index .' first."

    try:
        sync = SyncManager(db_dir=str(db_dir))
        
        # Safeguard against path traversal
        p = Path(file_path)
        if not p.is_absolute():
            p = Path.cwd() / file_path
        root = Path.cwd().resolve()
        resolved = p.resolve()
        # Use is_relative_to (not str.startswith) so a sibling dir like
        # 'project-secret' can't pass the check for root 'project'.
        if not resolved.is_relative_to(root):
            return f"Blocked: cannot read outside project root ({root})"

        # Look the snapshot up by the SAME validated path used by the check
        # above — not the raw input. Snapshots are keyed by absolute path at
        # index time, so try that exact key first, then the project-relative
        # path (which get_file_snapshot matches on a directory boundary).
        snap = sync.get_file_snapshot(str(resolved)) or sync.get_file_snapshot(
            str(resolved.relative_to(root))
        )
    except Exception as e:
        return f"Error reading snapshot DB: {e}"

    if not snap:
        return (
            f"File not found in indexed snapshots: {file_path}. "
            "Re-index if this file is new or was excluded."
        )

    lines = snap["content"].splitlines()
    # Two independent truncations, both worth surfacing:
    #   snapshot_truncated  → the stored snapshot was capped at index time (size limit)
    #   output_truncated    → this read is capped to max_lines for context budget
    output_truncated = len(lines) > max_lines
    preview = "\n".join(lines[:max_lines])
    header = f"--- {Path(snap['filepath']).name} ({snap['line_count']} lines, from index DB) ---\n"
    footer_lines = []
    if snap["is_truncated"]:
        footer_lines.append("... (snapshot truncated during indexing due to size limit)")
    if output_truncated:
        footer_lines.append(
            f"... (output truncated, showing first {max_lines} of {len(lines)} lines)"
        )
    footer = f"\n{chr(10).join(footer_lines)}" if footer_lines else ""
    return header + preview + footer


def inspect_index_impl(query: str = "", limit: int = 50) -> str:
    """
    Inspect what is actually available in the index DB (file manifest + metadata).
    """
    db_dir = Path.cwd() / ".codetrace"
    if not db_dir.exists():
        return "Index DB not found. Run 'codetrace index .' first."

    try:
        sync = SyncManager(db_dir=str(db_dir))
        # Support both slash styles and simple glob-like inputs.
        normalized = (query or "").strip()
        variants = {normalized}
        if normalized:
            variants.add(normalized.replace("/", "\\"))
            variants.add(normalized.replace("\\", "/"))
            variants.add(normalized.strip("/\\"))
            if any(ch in normalized for ch in "*?[]"):
                variants.add(re.sub(r"[*?\[\]]", "", normalized).strip("/\\"))

        files = []
        seen = set()
        for q in [v for v in variants if v] or [""]:
            for fp in sync.list_indexed_files(query=q, limit=limit):
                if fp not in seen:
                    seen.add(fp)
                    files.append(fp)
                if len(files) >= max(1, min(limit, 1000)):
                    break
            if len(files) >= max(1, min(limit, 1000)):
                break
        project_root = sync.get_metadata("project_root", "unknown")
        supported_count = sync.get_metadata("supported_file_count", "unknown")
        tracked_count = sync.get_metadata("tracked_snapshot_count", "unknown")
    except Exception as e:
        return f"Error inspecting index DB: {e}"

    if not files:
        base = (
            "No indexed files found"
            if not query
            else f"No indexed files found matching '{query}'"
        )
        return (
            f"{base}.\n"
            f"Project root: {project_root}\n"
            f"Supported files indexed: {supported_count}\n"
            f"Tracked text snapshots: {tracked_count}"
        )

    lines = [
        "Index inspection (DB-only):",
        f"Project root: {project_root}",
        f"Supported files indexed: {supported_count}",
        f"Tracked text snapshots: {tracked_count}",
        f"Showing {len(files)} file(s):",
    ]
    lines.extend([f"- {fp}" for fp in files])
    return "\n".join(lines)


def analyze_impact_impl(graph: CodeGraph, symbol_id: str) -> str:
    """Find all downstream dependents of a symbol (blast radius)."""
    symbol_id, error = _resolve_symbol_or_error(graph, symbol_id)
    if error:
        return error

    dependents = graph.get_all_downstream_dependents(symbol_id)
    if not dependents:
        return (
            f"No downstream dependents found for '{symbol_id}' — "
            f"nothing in the index calls it."
        )

    lines = [f"Impact analysis for: {symbol_id}",
             f"Total affected symbols: {len(dependents)}\n"]

    current_depth = 0
    for dep in dependents:
        if dep["depth"] != current_depth:
            current_depth = dep["depth"]
            lines.append(f"\n── Depth {current_depth} ({'direct' if current_depth == 1 else 'transitive'}) ──")
        lines.append(f"  {dep['type']:>10}  {dep['symbol']}")
        lines.append(f"             in {dep['file']}")

    return "\n".join(lines)


def write_file_impl(file_path: str, content: str, project_root: str | None = None) -> str:
    """Write content to a file. Creates parent directories if needed.
    Used directly by MCP server (IDEs have their own confirmation UX)."""
    p = Path(file_path)
    if not p.is_absolute():
        p = Path.cwd() / file_path

    # Block writes outside the project root.
    if project_root:
        root = Path(project_root).resolve()
        # is_relative_to avoids the str.startswith prefix-collision flaw
        # (e.g. 'project-secret' matching root 'project').
        if not p.resolve().is_relative_to(root):
            return f"Blocked: cannot write outside project root ({root})"

    # Block binary files.
    blocked_extensions = {".exe", ".dll", ".so", ".pyc", ".pyo", ".class", ".o"}
    if p.suffix.lower() in blocked_extensions:
        return f"Blocked: cannot write binary file ({p.suffix})"

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {p}"
    except Exception as e:
        return f"Error writing file: {e}"

_pending_writes: list[dict] = []


def get_pending_writes() -> list[dict]:
    """Return all queued write proposals."""
    return list(_pending_writes)


def clear_pending_writes() -> None:
    """Clear the pending writes queue."""
    _pending_writes.clear()


def replace_pending_writes(pending: list[dict]) -> None:
    """Replace pending write queue (used for batch processing across turns)."""
    _pending_writes.clear()
    _pending_writes.extend(pending)


def propose_write_impl(file_path: str, content: str) -> str:
    """Generate a diff preview and queue the write for user confirmation.
    Does NOT write to disk — the CLI asks the user first."""
    import difflib

    p = Path(file_path)
    if not p.is_absolute():
        p = Path.cwd() / file_path

    # Refuse to even propose a write outside the project. The approval prompt is
    # a second line of defence, not the only one — the model reads repo content
    # it didn't write, and that content can try to steer it at ~/.bashrc & co.
    root = Path.cwd().resolve()
    if not p.resolve().is_relative_to(root):
        return f"Blocked: cannot write outside project root ({root})"

    # Perform safety checks.
    blocked_extensions = {".exe", ".dll", ".so", ".pyc", ".pyo", ".class", ".o"}
    if p.suffix.lower() in blocked_extensions:
        return f"Blocked: cannot write binary file ({p.suffix})"

    # Read the existing file for the diff, or leave empty if it is a new file.
    if p.exists() and p.is_file():
        try:
            old_content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            old_content = ""
    else:
        old_content = ""

    # Generate a unified diff.
    old_lines = old_content.splitlines(keepends=True)
    new_lines = content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{p.name}",
        tofile=f"b/{p.name}",
        lineterm="",
    ))

    # Queue the pending write.
    _pending_writes.append({
        "file_path": str(p),
        "content": content,
        "diff": diff,
        "is_new_file": not p.exists(),
    })

    # Return a summary to the agent.
    if not p.exists():
        return (
            f"Proposed: CREATE new file {p.name} ({len(content)} chars). "
            f"Awaiting user confirmation."
        )

    additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    return (
        f"Proposed: MODIFY {p.name} (+{additions} -{deletions} lines). "
        f"Awaiting user confirmation."
    )


def summarize_file_blast_radius(graph: CodeGraph, file_path: str, max_files: int = 12) -> str:
    """
    Given a file about to be edited, list the OTHER files that contain symbols
    transitively depending on this file's symbols — i.e. the files that may also
    need coordinated changes.

    Returned as a compact, agent-readable block so the model proposes edits for
    the whole blast radius in one batch instead of one file at a time.
    """
    target = Path(file_path).as_posix().lstrip("./")
    target_name = Path(file_path).name

    affected: dict[str, int] = {}
    matched_any = False
    for node_id, data in graph.direct_graph.nodes(data=True):
        node_file = data.get("file", "")
        if not node_file:
            continue
        node_posix = Path(node_file).as_posix()
        # Precise match: same relative path (separator-normalized) or path suffix.
        if not (node_posix == target or node_posix.endswith(f"/{target}")):
            continue
        matched_any = True
        for dep in graph.get_all_downstream_dependents(node_id):
            dep_file = dep.get("file")
            if dep_file and Path(dep_file).as_posix() != node_posix:
                affected[dep_file] = affected.get(dep_file, 0) + 1

    if not matched_any:
        # File not indexed as a symbol owner (e.g. brand-new file) — nothing to report.
        return ""
    if not affected:
        return "Blast radius: no downstream dependent files found for this file."

    ranked = sorted(affected.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = ["Blast radius — dependent files that may also need coordinated edits:"]
    for f, count in ranked[:max_files]:
        lines.append(f"  - {f} ({count} dependent symbol(s))")
    if len(ranked) > max_files:
        lines.append(f"  - ... and {len(ranked) - max_files} more file(s)")
    lines.append(
        "Review each with analyze_impact / get_symbol_relations and, if they "
        "need changes, propose write_file for them in THIS batch."
    )
    return "\n".join(lines)


def git_diff_impl(target: str = "HEAD") -> str:
    """Run git diff on the project (cwd) and return the output.

    Always scoped to the current project directory — there is deliberately no
    caller- or model-supplied path, so git can't be pointed outside the project.

    target can be:
      - "HEAD"       → all uncommitted changes (staged + unstaged)
      - "--staged"   → staged changes
      - "HEAD~1"     → diff from last commit
      - a branch name → diff against that branch
    """
    try:
        cmd = ["git", "-C", str(Path.cwd().resolve()), "diff"]

        # Block malicious flag injections.
        if target != "--staged" and target.startswith("-"):
            return f"Blocked: invalid git diff target '{target}'"

        # The revision goes BEFORE "--". Anything after "--" is a pathspec, so
        # `git diff -- HEAD` would filter to a file named "HEAD" and always come
        # back empty. The trailing "--" also stops git from guessing whether an
        # ambiguous target is a path.
        cmd.append(target)
        cmd.append("--")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"git diff failed: {result.stderr.strip()}"

        diff = result.stdout.strip()
        if not diff:
            return "No changes found."

        # Limit output length to prevent context flooding.
        lines = diff.splitlines()
        if len(lines) > 300:
            return "\n".join(lines[:300]) + f"\n\n... (truncated, {len(lines)} total lines)"
        return diff

    except subprocess.TimeoutExpired:
        return "git diff timed out."
    except FileNotFoundError:
        return "git is not installed or not in PATH."



# OpenAI-compatible tool schemas.
# These plain JSON schemas stand in for the old LangChain decorators and work
# with any standard chat-completions API.


def create_tool_schemas() -> list[dict]:
    """Build OpenAI-compatible tool schemas for the chat completions API."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": (
                    "Search the indexed codebase for code symbols semantically related to the query. "
                    "Use this tool whenever you need to find relevant functions, classes, or code snippets. "
                    "The query should be a natural language description of what you are looking for. "
                    "Returns matching code snippets with file paths and symbol names."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query."}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_index",
                "description": (
                    "Inspect index DB coverage and list indexed files. "
                    "Use this before high-level architecture questions to confirm what files "
                    "are available in the index. query is optional path/keyword filter."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Optional path or keyword filter."},
                        "limit": {"type": "integer", "description": "Max files to return (default: 50)."},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_symbol_relations",
                "description": (
                    "Get the structural relationships of a code symbol in the dependency graph. "
                    "Use this to understand what a symbol calls (dependencies) and what calls it (callers). "
                    "The symbol_id is typically 'filepath:qualified_name', e.g. "
                    "'src/backend/vector_store.py:VectorStore.add_symbol'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol_id": {"type": "string", "description": "Symbol ID in 'filepath:qualified_name' format."}
                    },
                    "required": ["symbol_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read the full contents of a source file by its path. "
                    "Use this when you need to see imports, constants, or full context that "
                    "semantic search only partially returned. "
                    "Returns indexed file content from the DB snapshot."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative or absolute file path."},
                        "max_lines": {"type": "integer", "description": "Max lines to return (default: 200)."},
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_impact",
                "description": (
                    "Find all downstream dependents of a symbol — everything that would "
                    "be affected if this symbol changes. "
                    "Use this for impact analysis, e.g. 'If I change function X, what breaks?'"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol_id": {"type": "string", "description": "Symbol ID in 'filepath:qualified_name' format."}
                    },
                    "required": ["symbol_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Propose a file change for user approval. The change will NOT be applied "
                    "until the user confirms it. Use this to fix bugs, refactor code, or generate "
                    "new files. Content should be the COMPLETE file content (not a diff)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path to the file."},
                        "content": {"type": "string", "description": "Complete file content to write."},
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_diff",
                "description": (
                    "Show git diff for the current project. "
                    "target can be: 'HEAD' (all uncommitted changes), '--staged', 'HEAD~1' (last commit), "
                    "or a branch name to compare against."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Diff target (default: 'HEAD'). Options: 'HEAD' (all uncommitted changes), '--staged', 'HEAD~1', or a branch name."},
                    },
                    "required": [],
                },
            },
        },
    ]


def create_anthropic_tool_schemas() -> list[dict]:
    """
    Convert OpenAI tool schemas to Anthropic format.
    Anthropic uses 'input_schema' instead of 'parameters' and doesn't wrap
    in {"type": "function", "function": {...}}.
    """
    return [
        {
            "name": schema["function"]["name"],
            "description": schema["function"]["description"],
            "input_schema": schema["function"]["parameters"],
        }
        for schema in create_tool_schemas()
    ]


def dispatch_tool(
    tool_name: str,
    tool_args: dict,
    vector_store: VectorStore,
    graph: CodeGraph,
) -> str:
    """
    Run a tool by name, routing to its _impl function and returning the string
    it produces. Same job the old LangChain @tool closures did, minus the framework.
    """
    def _propose_write_with_radius() -> str:
        result = propose_write_impl(tool_args["file_path"], tool_args["content"])
        # Tack the blast radius onto the result so the model sees the dependent
        # files and can propose their edits in the same turn instead of one by one.
        try:
            radius = summarize_file_blast_radius(graph, tool_args["file_path"])
            if radius:
                result = f"{result}\n\n{radius}"
        except Exception:
            pass
        return result

    dispatchers = {
        "search_codebase":      lambda: search_codebase_impl(vector_store, tool_args["query"]),
        "get_symbol_relations": lambda: get_symbol_relations_impl(graph, tool_args["symbol_id"]),
        "read_file":            lambda: read_file_impl(tool_args["file_path"], tool_args.get("max_lines", 200)),
        "inspect_index":        lambda: inspect_index_impl(tool_args.get("query", ""), tool_args.get("limit", 50)),
        "analyze_impact":       lambda: analyze_impact_impl(graph, tool_args["symbol_id"]),
        "write_file":           _propose_write_with_radius,
        "git_diff":             lambda: git_diff_impl(tool_args.get("target", "HEAD")),
    }

    dispatcher = dispatchers.get(tool_name)
    if dispatcher is None:
        return f"Unknown tool: {tool_name}"

    try:
        return dispatcher()
    except Exception as e:
        return f"Tool '{tool_name}' failed: {e}"
