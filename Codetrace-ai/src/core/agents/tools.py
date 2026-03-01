"""
Codetrace Agent Tools: Shared core logic + LangChain tool wrappers.

The _impl functions contain the actual logic and can be called by
both the LangChain agent (CLI) and the MCP server (IDE integration).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from src.core.graph.builder import CodeGraph
from src.backend.vector_store import VectorStore
from src.core.database.sync_manager import SyncManager




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


def get_symbol_relations_impl(graph: CodeGraph, symbol_id: str) -> str:
    """Get structural relationships (callers + dependencies) of a symbol."""
    callers      = graph.get_callers(symbol_id)
    dependencies = graph.get_dependencies(symbol_id)

    if not callers and not dependencies:
        return f"Symbol '{symbol_id}' not found in the graph, or it has no relationships."

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
        snap = sync.get_file_snapshot(file_path)
    except Exception as e:
        return f"Error reading snapshot DB: {e}"

    if not snap:
        return (
            f"File not found in indexed snapshots: {file_path}. "
            "Re-index if this file is new or was excluded."
        )

    lines = snap["content"].splitlines()
    truncated = len(lines) > max_lines
    preview = "\n".join(lines[:max_lines])
    header = f"--- {Path(snap['filepath']).name} ({snap['line_count']} lines, from index DB) ---\n"
    footer_lines = []
    if snap["is_truncated"]:
        footer_lines.append("... (snapshot truncated during indexing due to size limit)")
    if truncated:
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
        # Support both slash styles and light glob-like inputs from users.
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
    dependents = graph.get_all_downstream_dependents(symbol_id)
    if not dependents:
        return (
            f"No downstream dependents found for '{symbol_id}'. "
            f"Either the symbol has no callers, or its ID is not in the graph."
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

    # Safety: block writes outside the project root
    if project_root:
        root = Path(project_root).resolve()
        if not str(p.resolve()).startswith(str(root)):
            return f"Blocked: cannot write outside project root ({root})"

    # Safety: block binary files
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

    # Safety checks (same as write_file_impl)
    blocked_extensions = {".exe", ".dll", ".so", ".pyc", ".pyo", ".class", ".o"}
    if p.suffix.lower() in blocked_extensions:
        return f"Blocked: cannot write binary file ({p.suffix})"

    # Read existing file for diff (empty if new file)
    if p.exists() and p.is_file():
        try:
            old_content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            old_content = ""
    else:
        old_content = ""

    # Generate unified diff
    old_lines = old_content.splitlines(keepends=True)
    new_lines = content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{p.name}",
        tofile=f"b/{p.name}",
        lineterm="",
    ))

    # Queue the pending write
    _pending_writes.append({
        "file_path": str(p),
        "content": content,
        "diff": diff,
        "is_new_file": not p.exists(),
    })

    # Return summary to the agent
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


def git_diff_impl(path: str = ".", target: str = "HEAD") -> str:
    """Run git diff and return the output.

    target can be:
      - "HEAD"       → unstaged changes
      - "--staged"   → staged changes
      - "HEAD~1"     → diff from last commit
      - a branch name → diff against that branch
    """
    try:
        cmd = ["git", "-C", str(Path(path).resolve()), "diff"]
        if target == "--staged":
            cmd.append("--staged")
        else:
            cmd.append(target)

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

        # Cap output to avoid flooding the LLM context
        lines = diff.splitlines()
        if len(lines) > 300:
            return "\n".join(lines[:300]) + f"\n\n... (truncated, {len(lines)} total lines)"
        return diff

    except subprocess.TimeoutExpired:
        return "git diff timed out."
    except FileNotFoundError:
        return "git is not installed or not in PATH."



# LangChain Tool Factory (thin wrappers for the CLI agent)


def create_tools(vector_store: VectorStore, graph: CodeGraph) -> list:
    """Build LangChain @tool wrappers bound to the given stores."""

    @tool
    def search_codebase(query: str) -> str:
        """Search the indexed codebase for code symbols semantically related to the query.
        Use this tool whenever you need to find relevant functions, classes, or code snippets.
        The query should be a natural language description of what you are looking for.
        Returns matching code snippets with file paths and symbol names."""
        return search_codebase_impl(vector_store, query)

    @tool
    def get_symbol_relations(symbol_id: str) -> str:
        """Get the structural relationships of a code symbol in the dependency graph.
        Use this to understand what a symbol calls (dependencies) and what calls it (callers).
        The symbol_id is typically in the format 'filepath:qualified_name',
        for example 'src/backend/vector_store.py:VectorStore.add_symbol'.
        Returns callers, dependencies, and ownership information."""
        return get_symbol_relations_impl(graph, symbol_id)

    @tool
    def read_file(file_path: str, max_lines: int = 200) -> str:
        """Read the full contents of a source file by its path.
        Use this when you need to see imports, constants, or full context that
        semantic search only partially returned.
        The path should be relative or absolute as stored in the index.
        Returns indexed file content from the DB snapshot (capped for safety)."""
        return read_file_impl(file_path, max_lines)

    @tool
    def inspect_index(query: str = "", limit: int = 50) -> str:
        """Inspect index DB coverage and list indexed files.
        Use this before high-level architecture questions to confirm what files
        are available in the index. query is optional path/keyword filter."""
        return inspect_index_impl(query, limit)

    @tool
    def analyze_impact(symbol_id: str) -> str:
        """Find all downstream dependents of a symbol — everything that would
        be affected if this symbol changes.
        Use this for impact analysis, e.g. 'If I change function X, what breaks?'
        The symbol_id format is 'filepath:qualified_name'.
        Returns a list of affected symbols grouped by depth of dependency."""
        return analyze_impact_impl(graph, symbol_id)

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Propose a file change for user approval. The change will NOT be applied
        until the user confirms it. Use this to fix bugs, refactor code, or generate
        new files. The file_path should be relative to the project root.
        The content should be the COMPLETE file content (not a diff)."""
        return propose_write_impl(file_path, content)

    @tool
    def git_diff(target: str = "HEAD") -> str:
        """Show git diff for the current project.
        Use this to review recent changes, understand what was modified, or do PR reviews.
        target can be: 'HEAD' (unstaged), '--staged' (staged), 'HEAD~1' (last commit),
        or a branch name to compare against."""
        return git_diff_impl(".", target)

    return [search_codebase, inspect_index, get_symbol_relations, read_file, analyze_impact, write_file, git_diff]
