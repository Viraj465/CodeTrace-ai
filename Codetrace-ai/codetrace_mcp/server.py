"""
Codetrace MCP Server: Exposes codebase analysis tools to AI-powered IDEs
(Cursor, VS Code, Claude Desktop, Windsurf, etc.) via the Model Context Protocol.

It leans on the exact same core logic as the CLI agent — nothing is duplicated.

Usage:
    python codetrace_mcp/server.py                          # stdio (for IDE integration)
    python codetrace_mcp/server.py --project /path/to/repo  # specify project root
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Quiet the HuggingFace logs, same as the CLI does.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["TOKENIZERS_PARALLELISM"]        = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from mcp.server.fastmcp import FastMCP as MCPServer
except ImportError:
    from mcp.server import MCPServer

# This file is an entry-point script, not part of the src/ package. It runs with
# the project root on sys.path (from the pyproject scripts entry or direct
# invocation), so the absolute 'src.*' imports below are deliberate. Don't switch
# them to relative imports — those only work from inside a package.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.agents.tools import (
    search_codebase_impl,
    inspect_index_impl,
    get_symbol_relations_impl,
    read_file_impl,
    analyze_impact_impl,
    write_file_impl,
    git_diff_impl,
)
from src.backend.vector_store import VectorStore, VectorStoreConfig
from src.core.graph.builder import CodeGraph

logger = logging.getLogger("codetrace.mcp")


# Set once at startup and reused for the life of the server.
vector_store: Optional[VectorStore] = None
graph: Optional[CodeGraph] = None

# The MCP server itself.
app = MCPServer("codetrace")


@app.tool(
    description=(
        "Search the indexed codebase for code symbols semantically "
        "related to the query. Returns matching code snippets with "
        "file paths and symbol names."
    )
)
def search_codebase(query: str) -> str:
    """Search the indexed codebase for code symbols semantically related to the query."""
    if vector_store is None or graph is None:
        return "Error: Codetrace server not initialized. Index a project first."
    return search_codebase_impl(vector_store, query)


@app.tool(
    description=(
        "Inspect index DB coverage and list indexed files. "
        "Use before architecture analysis to confirm available evidence."
    )
)
def inspect_index(query: str = "", limit: int = 50) -> str:
    """Inspect index DB coverage and list indexed files."""
    if vector_store is None or graph is None:
        return "Error: Codetrace server not initialized. Index a project first."
    return inspect_index_impl(query=query, limit=limit)


@app.tool(
    description=(
        "Get structural relationships of a code symbol: what calls it "
        "and what it depends on. Symbol ID format: 'filepath:qualified_name'."
    )
)
def get_symbol_relations(symbol_id: str) -> str:
    """Get structural relationships of a code symbol: what calls it and what it depends on."""
    if vector_store is None or graph is None:
        return "Error: Codetrace server not initialized. Index a project first."
    return get_symbol_relations_impl(graph, symbol_id)


@app.tool(
    description=(
        "Read the full contents of a source file by path. "
        "Use when you need imports, constants, or full file context."
    )
)
def read_file(file_path: str, max_lines: int = 200) -> str:
    """Read the full contents of a source file by path."""
    if vector_store is None or graph is None:
        return "Error: Codetrace server not initialized. Index a project first."
    return read_file_impl(file_path, max_lines)


@app.tool(
    description=(
        "Find all downstream dependents of a symbol — the blast radius "
        "if this symbol changes. Returns affected symbols by depth."
    )
)
def analyze_impact(symbol_id: str) -> str:
    """Find all downstream dependents of a symbol (blast radius)."""
    if vector_store is None or graph is None:
        return "Error: Codetrace server not initialized. Index a project first."
    return analyze_impact_impl(graph, symbol_id)


@app.tool(
    description=(
        "Write content to a file, creating it if needed or overwriting. "
        "Use for bug fixes, refactoring, or generating new files."
    )
)
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating it if needed or overwriting."""
    if vector_store is None or graph is None:
        return "Error: Codetrace server not initialized. Index a project first."
    return write_file_impl(
        file_path,
        content,
        project_root=str(Path.cwd().resolve()),
    )


@app.tool(
    description=(
        "Show git diff for the project. Use for PR reviews or "
        "understanding recent changes."
    )
)
def git_diff(target: str = "HEAD") -> str:
    """Show git diff for the project."""
    if vector_store is None or graph is None:
        return "Error: Codetrace server not initialized. Index a project first."
    return git_diff_impl(target)


def _init_stores(project_path: str) -> None:
    """Load VectorStore and CodeGraph from an indexed project."""
    global vector_store, graph

    resolved_path = Path(project_path).resolve()
    db_dir = resolved_path / ".codetrace"
    if not db_dir.exists():
        logger.error("No .codetrace directory found at %s", db_dir)
        logger.error("Run 'codetrace index .' on the project first.")
        sys.exit(1)

    os.chdir(resolved_path)

    logger.info("Loading Codetrace stores from: %s", db_dir)

    vs_config = VectorStoreConfig(persist_dir=str(db_dir / "chroma"))
    vector_store = VectorStore(config=vs_config)

    graph = CodeGraph()
    graph.db_path = db_dir / "graph_metadata.db"
    graph._init_db()
    graph.load_from_db()

    node_count = graph.direct_graph.number_of_nodes()
    logger.info("MCP server ready — %d symbols indexed.", node_count)


async def main(project_path: str = ".") -> None:
    """Run the Codetrace MCP server over stdio."""
    _init_stores(project_path)
    await app.run_stdio_async()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codetrace MCP Server")
    parser.add_argument(
        "--project", "-p",
        default=".",
        help="Path to the indexed project (must contain .codetrace/)",
    )
    args = parser.parse_args()

    import asyncio
    asyncio.run(main(args.project))
