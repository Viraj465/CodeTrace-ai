"""
Codetrace MCP Server: Exposes codebase analysis tools to AI-powered IDEs
(Cursor, VS Code, Claude Desktop, Windsurf, etc.) via the Model Context Protocol.

Reuses the same core logic as the CLI agent — zero duplicated code.

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

# Silence HF logs (same as CLI)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["TOKENIZERS_PARALLELISM"]        = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import shared core logic (no duplication)
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


# Globals (initialized once on server startup)
vector_store: Optional[VectorStore] = None
graph: Optional[CodeGraph] = None

# MCP Server Definition
app = Server("codetrace")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise available tools to the connected IDE."""
    return [
        Tool(
            name="search_codebase",
            description=(
                "Search the indexed codebase for code symbols semantically "
                "related to the query. Returns matching code snippets with "
                "file paths and symbol names."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    }
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="inspect_index",
            description=(
                "Inspect index DB coverage and list indexed files. "
                "Use before architecture analysis to confirm available evidence."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional substring filter for file paths",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max file paths to return (default 50)",
                        "default": 50,
                    },
                },
            },
        ),
        Tool(
            name="get_symbol_relations",
            description=(
                "Get structural relationships of a code symbol: what calls it "
                "and what it depends on. Symbol ID format: 'filepath:qualified_name'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_id": {
                        "type": "string",
                        "description": "Symbol ID in 'filepath:name' format",
                    }
                },
                "required": ["symbol_id"],
            },
        ),
        Tool(
            name="read_file",
            description=(
                "Read the full contents of a source file by path. "
                "Use when you need imports, constants, or full file context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the source file",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Max lines to return (default 200)",
                        "default": 200,
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="analyze_impact",
            description=(
                "Find all downstream dependents of a symbol — the blast radius "
                "if this symbol changes. Returns affected symbols by depth."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_id": {
                        "type": "string",
                        "description": "Symbol ID in 'filepath:name' format",
                    }
                },
                "required": ["symbol_id"],
            },
        ),
        Tool(
            name="write_file",
            description=(
                "Write content to a file, creating it if needed or overwriting. "
                "Use for bug fixes, refactoring, or generating new files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content to write",
                    },
                },
                "required": ["file_path", "content"],
            },
        ),
        Tool(
            name="git_diff",
            description=(
                "Show git diff for the project. Use for PR reviews or "
                "understanding recent changes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Diff target: 'HEAD', '--staged', 'HEAD~1', or branch name",
                        "default": "HEAD",
                    }
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route MCP tool calls to shared core logic."""
    if vector_store is None or graph is None:
        return [TextContent(
            type="text",
            text="Error: Codetrace server not initialized. Index a project first.",
        )]

    if name == "search_codebase":
        result = search_codebase_impl(vector_store, arguments["query"])

    elif name == "inspect_index":
        result = inspect_index_impl(
            arguments.get("query", ""),
            arguments.get("limit", 50),
        )

    elif name == "get_symbol_relations":
        result = get_symbol_relations_impl(graph, arguments["symbol_id"])

    elif name == "read_file":
        result = read_file_impl(
            arguments["file_path"],
            arguments.get("max_lines", 200),
        )

    elif name == "analyze_impact":
        result = analyze_impact_impl(graph, arguments["symbol_id"])

    elif name == "write_file":
        result = write_file_impl(arguments["file_path"], arguments["content"])

    elif name == "git_diff":
        result = git_diff_impl(".", arguments.get("target", "HEAD"))

    else:
        result = f"Unknown tool: {name}"

    return [TextContent(type="text", text=result)]

def _init_stores(project_path: str) -> None:
    """Load VectorStore and CodeGraph from an indexed project."""
    global vector_store, graph

    db_dir = Path(project_path).resolve() / ".codetrace"
    if not db_dir.exists():
        logger.error("No .codetrace directory found at %s", db_dir)
        logger.error("Run 'codetrace index .' on the project first.")
        sys.exit(1)

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

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="Codetrace MCP Server")
    parser.add_argument(
        "--project", "-p",
        default=".",
        help="Path to the indexed project (must contain .codetrace/)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.project))
