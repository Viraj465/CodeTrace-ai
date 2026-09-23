import pytest
import codetrace_mcp.server as mcp_server


@pytest.mark.asyncio
async def test_mcp_tools_registered():
    """Verify that all 7 MCP tools are defined and registered on the server."""
    tools = await mcp_server.app.list_tools()
    assert len(tools) == 7
    tool_names = {t.name for t in tools}
    expected_names = {
        "search_codebase",
        "inspect_index",
        "get_symbol_relations",
        "read_file",
        "analyze_impact",
        "write_file",
        "git_diff",
    }
    assert tool_names == expected_names


@pytest.mark.asyncio
async def test_mcp_handle_tool_call_uninitialized():
    """Verify handle_tool_call returns graceful error message when stores not initialized."""
    mcp_server.vector_store = None
    mcp_server.graph = None
    result = mcp_server.search_codebase("test")
    assert "not initialized" in result
