"""Unit tests for agent prompts, tool wrapping, and LLM configuration."""

import pytest
from app.agent.prompts import load_system_prompt
from app.agent.tools import (
    KNOWN_SCHEMAS,
    SearchWikipediaSchema,
    create_mcp_tool_wrapper,
)
from app.mcp.client import WikipediaMCPClient


def test_load_system_prompt():
    """Verify system prompt loads from file or fallback."""
    prompt = load_system_prompt()
    assert "Wikipedia research assistant" in prompt
    assert "TOOL SELECTION GUIDELINES" in prompt
    assert "ANSWER FORMATTING RULES" in prompt


def test_known_tool_schemas():
    """Verify predefined schemas exist and validate correctly."""
    assert "search_wikipedia" in KNOWN_SCHEMAS
    schema = SearchWikipediaSchema(query="Quantum", limit=3)
    assert schema.query == "Quantum"
    assert schema.limit == 3


@pytest.mark.asyncio
async def test_tool_wrapper_dispatch(monkeypatch):
    """Verify that StructuredTool wrapper forwards execution to MCP client."""
    client = WikipediaMCPClient()

    class MockMCPTool:
        name = "search_wikipedia"
        description = "Search Wikipedia test tool"

    # Mock call_tool on MCP client
    called_args = {}

    async def mock_call_tool(tool_name, arguments):
        called_args["name"] = tool_name
        called_args["arguments"] = arguments
        return '{"results": [{"title": "Test Article", "url": "https://en.wikipedia.org/wiki/Test"}]}'

    monkeypatch.setattr(client, "call_tool", mock_call_tool)

    wrapped_tool = create_mcp_tool_wrapper(MockMCPTool(), client)
    assert wrapped_tool.name == "search_wikipedia"

    result = await wrapped_tool.ainvoke({"query": "Computer Science", "limit": 4})
    assert called_args["name"] == "search_wikipedia"
    assert called_args["arguments"]["query"] == "Computer Science"
    assert called_args["arguments"]["limit"] == 4
    assert "Test Article" in result
