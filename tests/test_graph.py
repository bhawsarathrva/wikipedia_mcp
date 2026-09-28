"""Tests for the LangGraph StateGraph workflow, routing edges, and conversation loop."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from app.graph.edges import should_use_tool
from app.graph.workflow import build_agent_graph


def test_should_use_tool_routing():
    """Verify routing decision logic."""
    # When tool calls are present -> mcp_tool
    msg_with_tool = AIMessage(
        content="",
        tool_calls=[{"name": "search_wikipedia", "args": {"query": "Einstein"}, "id": "1"}],
    )
    assert should_use_tool({"messages": [msg_with_tool]}) == "mcp_tool"

    # When no tool calls are present -> final_answer
    msg_without_tool = AIMessage(content="Hello! How can I help you?", tool_calls=[])
    assert should_use_tool({"messages": [msg_without_tool]}) == "final_answer"


@pytest.mark.asyncio
async def test_workflow_casual_conversation_no_tool():
    """Test scenario: Casual prompt 'Hello' produces direct answer without invoking tools."""
    # Mock LLM that returns direct text
    class MockDirectLLM:
        async def ainvoke(self, messages):
            return AIMessage(content="Hello! How can I assist your research today?", tool_calls=[])

    dummy_tool = StructuredTool.from_function(
        func=lambda q: f"Results for {q}",
        name="search_wikipedia",
        description="Search tool",
    )

    app = build_agent_graph(
        llm_with_tools=MockDirectLLM(),
        tools=[dummy_tool],
        checkpointer=MemorySaver(),
    )

    result = await app.ainvoke(
        {"question": "Hello"},
        config={"configurable": {"thread_id": "test_casual"}},
    )

    assert result["final_answer"] == "Hello! How can I assist your research today?"
    assert result["tool_used"] is False
    assert len(result["tool_results"]) == 0


@pytest.mark.asyncio
async def test_workflow_tool_calling_loop():
    """Test scenario: Factual question triggers tool call, tool executes, LLM synthesizes answer in loop."""
    call_count = 0

    class MockLoopLLM:
        async def ainvoke(self, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First pass: LLM decides to search Wikipedia
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "search_wikipedia",
                        "args": {"query": "Alan Turing"},
                        "id": "call_123",
                    }],
                )
            else:
                # Second pass: LLM has observed ToolMessage, generates final answer
                return AIMessage(
                    content=(
                        "Alan Turing was a British mathematician and computer scientist.\n\n"
                        "Source: https://en.wikipedia.org/wiki/Alan_Turing"
                    ),
                    tool_calls=[],
                )

    from app.agent.tools import SearchWikipediaSchema

    async def mock_search(query: str, limit: int = 5) -> str:
        return '{"results": [{"title": "Alan Turing", "url": "https://en.wikipedia.org/wiki/Alan_Turing", "snippet": "British mathematician"}]}'

    test_tool = StructuredTool(
        name="search_wikipedia",
        description="Search Wikipedia articles",
        func=lambda **kw: "",
        coroutine=mock_search,
        args_schema=SearchWikipediaSchema,
    )

    app = build_agent_graph(
        llm_with_tools=MockLoopLLM(),
        tools=[test_tool],
        checkpointer=MemorySaver(),
    )

    result = await app.ainvoke(
        {"question": "Who was Alan Turing?"},
        config={"configurable": {"thread_id": "test_tool_loop"}},
    )

    # Verify both LLM passes occurred (explicit loop)
    assert call_count == 2
    assert result["tool_used"] is True
    assert len(result["tool_results"]) == 1
    assert "Alan Turing was a British mathematician" in result["final_answer"]
    assert "https://en.wikipedia.org/wiki/Alan_Turing" in result["sources"]
