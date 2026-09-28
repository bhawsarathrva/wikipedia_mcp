"""Conditional edge functions for the LangGraph workflow."""

import logging
from typing import Literal
from langchain_core.messages import AIMessage

from app.graph.state import AgentState

logger = logging.getLogger("app.graph.edges")

def should_use_tool(state: AgentState) -> Literal["mcp_tool", "final_answer"]:
    """Conditional router determining whether to execute MCP tools or finalize answer.

    Evaluates the most recent AIMessage in state['messages']:
    - If the LLM generated tool_calls: routes to 'mcp_tool' node.
    - If the LLM generated direct textual output: routes to 'final_answer' node.

    Args:
        state: The current AgentState.

    Returns:
        Next node name: 'mcp_tool' or 'final_answer'.
    """
    messages = state.get("messages", [])
    if not messages:
        logger.warning("[EDGES] No messages found in state. Defaulting to final_answer.")
        return "final_answer"

    last_message = messages[-1]

    # Check if the last message is an AIMessage containing tool calls
    if isinstance(last_message, AIMessage) and bool(last_message.tool_calls):
        tool_names = [tc.get("name") for tc in last_message.tool_calls]
        logger.info(f"[EDGES] Tool decision: YES -> Routing to 'mcp_tool' for tools: {tool_names}")
        return "mcp_tool"

    logger.info("[EDGES] Tool decision: NO -> Routing to 'final_answer'")
    return "final_answer"
