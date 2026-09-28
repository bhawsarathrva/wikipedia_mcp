"""LangGraph StateGraph workflow assembler for the Wikipedia MCP Agent.

This module builds the compiled LangGraph workflow graph.

Workflow Topology:
------------------
       START
         │
         ▼
 ┌──────────────────┐
 │ receive_question │
 └────────┬─────────┘
          │
          ▼
   ┌─────────────┐
   │     llm     │◄─────────────────┐
   └──────┬──────┘                  │
          │                         │
     [should_use_tool?]             │
      ├── YES ──► ┌──────────┐     │
      │           │ mcp_tool │─────┘
      │           └──────────┘
      │           (Explicit Tool Loop)
      └── NO
          │
          ▼
  ┌──────────────┐
  │ final_answer │
  └──────┬───────┘
         │
         ▼
        END

WHY THE TOOL-CALLING LOOP IS NECESSARY:
---------------------------------------
When the LLM decides that it requires external factual knowledge (e.g., asking "Who was Alan Turing?"),
it generates a ToolCall rather than a text answer. The 'mcp_tool' node executes that tool call via
the official MCP Client, sending the request to the Wikipedia MCP Server and returning the result
as a ToolMessage.

The agent CANNOT finish here: the user has not received an answer, only raw tool results exist.
Therefore, the workflow MUST loop back to the 'llm' node (`mcp_tool` -> `llm`).
Upon receiving the ToolMessage containing the retrieved Wikipedia data, the LLM reads the facts,
synthesizes the final answer, and produces a standard text response.
On this second pass, `should_use_tool` evaluates to 'NO', directing the workflow to 'final_answer'
and terminating at END.
"""

import logging
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.edges import should_use_tool
from app.graph.nodes import (
    create_call_llm_node,
    create_final_answer_node,
    create_mcp_tool_node,
    create_receive_question_node,
)
from app.graph.state import AgentState

logger = logging.getLogger("app.graph.workflow")


def build_agent_graph(
    llm_with_tools: Any,
    tools: List[BaseTool],
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    logger.info("Assembling LangGraph StateGraph workflow...")

    # 1. Initialize StateGraph with typed schema
    workflow = StateGraph(AgentState)

    # 2. Add Graph Nodes
    workflow.add_node("receive_question", create_receive_question_node())
    workflow.add_node("llm", create_call_llm_node(llm_with_tools))
    workflow.add_node("mcp_tool", create_mcp_tool_node(tools))
    workflow.add_node("final_answer", create_final_answer_node())

    # 3. Add Edges
    # START -> receive_question -> llm
    workflow.add_edge(START, "receive_question")
    workflow.add_edge("receive_question", "llm")

    # llm -> conditional router (should_use_tool)
    workflow.add_conditional_edges(
        "llm",
        should_use_tool,
        {
            "mcp_tool": "mcp_tool",
            "final_answer": "final_answer",
        },
    )

    # EXPLICIT TOOL CALLING LOOP: mcp_tool -> llm
    # After MCP execution, pass the tool results back to the LLM to synthesize the final answer.
    workflow.add_edge("mcp_tool", "llm")

    # final_answer -> END
    workflow.add_edge("final_answer", END)

    # 4. Compile graph with conversation checkpointer
    memory = checkpointer if checkpointer is not None else MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("LangGraph StateGraph workflow successfully compiled with memory checkpointer.")
    return app
