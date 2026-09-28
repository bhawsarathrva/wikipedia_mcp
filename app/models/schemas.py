from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolCallDetail(BaseModel):
    tool: str = Field(..., description="Name of the MCP tool called.")
    query: Optional[str] = Field(None, description="Extracted search query or article title if applicable.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments sent to the MCP tool.")
    result_snippet: Optional[str] = Field(None, description="Brief snippet or status of the tool execution.")


class ChatRequest(BaseModel):
    """Incoming chat message request schema."""
    message: str = Field(
        ...,
        description="User question or input message.",
        examples=["Who was Alan Turing?", "What is quantum mechanics?"],
    )
    thread_id: Optional[str] = Field(
        None,
        description="Optional session / thread ID for multi-turn conversation memory.",
        examples=["session-abc-123"],
    )


class ChatResponse(BaseModel):
    """Outgoing chat response schema."""
    answer: str = Field(..., description="Final synthesized answer from the agent.")
    sources: List[str] = Field(
        default_factory=list,
        description="List of Wikipedia canonical URLs referenced in the answer.",
        examples=[["https://en.wikipedia.org/wiki/Alan_Turing"]],
    )
    tool_used: bool = Field(
        default=False,
        description="Whether an MCP tool was executed during this turn.",
    )
    tool_calls: List[ToolCallDetail] = Field(
        default_factory=list,
        description="List of tool executions with arguments and snippets for UI visualization.",
    )
    thread_id: str = Field(..., description="Active session thread ID.")


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(default="healthy", description="Application health status.")
    mcp_server_connected: bool = Field(
        default=False,
        description="Whether the Wikipedia MCP server is reachable.",
    )
    llm_provider: str = Field(..., description="Active LLM provider (gemini or openai).")
