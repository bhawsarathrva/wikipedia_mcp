"""FastAPI router providing /chat and /health endpoints."""

import logging
import uuid
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, ToolCallDetail

logger = logging.getLogger("app.api.routes")
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    thread_id = payload.thread_id or f"session_{uuid.uuid4().hex[:10]}"

    agent_workflow = getattr(request.app.state, "agent_workflow", None)
    if not agent_workflow:
        raise HTTPException(
            status_code=503,
            detail="Agent workflow is not initialized or still starting up.",
        )

    config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    initial_state: Dict[str, Any] = {
        "question": message,
        "tool_used": False,
        "tool_results": [],
        "tool_calls_info": [],
        "sources": [],
    }

    try:
        logger.info(f"Invoking LangGraph agent for thread_id={thread_id} with query='{message}'")
        final_state = await agent_workflow.ainvoke(initial_state, config=config)

        answer = final_state.get("final_answer", "")
        sources = final_state.get("sources", [])
        tool_used = final_state.get("tool_used", False)
        raw_tool_calls = final_state.get("tool_calls_info", [])

        tool_calls_detail = [
            ToolCallDetail(
                tool=tc.get("tool", "unknown"),
                query=tc.get("query"),
                arguments=tc.get("arguments", {}),
                result_snippet=tc.get("result_snippet"),
            )
            for tc in raw_tool_calls
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            tool_used=tool_used,
            tool_calls=tool_calls_detail,
            thread_id=thread_id,
        )

    except Exception as exc:
        logger.error(f"Error during agent invocation: {exc}", exc_info=True)
        # Friendly error message for UI without crashing
        error_msg = f"An error occurred while processing your request: {str(exc)}"
        if "API key" in str(exc) or "API_KEY" in str(exc):
            error_msg = (
                f"LLM API Authentication failed. Please check that {settings.LLM_PROVIDER.upper()}_API_KEY "
                "is properly configured in your .env file."
            )
        return ChatResponse(
            answer=f"⚠️ {error_msg}",
            sources=[],
            tool_used=False,
            tool_calls=[],
            thread_id=thread_id,
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Verifies service health, active LLM provider, and MCP client connection status."""
    mcp_client = getattr(request.app.state, "mcp_client", None)
    is_connected = mcp_client.is_connected if mcp_client else False

    return HealthResponse(
        status="healthy",
        mcp_server_connected=is_connected,
        llm_provider=settings.LLM_PROVIDER,
    )
