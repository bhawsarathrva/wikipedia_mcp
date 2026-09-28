import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.agent.agent import bind_tools_to_llm, get_llm
from app.agent.tools import build_langchain_tools_from_mcp
from app.api.routes import router as api_router
from app.config import settings
from app.graph.workflow import build_agent_graph
from app.mcp.client import WikipediaMCPClient

# Configure root logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to orchestrate MCP client and LangGraph initialization."""
    logger.info("==================================================")
    logger.info("Initializing Wikipedia MCP Agent Backend Service...")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info("==================================================")

    # 1. Start and connect MCP Client to the Wikipedia MCP Server
    mcp_client = WikipediaMCPClient()
    try:
        await mcp_client.connect()
        logger.info("[STARTUP] Connected to Wikipedia MCP Server.")
    except Exception as exc:
        logger.error(f"[STARTUP] Could not connect to MCP server: {exc}", exc_info=True)

    # 2. Discover MCP tools and wrap them for LangGraph
    try:
        langchain_tools = await build_langchain_tools_from_mcp(mcp_client)
        logger.info(f"[STARTUP] Successfully loaded {len(langchain_tools)} MCP tools into LangChain.")
    except Exception as exc:
        logger.error(f"[STARTUP] Failed to build tools from MCP server: {exc}", exc_info=True)
        langchain_tools = []

    # 3. Instantiate LLM and bind MCP tools
    try:
        llm = get_llm()
        llm_with_tools = bind_tools_to_llm(llm, langchain_tools)
        logger.info("[STARTUP] Bound MCP tools to LLM.")
    except Exception as exc:
        logger.error(f"[STARTUP] Failed to initialize LLM: {exc}", exc_info=True)
        llm_with_tools = None

    # 4. Compile LangGraph StateGraph workflow
    if llm_with_tools and langchain_tools:
        agent_workflow = build_agent_graph(llm_with_tools=llm_with_tools, tools=langchain_tools)
        logger.info("[STARTUP] LangGraph StateGraph workflow successfully compiled.")
    else:
        logger.warning("[STARTUP] Agent workflow could not be compiled due to missing LLM or tools.")
        agent_workflow = None

    # Store references on application state
    app.state.mcp_client = mcp_client
    app.state.agent_workflow = agent_workflow
    app.state.langchain_tools = langchain_tools

    yield

    # Clean shutdown
    logger.info("[SHUTDOWN] Shutting down Wikipedia MCP Agent...")
    if mcp_client:
        await mcp_client.close()
    logger.info("[SHUTDOWN] Cleanup complete.")


# Instantiate FastAPI application
app = FastAPI(
    title="Wikipedia MCP Agent API",
    description="Production-style Wikipedia Model Context Protocol (MCP) Agent powered by LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT,
        reload=True,
    )
