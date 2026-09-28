"""Bridges official MCP Tools to LangChain / LangGraph compatible tools.

========================================================================================
ARCHITECTURAL FLOW OF TOOL EXECUTION:
========================================================================================
[STAGE 1]: The LLM (Gemini or OpenAI) analyzes the user prompt and decides to invoke a tool.
           Example: ToolCall(name="search_wikipedia", args={"query": "Alan Turing", "limit": 5})

[STAGE 2]: LangGraph routes the tool call into the Tool Node / LangChain StructuredTool wrapper.

[STAGE 3]: The StructuredTool coroutine calls WikipediaMCPClient.call_tool(tool_name, arguments).
           The MCP Client formats the request into a standard JSON-RPC 2.0 message:
           {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "search_wikipedia", ...}}
           and transmits it across the STDIO pipe to the MCP Server process.

[STAGE 4]: The Wikipedia MCP Server (running mcp_server/server.py) receives the JSON-RPC request,
           dispatches it to mcp_server/wikipedia.py, which makes an HTTP GET request to the
           official Wikipedia API (https://en.wikipedia.org/w/api.php).

[STAGE 5]: Wikipedia's public API returns the raw search results / article extract JSON to the
           MCP Server.

[STAGE 6]: The MCP Server cleans snippets, sanitizes HTML, wraps the structured dictionary
           into an MCP TextContent payload, and returns the JSON-RPC response back over STDIO.

[STAGE 7]: The MCP Client deserializes the JSON-RPC response, extracts the text content, and
           returns it to LangGraph as a ToolMessage.

[STAGE 8]: LangGraph loops back to the LLM node with the ToolMessage. The LLM reads the Wikipedia
           facts and synthesizes the final human-readable answer with source URLs.
========================================================================================
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from app.mcp.client import WikipediaMCPClient

logger = logging.getLogger("app.agent.tools")


# Fallback / explicit input schemas for well-known tools
class SearchWikipediaSchema(BaseModel):
    query: str = Field(description="The search string or concept keywords to look up on Wikipedia.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of search results to return.")


class GetWikipediaArticleSchema(BaseModel):
    title: str = Field(description="The exact title of the Wikipedia article to retrieve.")


class GetWikipediaSummarySchema(BaseModel):
    title: str = Field(description="The exact title of the Wikipedia article for which to retrieve the summary.")


KNOWN_SCHEMAS: Dict[str, type[BaseModel]] = {
    "search_wikipedia": SearchWikipediaSchema,
    "get_wikipedia_article": GetWikipediaArticleSchema,
    "get_wikipedia_summary": GetWikipediaSummarySchema,
}


def _create_pydantic_schema_from_mcp_json_schema(name: str, input_schema: Dict[str, Any]) -> type[BaseModel]:
    """Dynamically builds a Pydantic model from an MCP tool's JSON schema properties."""
    properties = input_schema.get("properties", {})
    required_fields = set(input_schema.get("required", []))

    field_definitions: Dict[str, Any] = {}
    for prop_name, prop_data in properties.items():
        prop_type = prop_data.get("type", "string")
        prop_desc = prop_data.get("description", "")
        prop_default = prop_data.get("default", ... if prop_name in required_fields else None)

        py_type: Any = str
        if prop_type == "integer":
            py_type = int
        elif prop_type == "number":
            py_type = float
        elif prop_type == "boolean":
            py_type = bool
        elif prop_type == "array":
            py_type = list
        elif prop_type == "object":
            py_type = dict

        field_definitions[prop_name] = (
            py_type,
            Field(default=prop_default, description=prop_desc),
        )

    model_name = f"{name.title().replace('_', '')}Input"
    return create_model(model_name, **field_definitions)


def create_mcp_tool_wrapper(mcp_tool: Any, mcp_client: WikipediaMCPClient) -> StructuredTool:
    """Wraps an MCP tool into a LangChain StructuredTool.

    The wrapped tool executes entirely via the official MCP Client, forwarding
    the call to the MCP Server over STDIO JSON-RPC.
    """
    tool_name = mcp_tool.name
    tool_description = mcp_tool.description or f"MCP tool {tool_name}"

    # Determine input Pydantic schema
    if tool_name in KNOWN_SCHEMAS:
        schema = KNOWN_SCHEMAS[tool_name]
    else:
        raw_schema = getattr(mcp_tool, "inputSchema", {}) or {}
        try:
            schema = _create_pydantic_schema_from_mcp_json_schema(tool_name, raw_schema)
        except Exception as exc:
            logger.warning(f"Could not dynamically derive schema for {tool_name}: {exc}. Using dict fallback.")
            schema = None

    if schema is None:
        class FallbackToolSchema(BaseModel):
            model_config = {"extra": "allow"}

        schema = FallbackToolSchema

    # Define the asynchronous coroutine that dispatches to MCP Client
    async def _tool_coroutine(**kwargs: Any) -> str:
        """STAGE 2 -> 3: Coroutine executed by LangGraph when LLM requests this tool."""
        logger.info(f"[LANGGRAPH] Tool dispatched: {tool_name} with parameters: {kwargs}")

        # STAGE 3: Call through MCP Client
        # The MCP Client sends the JSON-RPC call to the MCP Server over STDIO
        mcp_result_text = await mcp_client.call_tool(tool_name=tool_name, arguments=kwargs)

        # STAGE 7: Return string back to LangGraph ToolNode / StateGraph
        return mcp_result_text

    # Synchronous stub (LangGraph async workflow uses coroutine)
    def _tool_sync_stub(**kwargs: Any) -> str:
        raise NotImplementedError("Use async execution for MCP tools.")

    return StructuredTool(
        name=tool_name,
        description=tool_description,
        func=_tool_sync_stub,
        coroutine=_tool_coroutine,
        args_schema=schema,
    )


async def build_langchain_tools_from_mcp(mcp_client: WikipediaMCPClient) -> List[StructuredTool]:
    """Discovers MCP tools on the MCP server and returns LangChain StructuredTools."""
    mcp_tools = await mcp_client.list_tools()
    langchain_tools: List[StructuredTool] = []

    for tool in mcp_tools:
        wrapped = create_mcp_tool_wrapper(tool, mcp_client)
        langchain_tools.append(wrapped)
        logger.info(f"Registered LangChain tool for MCP tool: {wrapped.name}")

    return langchain_tools
