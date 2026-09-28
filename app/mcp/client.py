import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.mcp.config import MCPClientConfig

logger = logging.getLogger("app.mcp.client")


class WikipediaMCPClient:
    """Manages an active MCP connection to the Wikipedia MCP server over stdio."""

    def __init__(self, config: Optional[MCPClientConfig] = None):
        self.config = config or MCPClientConfig()
        self._cm_stdio = None
        self._cm_session = None
        self.session: Optional[ClientSession] = None
        self._lock = asyncio.Lock()
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Indicates if the client is connected to the MCP server."""
        return self._is_connected and self.session is not None

    async def connect(self) -> None:
        """Starts the MCP server subprocess and initializes the JSON-RPC session."""
        async with self._lock:
            if self._is_connected and self.session:
                return

            logger.info(
                f"[MCP CLIENT] Connecting to MCP server via: "
                f"{self.config.command} {' '.join(self.config.args)}"
            )

            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
            )

            try:
                # Enter stdio client context
                self._cm_stdio = stdio_client(server_params)
                read_stream, write_stream = await self._cm_stdio.__aenter__()

                # Enter client session context
                self._cm_session = ClientSession(read_stream, write_stream)
                self.session = await self._cm_session.__aenter__()

                # Initialize MCP protocol handshake
                init_result = await self.session.initialize()
                protocol_ver = getattr(init_result, "protocol_version", getattr(init_result, "protocolVersion", "unknown"))
                server_info = getattr(init_result, "server_info", getattr(init_result, "serverInfo", None))
                server_name = getattr(server_info, "name", "unknown") if server_info else "unknown"
                self._is_connected = True
                logger.info(
                    f"[MCP CLIENT] Successfully connected to MCP server: "
                    f"protocol_version={protocol_ver}, "
                    f"server_name={server_name}"
                )
            except Exception as exc:
                self._is_connected = False
                logger.error(f"[MCP CLIENT] Failed to connect to MCP server: {exc}", exc_info=True)
                await self._cleanup()
                raise ConnectionError(f"Could not connect to Wikipedia MCP server: {exc}") from exc

    async def list_tools(self) -> List[Any]:
        """Queries the MCP server for available tools and their JSON schemas."""
        if not self.is_connected:
            await self.connect()

        try:
            logger.info("[MCP CLIENT] Requesting tool definitions from MCP server...")
            response = await self.session.list_tools()
            tools = response.tools
            tool_names = [t.name for t in tools]
            logger.info(f"[MCP CLIENT] Discovered {len(tools)} tools on MCP server: {tool_names}")
            return tools
        except Exception as exc:
            logger.error(f"[MCP CLIENT] Error listing tools: {exc}", exc_info=True)
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Calls an MCP tool on the server and returns the text result.

        Args:
            tool_name: The name of the registered MCP tool.
            arguments: Dictionary of arguments matching the tool's input schema.

        Returns:
            String output of the tool execution (usually JSON formatted string).
        """
        if not self.is_connected:
            await self.connect()

        logger.info(f"[MCP CLIENT] Calling {tool_name} with arguments: {arguments}")
        try:
            async with asyncio.timeout(self.config.timeout):
                result = await self.session.call_tool(tool_name, arguments)

            # Process returned content items
            text_chunks: List[str] = []
            for item in result.content:
                if hasattr(item, "text"):
                    text_chunks.append(item.text)
                else:
                    text_chunks.append(str(item))

            output_text = "\n".join(text_chunks)
            snippet = output_text[:120].replace("\n", " ")
            logger.info(f"[MCP CLIENT] Tool result received for {tool_name}: '{snippet}...'")
            return output_text

        except asyncio.TimeoutError:
            err_msg = f"Tool call '{tool_name}' timed out after {self.config.timeout}s."
            logger.error(f"[MCP CLIENT] {err_msg}")
            return json.dumps({"error": err_msg})
        except Exception as exc:
            logger.error(f"[MCP CLIENT] Error calling tool '{tool_name}': {exc}", exc_info=True)
            return json.dumps({"error": f"MCP execution failed: {str(exc)}"})

    async def _cleanup(self) -> None:
        """Closes session and stdio streams safely."""
        self._is_connected = False
        if self._cm_session:
            try:
                await self._cm_session.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm_session = None
            self.session = None

        if self._cm_stdio:
            try:
                await self._cm_stdio.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm_stdio = None

    async def close(self) -> None:
        """Cleanly disconnects and shuts down the MCP server subprocess."""
        async with self._lock:
            logger.info("[MCP CLIENT] Disconnecting from MCP server...")
            await self._cleanup()
            logger.info("[MCP CLIENT] MCP Client disconnected cleanly.")
