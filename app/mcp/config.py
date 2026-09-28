import os
import shutil
import sys
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import settings


def resolve_python_executable(cmd: str) -> str:
    """Ensures a valid Python interpreter path is used for spawning subprocesses."""
    # If an explicit absolute path is provided that exists, use it
    if cmd and os.path.isabs(cmd) and os.path.exists(cmd):
        return cmd

    # Always prioritize current running Python interpreter (virtual environment)
    if sys.executable and os.path.exists(sys.executable):
        return sys.executable

    # Fallback to PATH lookup
    if cmd and shutil.which(cmd):
        return cmd

    if shutil.which("python3"):
        return "python3"

    return "python"


class MCPClientConfig(BaseModel):
    """Configuration for connecting to the Wikipedia MCP server."""
    command: str = Field(
        default_factory=lambda: resolve_python_executable(settings.MCP_SERVER_COMMAND),
        description="Path to python executable to launch MCP server.",
    )
    args: List[str] = Field(
        default_factory=lambda: [settings.MCP_SERVER_SCRIPT],
        description="CLI arguments to run the server script.",
    )
    env: Dict[str, str] = Field(
        default_factory=lambda: dict(os.environ),
        description="Environment variables to forward to the MCP server subprocess.",
    )
    timeout: float = Field(
        default_factory=lambda: settings.MCP_SERVER_TIMEOUT,
        description="Timeout in seconds for MCP JSON-RPC tool calls.",
    )
