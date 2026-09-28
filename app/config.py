"""Application configuration settings using pydantic-settings.

Loads environment variables from .env file and provides typed settings
for LLM providers, MCP server parameters, and network endpoints.
"""

import os
import sys
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
LOGS_DIR = BASE_DIR / "logs"


class Settings(BaseSettings):
    """Configuration settings for the Wikipedia MCP Agent application."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging Configuration
    LOG_FILE: str = str(LOGS_DIR / "app.log.jsonl")
    LOG_LEVEL: str = "INFO"

    # LLM Settings
    LLM_PROVIDER: Literal["ollama", "gemini", "openai"] = "ollama"
    OLLAMA_MODEL: str = "gemma4:latest"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1

    # MCP Server Process Configuration
    # Uses current python virtual environment executable by default
    MCP_SERVER_COMMAND: str = sys.executable
    MCP_SERVER_SCRIPT: str = str(BASE_DIR / "mcp_server" / "server.py")
    MCP_SERVER_TIMEOUT: float = 25.0

    # Backend API Server Configuration
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    BACKEND_URL: str = "http://localhost:8000"

    # Wikimedia API Compliance User-Agent
    WIKIPEDIA_USER_AGENT: str = (
        "WikipediaMCPAgentBot/1.0 (https://github.com/athrvabhawsar07/wikipedia-mcp-agent; athrvabhawsar07@gmail.com)"
    )

    # Prompt paths
    SYSTEM_PROMPT_PATH: str = str(PROMPTS_DIR / "system_prompt.txt")
    WIKIPEDIA_PROMPT_PATH: str = str(PROMPTS_DIR / "wikipedia_prompt.txt")
    ANSWER_PROMPT_PATH: str = str(PROMPTS_DIR / "answer_prompt.txt")


settings = Settings()
