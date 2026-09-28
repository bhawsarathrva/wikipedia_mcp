"""LLM model provider factory and tool binding.

Supports:
- Google Gemini via langchain-google-genai (ChatGoogleGenerativeAI)
- OpenAI via langchain-openai (ChatOpenAI)

Allows switching providers via environment variable LLM_PROVIDER ('gemini' or 'openai').
"""

import logging
import os
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.config import settings

logger = logging.getLogger("app.agent.agent")


def get_llm(provider: Optional[str] = None) -> BaseChatModel:
    """Instantiates and returns the configured ChatModel (Ollama, Gemini, or OpenAI).

    Args:
        provider: Optional override ('ollama', 'gemini', or 'openai'). Defaults to settings.LLM_PROVIDER.

    Returns:
        Configured BaseChatModel instance.
    """
    active_provider = (provider or settings.LLM_PROVIDER).lower()

    if active_provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.info(
            f"Using Local Ollama model: {settings.OLLAMA_MODEL} at {settings.OLLAMA_BASE_URL}"
        )
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
        )

    elif active_provider == "gemini":
        google_api_key = settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            logger.warning("GOOGLE_API_KEY not found in environment. Gemini calls will fail without a key.")

        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info(f"Using Google Gemini model: {settings.GEMINI_MODEL}")
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=google_api_key,
            temperature=settings.LLM_TEMPERATURE,
        )

    elif active_provider == "openai":
        openai_api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment. OpenAI calls will fail without a key.")

        from langchain_openai import ChatOpenAI

        logger.info(f"Using OpenAI model: {settings.OPENAI_MODEL}")
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=openai_api_key,
            temperature=settings.LLM_TEMPERATURE,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {active_provider}. Choose 'ollama', 'gemini', or 'openai'.")


def bind_tools_to_llm(llm: BaseChatModel, tools: List[BaseTool]) -> Any:
    """Binds MCP-derived tools to the LLM model instance for function calling."""
    if not tools:
        logger.warning("No tools provided to bind_tools_to_llm.")
        return llm

    logger.info(f"Binding {len(tools)} tools to LLM: {[t.name for t in tools]}")
    return llm.bind_tools(tools)
