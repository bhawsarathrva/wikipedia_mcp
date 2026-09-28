"""Prompt loader and assembler for the Wikipedia MCP Agent."""

import logging
from pathlib import Path
from app.config import settings

logger = logging.getLogger("app.agent.prompts")


def _read_prompt_file(path_str: str, default: str) -> str:
    """Reads prompt content from file or returns default."""
    path = Path(path_str)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning(f"Could not read prompt file {path}: {exc}. Using default.")
    return default.strip()


def load_system_prompt() -> str:
    """Loads and combines system prompt, tool usage prompt, and answer formatting rules."""
    default_system = (
        "You are a Wikipedia research assistant. "
        "Your job is to answer user questions accurately using information retrieved from Wikipedia through MCP tools. "
        "Only call tools when factual retrieval is required."
    )
    default_wiki = (
        "If the question references factual entities, concepts, or events, search Wikipedia using concise queries. "
        "Do not invoke tools for simple greetings."
    )
    default_answer = (
        "Construct clear, direct answers grounded in retrieved data. "
        "Avoid raw JSON formatting and always include canonical Wikipedia URLs."
    )

    system_text = _read_prompt_file(settings.SYSTEM_PROMPT_PATH, default_system)
    wiki_text = _read_prompt_file(settings.WIKIPEDIA_PROMPT_PATH, default_wiki)
    answer_text = _read_prompt_file(settings.ANSWER_PROMPT_PATH, default_answer)

    combined_prompt = (
        f"{system_text}\n\n"
        f"--- TOOL SELECTION GUIDELINES ---\n"
        f"{wiki_text}\n\n"
        f"--- ANSWER FORMATTING RULES ---\n"
        f"{answer_text}"
    )
    return combined_prompt
