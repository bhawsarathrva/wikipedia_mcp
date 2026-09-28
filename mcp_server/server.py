
import logging
import os
import sys
from typing import Any, Dict

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.server.mcpserver import MCPServer

from mcp_server.wikipedia import (
    search_wikipedia as api_search_wikipedia,
    get_wikipedia_article as api_get_wikipedia_article,
    get_wikipedia_summary as api_get_wikipedia_summary,
)
from app.logger import setup_json_logging

# Configure logging with structured JSON file output and stderr console stream
setup_json_logging(console_stream=sys.stderr)
logger = logging.getLogger("mcp_server.server")
server = MCPServer(name="wikipedia-mcp-server")


@server.tool(
    name="search_wikipedia",
    description=(
        "Search Wikipedia for articles matching a search query. "
        "Returns a list of matching articles including titles, canonical URLs, "
        "and short description snippets. Use this to discover relevant articles."
    ),
)
async def search_wikipedia(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search Wikipedia articles.

    Args:
        query: The search term or concept keywords to search for.
        limit: Maximum number of articles to return (default 5, between 1 and 20).

    Returns:
        Structured JSON dictionary with search hits, titles, URLs, and snippets.
    """
    logger.info(f"[MCP SERVER] Tool called: search_wikipedia(query='{query}', limit={limit})")
    result = await api_search_wikipedia(query=query, limit=limit)
    hits = len(result.get("results", []))
    logger.info(f"[MCP SERVER] search_wikipedia completed: found {hits} results")
    return result


@server.tool(
    name="get_wikipedia_article",
    description=(
        "Retrieve the full plain-text content, summary, and canonical URL "
        "of a specific Wikipedia article by title. Use this when in-depth "
        "factual details, sections, dates, or context are required."
    ),
)
async def get_wikipedia_article(title: str) -> Dict[str, Any]:
    """Retrieve full Wikipedia article content.

    Args:
        title: The exact or close title of the Wikipedia article.

    Returns:
        Structured dictionary with title, summary, URL, and full article text.
    """
    logger.info(f"[MCP SERVER] Tool called: get_wikipedia_article(title='{title}')")
    result = await api_get_wikipedia_article(title=title)
    found = result.get("found", False)
    logger.info(f"[MCP SERVER] get_wikipedia_article completed for '{title}': found={found}")
    return result


@server.tool(
    name="get_wikipedia_summary",
    description=(
        "Retrieve a concise overview summary and canonical URL of a specific "
        "Wikipedia article by title. Use this for quick factual overviews, definitions, "
        "or biographical summaries without loading the entire article."
    ),
)
async def get_wikipedia_summary(title: str) -> Dict[str, Any]:
    """Retrieve concise Wikipedia article summary.

    Args:
        title: The exact or close title of the Wikipedia article.

    Returns:
        Structured dictionary with title, description, summary, and URL.
    """
    logger.info(f"[MCP SERVER] Tool called: get_wikipedia_summary(title='{title}')")
    result = await api_get_wikipedia_summary(title=title)
    found = result.get("found", False)
    logger.info(f"[MCP SERVER] get_wikipedia_summary completed for '{title}': found={found}")
    return result


def main():
    """Entry point to run the Wikipedia MCP Server on stdio transport."""
    logger.info("[MCP SERVER] Starting Wikipedia MCP server on stdio transport...")
    try:
        server.run("stdio")
    except Exception as exc:
        logger.error(f"[MCP SERVER] Fatal server error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
