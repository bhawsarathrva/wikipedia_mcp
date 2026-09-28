import html
import logging
import re
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger("mcp_server.wikipedia")

DEFAULT_USER_AGENT = "WikipediaMCPAgentBot/1.0 (https://github.com/athrvabhawsar07/wikipedia-mcp-agent; athrvabhawsar07@gmail.com)"
WIKIPEDIA_ACTION_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST_API = "https://en.wikipedia.org/api/rest_v1"
REQUEST_TIMEOUT_SECONDS = 12.0


def _clean_snippet(raw_snippet: str) -> str:
    """Removes HTML searchmatch tags and unescapes entities."""
    if not raw_snippet:
        return ""
    cleaned = re.sub(r"<[^>]+>", "", raw_snippet)
    return html.unescape(cleaned).strip()


def _format_wiki_url(title: str) -> str:
    """Generates canonical Wikipedia article URL."""
    sanitized = title.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{sanitized}"


async def search_wikipedia(
    query: str,
    limit: int = 5,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, Any]:
    """Searches Wikipedia articles via Action API.

    Args:
        query: Search string or keywords.
        limit: Maximum number of articles to return (1 to 20).
        user_agent: HTTP User-Agent string.

    Returns:
        Structured dictionary containing query, count, and list of article items.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        logger.warning("[MCP SERVER] Empty query passed to search_wikipedia")
        return {
            "query": "",
            "total_results": 0,
            "results": [],
            "message": "Query was empty. Please provide a search term.",
        }

    clamped_limit = max(1, min(limit, 20))
    logger.info(f"[MCP SERVER] Searching Wikipedia: '{clean_query}' (limit={clamped_limit})")

    params = {
        "action": "query",
        "list": "search",
        "srsearch": clean_query,
        "srlimit": clamped_limit,
        "format": "json",
        "utf8": 1,
    }
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(WIKIPEDIA_ACTION_API, params=params, headers=headers)

            if response.status_code == 429:
                logger.error("[MCP SERVER] Wikipedia rate limit (429) encountered")
                return {
                    "query": clean_query,
                    "total_results": 0,
                    "results": [],
                    "error": "Rate limit exceeded from Wikipedia API. Please wait a few moments.",
                }

            response.raise_for_status()
            data = response.json()

        search_entries = data.get("query", {}).get("search", [])
        total_hits = data.get("query", {}).get("searchinfo", {}).get("totalhits", len(search_entries))

        formatted_results: List[Dict[str, Any]] = []
        for item in search_entries:
            title = item.get("title", "")
            raw_snippet = item.get("snippet", "")
            cleaned_snippet = _clean_snippet(raw_snippet)
            formatted_results.append({
                "title": title,
                "url": _format_wiki_url(title),
                "snippet": cleaned_snippet,
                "wordcount": item.get("wordcount", 0),
            })

        logger.info(f"[MCP SERVER] Found {len(formatted_results)} results for '{clean_query}'")
        return {
            "query": clean_query,
            "total_results": total_hits,
            "returned_count": len(formatted_results),
            "results": formatted_results,
        }

    except httpx.TimeoutException:
        logger.error(f"[MCP SERVER] Timeout connecting to Wikipedia for query '{clean_query}'")
        return {
            "query": clean_query,
            "total_results": 0,
            "results": [],
            "error": "Request to Wikipedia timed out. Please try again.",
        }
    except Exception as exc:
        logger.error(f"[MCP SERVER] Error executing search_wikipedia: {exc}", exc_info=True)
        return {
            "query": clean_query,
            "total_results": 0,
            "results": [],
            "error": f"Failed to search Wikipedia: {str(exc)}",
        }


async def get_wikipedia_summary(
    title: str,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, Any]:
    """Retrieves article summary from Wikipedia REST API.

    Args:
        title: Exact or close Wikipedia article title.
        user_agent: HTTP User-Agent string.

    Returns:
        Dictionary with title, summary extract, description, and canonical URL.
    """
    clean_title = title.strip() if title else ""
    if not clean_title:
        return {
            "title": "",
            "found": False,
            "error": "Article title was empty. Please provide a title.",
        }

    logger.info(f"[MCP SERVER] Fetching Wikipedia summary for '{clean_title}'")
    encoded_title = clean_title.replace(" ", "_")
    endpoint = f"{WIKIPEDIA_REST_API}/page/summary/{encoded_title}"
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(endpoint, headers=headers)

            if response.status_code == 404:
                logger.warning(f"[MCP SERVER] Article not found: '{clean_title}'")
                return {
                    "title": clean_title,
                    "found": False,
                    "error": f"Wikipedia article '{clean_title}' was not found.",
                }

            if response.status_code == 429:
                return {
                    "title": clean_title,
                    "found": False,
                    "error": "Wikipedia rate limit reached. Please wait briefly.",
                }

            response.raise_for_status()
            data = response.json()

        canonical_url = (
            data.get("content_urls", {}).get("desktop", {}).get("page")
            or _format_wiki_url(data.get("title", clean_title))
        )

        return {
            "title": data.get("title", clean_title),
            "description": data.get("description", ""),
            "summary": data.get("extract", ""),
            "url": canonical_url,
            "found": True,
        }

    except httpx.TimeoutException:
        logger.error(f"[MCP SERVER] Timeout retrieving summary for '{clean_title}'")
        return {
            "title": clean_title,
            "found": False,
            "error": f"Request timed out while retrieving summary for '{clean_title}'.",
        }
    except Exception as exc:
        logger.error(f"[MCP SERVER] Error retrieving summary for '{clean_title}': {exc}", exc_info=True)
        return {
            "title": clean_title,
            "found": False,
            "error": f"Failed to retrieve summary for '{clean_title}': {str(exc)}",
        }


async def get_wikipedia_article(
    title: str,
    max_characters: int = 6000,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, Any]:
    """Retrieves full article text and summary via Wikipedia Action API.

    Args:
        title: Exact or close Wikipedia article title.
        max_characters: Maximum length of text to return to prevent token blowout.
        user_agent: HTTP User-Agent string.

    Returns:
        Structured article information including title, summary, URL, and plain text.
    """
    clean_title = title.strip() if title else ""
    if not clean_title:
        return {
            "title": "",
            "found": False,
            "error": "Article title was empty.",
        }

    logger.info(f"[MCP SERVER] Fetching full Wikipedia article text for '{clean_title}'")

    # Fetch summary first to get canonical title and quick description
    summary_data = await get_wikipedia_summary(clean_title, user_agent=user_agent)
    canonical_title = summary_data.get("title", clean_title)
    canonical_url = summary_data.get("url", _format_wiki_url(canonical_title))
    summary_text = summary_data.get("summary", "")

    # Query full plain-text extract using Action API
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": canonical_title,
        "format": "json",
        "utf8": 1,
    }
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(WIKIPEDIA_ACTION_API, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

        pages = data.get("query", {}).get("pages", {})
        page_id = next(iter(pages), None)

        if not page_id or page_id == "-1":
            if summary_data.get("found"):
                # If summary succeeded but full text failed, provide summary
                return {
                    "title": canonical_title,
                    "summary": summary_text,
                    "content": summary_text,
                    "url": canonical_url,
                    "found": True,
                    "truncated": False,
                }
            return {
                "title": clean_title,
                "found": False,
                "error": f"Article '{clean_title}' not found on Wikipedia.",
            }

        page_info = pages[page_id]
        full_text = page_info.get("extract", "") or summary_text

        is_truncated = len(full_text) > max_characters
        content = full_text[:max_characters] if is_truncated else full_text

        return {
            "title": page_info.get("title", canonical_title),
            "summary": summary_text,
            "content": content,
            "url": canonical_url,
            "found": True,
            "truncated": is_truncated,
            "total_length": len(full_text),
        }

    except Exception as exc:
        logger.error(f"[MCP SERVER] Error retrieving full article for '{clean_title}': {exc}", exc_info=True)
        # Fall back to summary if available
        if summary_data.get("found"):
            return {
                "title": canonical_title,
                "summary": summary_text,
                "content": summary_text,
                "url": canonical_url,
                "found": True,
                "truncated": False,
            }
        return {
            "title": clean_title,
            "found": False,
            "error": f"Failed to retrieve article for '{clean_title}': {str(exc)}",
        }
