"""Unit and integration tests for Wikipedia API and MCP Server."""

import pytest
from mcp_server.wikipedia import (
    search_wikipedia,
    get_wikipedia_summary,
    get_wikipedia_article,
    _clean_snippet,
)
from mcp_server.tools import (
    execute_search_wikipedia,
    execute_get_wikipedia_summary,
    execute_get_wikipedia_article,
)
from app.mcp.client import WikipediaMCPClient


def test_clean_snippet():
    """Verify HTML stripping from search snippets."""
    raw = 'Albert <span class="searchmatch">Einstein</span> was a physicist &amp; genius.'
    cleaned = _clean_snippet(raw)
    assert '<span class="searchmatch">' not in cleaned
    assert "</span>" not in cleaned
    assert "&amp;" not in cleaned
    assert "Albert Einstein was a physicist & genius." == cleaned


@pytest.mark.asyncio
async def test_search_wikipedia_empty():
    """Verify handling of empty search query."""
    res = await search_wikipedia("")
    assert res["total_results"] == 0
    assert len(res["results"]) == 0
    assert "empty" in res.get("message", "").lower()


@pytest.mark.asyncio
async def test_search_wikipedia_live():
    """Verify live search against Wikipedia Action API."""
    res = await search_wikipedia(query="Alan Turing", limit=3)
    assert res["query"] == "Alan Turing"
    assert len(res["results"]) > 0
    first = res["results"][0]
    assert "Alan Turing" in first["title"]
    assert "https://en.wikipedia.org/wiki/" in first["url"]
    assert "snippet" in first


@pytest.mark.asyncio
async def test_get_wikipedia_summary_live():
    """Verify live summary retrieval via Wikipedia REST API."""
    res = await get_wikipedia_summary(title="Alan Turing")
    assert res["found"] is True
    assert res["title"] == "Alan Turing"
    assert "mathematician" in res["summary"].lower()
    assert res["url"] == "https://en.wikipedia.org/wiki/Alan_Turing"


@pytest.mark.asyncio
async def test_get_wikipedia_article_live():
    """Verify full article text retrieval via Wikipedia Action API."""
    res = await get_wikipedia_article(title="Alan Turing", max_characters=1000)
    assert res["found"] is True
    assert "Turing" in res["title"]
    assert len(res["content"]) <= 1000
    assert res["url"] == "https://en.wikipedia.org/wiki/Alan_Turing"


@pytest.mark.asyncio
async def test_mcp_client_server_integration():
    """Integration test: Launch MCP server subprocess via stdio and call tools."""
    client = WikipediaMCPClient()
    try:
        await client.connect()
        assert client.is_connected is True

        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        assert "search_wikipedia" in tool_names
        assert "get_wikipedia_article" in tool_names
        assert "get_wikipedia_summary" in tool_names

        # Call search_wikipedia over stdio JSON-RPC
        result_text = await client.call_tool("search_wikipedia", {"query": "Isaac Newton", "limit": 2})
        assert "Isaac Newton" in result_text
        assert "https://en.wikipedia.org/wiki/" in result_text

    finally:
        await client.close()
