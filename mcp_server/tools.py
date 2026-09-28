from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mcp_server.wikipedia import (
    search_wikipedia as api_search_wikipedia,
    get_wikipedia_article as api_get_wikipedia_article,
    get_wikipedia_summary as api_get_wikipedia_summary,
)


class SearchWikipediaInput(BaseModel):
    """Input model for the search_wikipedia tool."""
    query: str = Field(
        ...,
        description="The search string or concept keywords to look up on Wikipedia.",
        examples=["Albert Einstein", "Theory of relativity", "Eiffel Tower"],
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to return (default 5, min 1, max 20).",
    )


class GetWikipediaArticleInput(BaseModel):
    """Input model for the get_wikipedia_article tool."""
    title: str = Field(
        ...,
        description="The exact title of the Wikipedia article to retrieve.",
        examples=["Albert Einstein", "Artificial intelligence"],
    )


class GetWikipediaSummaryInput(BaseModel):
    """Input model for the get_wikipedia_summary tool."""
    title: str = Field(
        ...,
        description="The exact title of the Wikipedia article for which to retrieve the summary.",
        examples=["Quantum mechanics", "Alan Turing"],
    )


async def execute_search_wikipedia(query: str, limit: int = 5) -> Dict[str, Any]:
    """Executes Wikipedia search and returns structured results."""
    return await api_search_wikipedia(query=query, limit=limit)


async def execute_get_wikipedia_article(title: str) -> Dict[str, Any]:
    """Executes full Wikipedia article text retrieval."""
    return await api_get_wikipedia_article(title=title)


async def execute_get_wikipedia_summary(title: str) -> Dict[str, Any]:
    """Executes Wikipedia article summary retrieval."""
    return await api_get_wikipedia_summary(title=title)
