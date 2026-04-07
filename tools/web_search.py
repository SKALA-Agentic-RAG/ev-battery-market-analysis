"""
Web search tool using Tavily API.
Provides structured search results for battery/EV market research.
"""

import os
from typing import Dict, List

from config import TAVILY_API_KEY


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a web search using the Tavily API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of dicts with keys: title, url, content
        Returns empty list on error.
    """
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=False,
            include_answer=True,
        )

        results = []

        # Include the synthesized answer if available
        if response.get("answer"):
            results.append({
                "title": "Tavily 종합 답변",
                "url": "tavily://answer",
                "content": response["answer"],
            })

        # Include individual search results
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", "제목 없음"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            })

        return results[:max_results + 1]  # +1 for the answer entry

    except ImportError:
        print("[web_search] tavily 패키지가 설치되지 않았습니다. pip install tavily-python")
        return []
    except Exception as e:
        print(f"[web_search] 검색 오류 (query='{query}'): {e}")
        return []
