"""
Web search tool using Tavily API.
Includes domain quality filtering to avoid low-credibility sources.
"""

from typing import Dict, List

from config import TAVILY_API_KEY

# High-credibility domains that get priority
TRUSTED_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "techcrunch.com", "electrek.co", "thedriven.io",
    "skon.com", "sk.com", "catl.com",
    "sneresearch.com", "sne-research.com", "woodmac.com", "benchmarkminerals.com",
    "sec.gov", "hkexnews.hkex.com.hk", "cninfo.com.cn",
    "mk.co.kr", "hankyung.com", "chosun.com", "joongang.co.kr",
    "businesskorea.co.kr", "koreaherald.com", "yonhapnewstv.co.kr", "yna.co.kr",
    "energy-storage.news", "pv-tech.org", "electrive.com",
    "theicct.org", "iea.org", "statista.com",
}

# Top-tier domains for critical quantitative claims
TOP_TIER_DOMAINS = {
    "reuters.com", "bloomberg.com", "iea.org",
    "sneresearch.com", "sne-research.com",
    "sec.gov", "hkexnews.hkex.com.hk",
    "skon.com", "sk.com", "catl.com",
}

# Low-credibility domains to skip
BLOCKED_DOMAINS = {
    "tistory.com", "naver.com/blog", "blog.naver.com",
    "brunch.co.kr", "medium.com", "velog.io",
    "wordpress.com", "blogspot.com", "tumblr.com",
    "quora.com", "reddit.com",
    "linkedin.com", "threads.net", "x.com", "twitter.com",
    "facebook.com", "instagram.com", "weibo.com",
    "pinterest.com", "discord.com", "t.me",
}


def _is_top_tier(url: str) -> bool:
    """Return True if URL belongs to top-tier evidence domains."""
    if not url:
        return False
    u = url.lower()
    return any(d in u for d in TOP_TIER_DOMAINS)


def _domain_score(url: str) -> int:
    """
    Return a quality score for a URL.
    Higher = more credible.
    """
    if not url or url == "tavily://answer":
        return 0
    url_lower = url.lower()
    for blocked in BLOCKED_DOMAINS:
        if blocked in url_lower:
            return -1
    if _is_top_tier(url_lower):
        return 3
    for trusted in TRUSTED_DOMAINS:
        if trusted in url_lower:
            return 2
    return 1


def web_search(
    query: str,
    max_results: int = 5,
    include_tavily_answer: bool = False,
    min_domain_score: int = 1,
    require_top_tier: bool = False,
) -> List[Dict]:
    """
    Perform a web search using the Tavily API.
    Filters low-quality domains and ranks by credibility.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        include_tavily_answer: include synthesized Tavily answer as hint item
        min_domain_score: minimum domain quality score (0/1/2)
        require_top_tier: keep only top-tier domains (for critical metrics)

    Returns:
        List of dicts with keys: title, url, content, domain_score
        Returns empty list on error.
    """
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            max_results=max_results + 3,  # fetch extra to compensate for filtered items
            include_raw_content=False,
            include_answer=include_tavily_answer,
        )

        results = []

        # Optional synthesized answer (hint only, not a primary citable source)
        if include_tavily_answer and response.get("answer"):
            results.append({
                "title": "Tavily 종합 답변",
                "url": "tavily://answer",
                "content": response["answer"],
                "domain_score": 1,
                "source_tier": "hint",
                "is_top_tier": False,
            })

        # Score and filter individual results
        for item in response.get("results", []):
            url = item.get("url", "")
            score = _domain_score(url)
            is_top = _is_top_tier(url)
            if score < max(0, min_domain_score):
                continue  # skip blocked/low-quality domains
            if require_top_tier and not is_top:
                continue

            if score >= 3:
                tier = "top"
            elif score == 2:
                tier = "trusted"
            else:
                tier = "general"

            results.append({
                "title": item.get("title", "제목 없음"),
                "url": url,
                "content": item.get("content", ""),
                "domain_score": score,
                "source_tier": tier,
                "is_top_tier": is_top,
            })

        # Sort by domain credibility (trusted first)
        results.sort(key=lambda x: x["domain_score"], reverse=True)

        return results[:max_results + (1 if include_tavily_answer else 0)]

    except ImportError:
        print("[web_search] tavily 패키지가 설치되지 않았습니다. pip install tavily-python")
        return []
    except Exception as e:
        print(f"[web_search] 검색 오류 (query='{query}'): {e}")
        return []
