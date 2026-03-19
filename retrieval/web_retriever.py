"""
Web Retriever — Wraps web search results for the retrieval pipeline.
"""

from tools.web_search import search_web


async def web_retrieve(query: str, k: int = 5) -> list:
    """
    Retrieve documents from web search.
    Returns list of dicts: [{content, url, title, source_type}, ...]
    """
    try:
        results = await search_web(query)
        docs = []
        for r in results[:k]:
            docs.append({
                "content": r["content"],
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "source_type": "web",
            })
        return docs
    except Exception as e:
        print(f"Web retriever error: {e}")
        return []
