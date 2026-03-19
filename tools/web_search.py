from tavily import AsyncTavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = AsyncTavilyClient(
    api_key= os.getenv("TAVILY_API_KEY")
)

async def search_web(query):
    """
    Search the web using Tavily and return results with full metadata.
    
    Returns: list of dicts [{"content": ..., "url": ..., "title": ...}, ...]
    """
    try:
        results = await client.search(query)

        documents = []

        for r in results.get("results", []):
            documents.append({
                "content": r["content"],
                "url": r.get("url", ""),
                "title": r.get("title", ""),
            })

        return documents
    except Exception as e:
        print(f"Web search error for query '{query}': {e}")
        return []