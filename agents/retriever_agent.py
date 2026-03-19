from tools.web_search import search_web
from retrieval.hybrid_retriever import hybrid_search
import asyncio


async def _fetch_one_async(qt, local_only=False):
    """
    Fetch docs for a single query-task asynchronously.
    If local_only=True, skips web search entirely.
    """
    docs = []

    # --- Web search (skip if local_only) ---
    if not local_only:
        try:
            web_results = await search_web(qt["query"])
            for doc in web_results[:3]:
                docs.append({
                    "task_id": qt["task_id"],
                    "query": qt["query"],
                    "content": doc["content"],
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "source_type": "web",
                })
        except Exception as e:
            print(f"Web search error for: {qt['query']} -- {e}")

    # --- Local hybrid search (if index exists) ---
    try:
        local_k = 5 if local_only else 3
        local_results = await asyncio.to_thread(hybrid_search, qt["query"], local_k)
        for doc in local_results:
            docs.append({
                "task_id": qt["task_id"],
                "query": qt["query"],
                "content": doc["content"],
                "url": doc.get("url", ""),
                "title": doc.get("title", "local document"),
                "source_type": "local",
            })
    except Exception:
        pass

    return docs


async def retrieve_documents_async(search_queries, local_only=False):
    """
    Asynchronously retrieves task-tagged documents with full metadata.
    
    Args:
        search_queries: Query plan from query_generator
        local_only: If True, skip web search — only retrieve from uploaded docs
    """
    all_documents = []
    query_tasks = []

    # -------- Extract queries with task context --------
    if isinstance(search_queries, dict):
        if "task_queries" in search_queries:
            for task in search_queries["task_queries"]:
                task_id = task.get("task_id", "unknown")
                for q in task.get("queries", []):
                    query_tasks.append({"task_id": task_id, "query": q})
        elif "queries" in search_queries:
            for q in search_queries["queries"]:
                query_tasks.append({"task_id": "untracked", "query": q})
        elif "search_tasks" in search_queries:
            for task in search_queries["search_tasks"]:
                task_id = task.get("task_id", "unknown")
                for q in task.get("queries", []):
                    query_tasks.append({"task_id": task_id, "query": q})

    elif isinstance(search_queries, str):
        for line in search_queries.split("\n"):
            query = line.strip()
            if not query or len(query) < 3:
                continue
            if query[0].isdigit():
                query = query.split(".", 1)[-1].strip()
            if query.startswith("-"):
                query = query[1:].strip()
            if len(query) >= 3:
                query_tasks.append({"task_id": "untracked", "query": query})
    else:
        print(f"Unexpected input type for retrieve_documents: {type(search_queries)}")
        return []

    query_tasks = query_tasks[:6]

    if not query_tasks:
        print("No queries to retrieve documents for.")
        return []

    # -------- Retrieve in parallel with task tagging --------
    seen_content = set()

    results = await asyncio.gather(
        *[_fetch_one_async(qt, local_only=local_only) for qt in query_tasks],
        return_exceptions=True
    )

    for i, res in enumerate(results):
        if isinstance(res, Exception):
            qt = query_tasks[i]
            print(f"Retrieval error for: {qt['query']} -- {res}")
            continue
        for doc in res:
            fingerprint = doc["content"][:200].strip().lower()
            if fingerprint and fingerprint not in seen_content:
                seen_content.add(fingerprint)
                all_documents.append(doc)

    return all_documents[:20]

def retrieve_documents(search_queries, local_only=False):
    """
    Synchronous wrapper. Safe to call from both sync (main.py) and 
    async (FastAPI) contexts — detects running event loops automatically.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, retrieve_documents_async(search_queries, local_only=local_only))
            return future.result()
    else:
        return asyncio.run(retrieve_documents_async(search_queries, local_only=local_only))


def get_doc_contents(tagged_docs):
    """
    Utility: extract just the content strings from task-tagged documents.
    """
    return [d["content"] for d in tagged_docs if isinstance(d, dict) and "content" in d]
