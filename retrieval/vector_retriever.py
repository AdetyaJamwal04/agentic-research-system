"""
Vector Retriever — Dense semantic search via ChromaDB.
"""

import os
from pathlib import Path

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "rag_documents"


def vector_retrieve(query: str, k: int = 5) -> list:
    """
    Search ChromaDB with semantic similarity.
    Returns list of dicts: [{content, source, chunk_id, score}, ...]
    """
    if not Path(CHROMA_DIR).exists():
        return []

    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(COLLECTION_NAME)

        results = collection.query(
            query_texts=[query],
            n_results=k,
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "chunk_id": results["ids"][0][i],
                "score": 1.0 - (results["distances"][0][i] if results.get("distances") else 0),
            })
        return docs
    except Exception as e:
        print(f"Vector retriever error: {e}")
        return []

def ingest_documents(docs: list) -> int:
    """
    Ingest chunked documents into ChromaDB.
    docs is a list of dicts: [{"content": "...", "metadata": {"source_id": "...", "type": "pdf"}}, ...]
    """
    import chromadb
    import uuid
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    docs_to_add = []
    metas_to_add = []
    ids_to_add = []
    
    for doc in docs:
        docs_to_add.append(doc["content"])
        metas_to_add.append(doc.get("metadata", {"source": "uploaded"}))
        ids_to_add.append(str(uuid.uuid4()))
        
    if docs_to_add:
        try:
            collection.add(
                documents=docs_to_add,
                metadatas=metas_to_add,
                ids=ids_to_add
            )
            return len(docs_to_add)
        except Exception as e:
            print(f"Ingestion error: {e}")
            return 0
    return 0
