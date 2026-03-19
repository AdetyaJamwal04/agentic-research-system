"""
BM25 Retriever — Sparse keyword-based retrieval.
"""

import os
import pickle
from pathlib import Path

BM25_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bm25_index.pkl")


def bm25_retrieve(query: str, k: int = 5) -> list:
    """
    Search with BM25 keyword matching.
    Returns list of dicts: [{content, source, chunk_id, score}, ...]
    """
    if not Path(BM25_PATH).exists():
        return []

    try:
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)

        bm25_index = data["index"]
        chunks = data["chunks"]
        metadata = data["metadata"]

        tokenized_query = query.lower().split()
        scores = bm25_index.get_scores(tokenized_query)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        docs = []
        for idx, score in ranked:
            if score > 0:
                docs.append({
                    "content": chunks[idx],
                    "source": metadata[idx].get("source", "unknown"),
                    "chunk_id": f"bm25_{idx}",
                    "score": float(score),
                })
        return docs
    except Exception as e:
        print(f"BM25 retriever error: {e}")
        return []
