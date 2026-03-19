"""
Hybrid Retriever — Combines dense (vector) and sparse (BM25) retrieval
using Reciprocal Rank Fusion (RRF).
"""

import os
from pathlib import Path
from retrieval.vector_retriever import vector_retrieve
from retrieval.bm25_retriever import bm25_retrieve

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
BM25_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bm25_index.pkl")


def _has_local_index() -> bool:
    """Check if local indexes exist."""
    return Path(CHROMA_DIR).exists() and Path(BM25_PATH).exists()


def reciprocal_rank_fusion(result_lists: list, k: int = 60) -> list:
    """
    RRF score = sum(1 / (k + rank)) for each document across all lists.
    """
    scores = {}
    doc_map = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            content = doc["content"][:200]
            rrf_score = 1.0 / (k + rank + 1)

            if content in scores:
                scores[content] += rrf_score
            else:
                scores[content] = rrf_score
                doc_map[content] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[content] for content, _ in ranked]


def hybrid_search(query: str, k: int = 5) -> list:
    """
    Run both dense and sparse search, fuse with RRF.
    Falls back to vector-only if BM25 index doesn't exist.
    Returns empty list if no local index exists at all.
    """
    has_chroma = Path(CHROMA_DIR).exists()
    has_bm25 = Path(BM25_PATH).exists()

    if not has_chroma:
        return []

    dense_results = vector_retrieve(query, k=k * 2)

    if has_bm25:
        sparse_results = bm25_retrieve(query, k=k * 2)
    else:
        sparse_results = []

    if not dense_results and not sparse_results:
        return []

    # If we have both, fuse with RRF. If only dense, just use those.
    if sparse_results:
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
    else:
        fused = dense_results

    for doc in fused:
        doc["source_type"] = "local"
        doc["url"] = ""
        doc["title"] = doc.get("source", "local document")

    return fused[:k]
