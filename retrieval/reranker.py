from sentence_transformers import CrossEncoder
import logging
import os

logger = logging.getLogger(__name__)

_model = None

def _get_model():
    """Lazy-load the cross-encoder to avoid OOM on startup (Render 512MB limit)."""
    global _model
    if _model is None:
        logger.info("Loading cross-encoder model (first use)...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model

# Cross-encoder scores are log-odds. Typical range: -10 to +10.
# Docs below this threshold are considered irrelevant.
MIN_RELEVANCE_SCORE = -2.0


def rerank_documents(query, documents, top_k=5):
    """
    Rerank documents using a cross-encoder model.
    Filters out documents below MIN_RELEVANCE_SCORE.
    Returns up to top_k relevant documents.
    """
    if not documents:
        return []

    try:
        sentence_pairs = [[query, doc] for doc in documents]
        scores = _get_model().predict(sentence_pairs)

        # Pair, sort, and filter
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Log scores for observability
        for i, (doc, score) in enumerate(ranked[:8]):
            preview = doc[:60].replace("\n", " ")
            logger.info(f"  Rerank #{i+1}: score={score:.2f} | {preview}...")

        # Filter by minimum score, then take top_k
        top_documents = [
            doc for doc, score in ranked[:top_k]
            if score >= MIN_RELEVANCE_SCORE
        ]

        if not top_documents:
            # If nothing passes threshold, return the single best doc
            top_documents = [ranked[0][0]] if ranked else []

        return top_documents
    except Exception as e:
        logger.error(f"Error reranking documents, returning top docs unranked: {e}")
        return documents[:top_k]
