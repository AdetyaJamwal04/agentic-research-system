"""
Long-Term Memory — Persistent store of verified, high-confidence claims.

Uses Semantic Search via ChromaDB.
"""

import os
import uuid
import chromadb
from datetime import datetime

SAVE_CONFIDENCE_THRESHOLD = 0.8

class LongTermMemory:
    """
    Stores verified claims across research sessions using ChromaDB.
    """

    def __init__(self):
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ltm_chroma_db")
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="verified_claims",
            metadata={"hnsw:space": "cosine"}
        )

    # ---- Relevance-Gated Load ----

    def retrieve_relevant(self, query: str, top_k: int = 10) -> list:
        """
        Return only claims relevant to the current query using semantic search.
        """
        if not self.collection.count():
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count())
            )
            
            relevant = []
            if results and "documents" in results and results["documents"]:
                # ChromaDB distance is 1 - cosine similarity. Usually < 0.5 is good.
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0] if "distances" in results else [0] * len(docs)
                
                for i in range(len(docs)):
                    # Distance threshold for relevance
                    if distances[i] > 0.5:
                        continue
                        
                    claim = {
                        "claim": docs[i],
                        "mechanism": metas[i].get("mechanism", ""),
                        "source_id": metas[i].get("source_id", 0),
                        "confidence": metas[i].get("confidence", 0.0),
                        "query_context": metas[i].get("query_context", ""),
                        "timestamp": metas[i].get("timestamp", "")
                    }
                    relevant.append(claim)
            return relevant
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []

    def format_for_context(self, claims: list) -> str:
        """Format relevant long-term claims as supplementary context."""
        if not claims:
            return ""

        parts = ["## Prior Research (Long-Term Memory)", ""]
        for c in claims:
            conf = f"{float(c.get('confidence', 0)):.0%}"
            parts.append(f"- ({conf}) {c['claim']}")
            if c.get("mechanism"):
                parts.append(f"  Mechanism: {c['mechanism']}")

        return "\n".join(parts)

    # ---- Quality-Gated Save ----

    def save_verified_claims(self, claims: list, query: str,
                              verdict: str, min_confidence: float = None):
        """
        Save only high-quality claims from critic-approved runs.
        Deduplicates against existing claims in the database.
        """
        if verdict != "sufficient":
            return 0

        threshold = min_confidence or SAVE_CONFIDENCE_THRESHOLD
        
        docs_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for claim in claims:
            confidence = claim.get("confidence", 0)
            if confidence < threshold:
                continue
                
            claim_text = claim.get("claim", "")
            if not claim_text:
                continue

            # ── Deduplication: skip if a near-identical claim already exists ──
            if self.collection.count() > 0:
                try:
                    existing = self.collection.query(
                        query_texts=[claim_text],
                        n_results=1,
                    )
                    if (existing and existing["distances"]
                            and existing["distances"][0]
                            and existing["distances"][0][0] < 0.05):
                        # Very close match — skip this duplicate
                        continue
                except Exception:
                    pass  # If dedup check fails, proceed with adding

            doc_id = str(uuid.uuid4())
            docs_to_add.append(claim_text)
            metadatas_to_add.append({
                "mechanism": claim.get("mechanism", ""),
                "confidence": confidence,
                "source_id": claim.get("source_id", 0),
                "query_context": query,
                "timestamp": datetime.now().isoformat()
            })
            ids_to_add.append(doc_id)

        if docs_to_add:
            try:
                self.collection.add(
                    documents=docs_to_add,
                    metadatas=metadatas_to_add,
                    ids=ids_to_add
                )
                return len(docs_to_add)
            except Exception as e:
                print(f"Error saving claims to ChromaDB: {e}")
                return 0
                
        return 0

    # ---- Stats ----

    def summary(self) -> str:
        count = self.collection.count()
        if count == 0:
            return "Long-term memory: empty"

        return f"Long-term memory (ChromaDB): {count} verified claims"

