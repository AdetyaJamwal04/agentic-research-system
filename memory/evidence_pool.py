"""
Evidence Pool — Manages documents AND atomic claims across retrieval iterations.

Handles deduplication, source tracking, accumulation, persistence,
and formatting for claim-based reasoning.

Snapshot persistence: SQLite (data/research.db → evidence_snapshots table)
"""

import json
from datetime import datetime


class EvidencePool:
    """
    Accumulates evidence documents and extracted claims across iterations.
    Supports persistence for cross-session knowledge.
    """

    def __init__(self):
        self._documents = []
        self._claims = []
        self._seen_content = set()
        self._seen_claims = set()
        self._iteration = 0

    # ---- Document Management ----

    def add(self, docs: list) -> int:
        """Add new documents to the pool. Duplicates are skipped."""
        added = 0
        for doc in docs:
            content = doc.get("content", "")
            fingerprint = content[:200].strip().lower()
            if fingerprint and fingerprint not in self._seen_content:
                self._seen_content.add(fingerprint)
                doc["iteration"] = self._iteration
                doc["source_id"] = len(self._documents) + 1
                self._documents.append(doc)
                added += 1
        return added

    def next_iteration(self):
        """Advance to the next retrieval iteration."""
        self._iteration += 1

    def get_all(self) -> list:
        return self._documents

    def get_contents(self) -> list:
        return [d["content"] for d in self._documents]

    def get_new(self) -> list:
        return [d for d in self._documents if d.get("iteration") == self._iteration]

    # ---- Claim Management ----

    def add_claims(self, claims: list) -> int:
        """Add extracted claims. Deduplicates by claim text."""
        added = 0
        for claim in claims:
            fingerprint = claim.get("claim", "")[:100].strip().lower()
            if fingerprint and fingerprint not in self._seen_claims:
                self._seen_claims.add(fingerprint)
                claim["iteration"] = self._iteration
                self._claims.append(claim)
                added += 1
        return added

    def get_claims(self) -> list:
        """Return all claims, sorted by confidence (highest first)."""
        return sorted(self._claims, key=lambda c: c.get("confidence", 0), reverse=True)

    def get_high_confidence_claims(self, threshold: float = 0.7) -> list:
        """Return only claims above the confidence threshold."""
        return [c for c in self._claims if c.get("confidence", 0) >= threshold]

    def format_claims_for_reasoning(self) -> str:
        """Format claims as numbered list for the reasoning agent."""
        claims = self.get_claims()
        if not claims:
            return self.format_for_reasoning()  # fallback to doc-level

        parts = []
        for i, c in enumerate(claims, 1):
            src = c.get("source_id", "?")
            conf = f"{c.get('confidence', 0):.0%}"
            mechanism = c.get("mechanism", "")
            parts.append(f"[{src}] (confidence: {conf}) {c['claim']}")
            if mechanism:
                parts.append(f"    Mechanism: {mechanism}")
            parts.append("")
        return "\n".join(parts)

    # ---- Source Formatting ----

    def get_sources(self) -> list:
        sources = []
        for doc in self._documents:
            sources.append({
                "source_id": doc.get("source_id", 0),
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "source_type": doc.get("source_type", "unknown"),
            })
        return sources

    def format_for_reasoning(self) -> str:
        """Format all evidence as numbered passages with source labels."""
        passages = []
        for doc in self._documents:
            sid = doc.get("source_id", "?")
            source_type = doc.get("source_type", "web")
            title = doc.get("title", "")
            url = doc.get("url", "")

            if source_type == "web" and url:
                label = f"[{sid}] (web: {title or url})"
            elif source_type == "local":
                label = f"[{sid}] (local: {title})"
            else:
                label = f"[{sid}]"

            passages.append(f"{label}\n{doc.get('content', '')}")

        return "\n\n---\n\n".join(passages)

    def format_sources_list(self) -> str:
        lines = []
        for doc in self._documents:
            sid = doc.get("source_id", "?")
            title = doc.get("title", "Untitled")
            url = doc.get("url", "")
            source_type = doc.get("source_type", "unknown")
            if url:
                lines.append(f"  [{sid}] {title} — {url} ({source_type})")
            else:
                lines.append(f"  [{sid}] {title} ({source_type})")
        return "\n".join(lines)

    # ---- Persistence (SQLite snapshots) ----

    def save(self, query: str = ""):
        """Save documents and claims as a snapshot to SQLite."""
        from memory.database import get_connection

        data = {
            "documents": self._documents,
            "claims": self._claims,
            "iteration": self._iteration,
        }

        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO evidence_snapshots (query, data_json, timestamp)
                   VALUES (?, ?, ?)""",
                (query, json.dumps(data, default=str), datetime.now().isoformat())
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error saving evidence snapshot: {e}")
        finally:
            conn.close()

    def load(self, snapshot_id: int = None):
        """Load the most recent evidence snapshot from SQLite."""
        from memory.database import get_connection

        conn = get_connection()
        try:
            if snapshot_id:
                row = conn.execute(
                    "SELECT data_json FROM evidence_snapshots WHERE id = ?", (snapshot_id,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT data_json FROM evidence_snapshots ORDER BY id DESC LIMIT 1"
                ).fetchone()

            if not row:
                return

            data = json.loads(row["data_json"])

            for doc in data.get("documents", []):
                fingerprint = doc.get("content", "")[:200].strip().lower()
                if fingerprint:
                    self._seen_content.add(fingerprint)
                self._documents.append(doc)

            for claim in data.get("claims", []):
                fingerprint = claim.get("claim", "")[:100].strip().lower()
                if fingerprint:
                    self._seen_claims.add(fingerprint)
                self._claims.append(claim)

            self._iteration = data.get("iteration", 0) + 1
        except Exception as e:
            print(f"Error loading evidence snapshot: {e}")
        finally:
            conn.close()

    # ---- Stats ----

    def summary(self) -> str:
        total_docs = len(self._documents)
        total_claims = len(self._claims)
        by_iteration = {}
        for doc in self._documents:
            it = doc.get("iteration", 0)
            by_iteration[it] = by_iteration.get(it, 0) + 1

        parts = [f"Evidence Pool: {total_docs} docs, {total_claims} claims"]
        for it, count in sorted(by_iteration.items()):
            parts.append(f"  Iteration {it}: {count} docs")
        return "\n".join(parts)
