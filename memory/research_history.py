"""
Research History — Tracks which queries have been researched and what was found.
Enables cross-session knowledge to avoid duplicate research.

Storage: SQLite (data/research.db → research_history table)
"""

from datetime import datetime
from memory.database import get_connection


class ResearchHistory:
    """Logs past research sessions: queries, claims found, timestamps."""

    def __init__(self):
        self._entries = []
        self._load()

    def _load(self):
        """Load all history entries from SQLite."""
        try:
            conn = get_connection()
            rows = conn.execute(
                "SELECT query, claims_found, verdict, timestamp FROM research_history ORDER BY id"
            ).fetchall()
            self._entries = [dict(row) for row in rows]
            conn.close()
        except Exception:
            self._entries = []

    def log_query(self, query: str, claims_found: int, verdict: str = ""):
        """Log a completed research query."""
        entry = {
            "query": query,
            "claims_found": claims_found,
            "verdict": verdict,
            "timestamp": datetime.now().isoformat(),
        }

        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO research_history (query, claims_found, verdict, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (entry["query"], entry["claims_found"], entry["verdict"], entry["timestamp"])
            )
            conn.commit()
            self._entries.append(entry)
        except Exception as e:
            conn.rollback()
            print(f"Error logging query to SQLite: {e}")
        finally:
            conn.close()

    def has_prior_research(self, query: str) -> bool:
        """Check if a similar query has been researched before."""
        query_lower = query.lower().strip()
        for entry in self._entries:
            if entry.get("query", "").lower().strip() == query_lower:
                return True
        return False

    def get_history(self) -> list:
        """Return all research history entries."""
        return self._entries

    def summary(self) -> str:
        if not self._entries:
            return "No prior research history."
        parts = [f"Research History: {len(self._entries)} past queries"]
        for entry in self._entries[-5:]:  # Show last 5
            parts.append(f"  • {entry['query'][:60]}... ({entry.get('claims_found', 0)} claims)")
        return "\n".join(parts)
