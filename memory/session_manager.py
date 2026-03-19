"""
Session Manager — CRUD for chat sessions and messages.
Stores conversation history in SQLite for multi-turn research.
"""

import uuid
import json
from datetime import datetime
from memory.database import get_connection


class SessionManager:
    """Manages research sessions with persistent message history."""

    def create_session(self, title: str = "New Research") -> str:
        """Create a new session. Returns the session ID."""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
            conn.commit()
        finally:
            conn.close()
        return session_id

    def list_sessions(self) -> list:
        """Return all sessions ordered by most recent first."""
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_session(self, session_id: str) -> dict | None:
        """Get a single session with its messages."""
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None

            messages = conn.execute(
                "SELECT id, role, content, metadata_json, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

            return {
                **dict(row),
                "messages": [
                    {
                        **dict(m),
                        "metadata": json.loads(m["metadata_json"]) if m["metadata_json"] else None,
                    }
                    for m in messages
                ],
            }
        finally:
            conn.close()

    def rename_session(self, session_id: str, title: str) -> bool:
        """Rename a session."""
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, datetime.now().isoformat(), session_id),
            )
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        conn = get_connection()
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    def add_message(
        self, session_id: str, role: str, content: str, metadata: dict = None
    ) -> int:
        """Add a message to a session. Returns the message ID."""
        now = datetime.now().isoformat()
        meta_json = json.dumps(metadata, default=str) if metadata else None
        conn = get_connection()
        try:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, meta_json, now),
            )
            # Touch the session's updated_at
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_messages(self, session_id: str) -> list:
        """Get all messages for a session."""
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT id, role, content, metadata_json, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [
                {
                    **dict(r),
                    "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else None,
                }
                for r in rows
            ]
        finally:
            conn.close()

    def build_context(self, session_id: str, limit: int = 10) -> str:
        """
        Build conversation context string from recent messages.
        Returns formatted prior Q&A for injection into the pipeline.
        """
        messages = self.get_messages(session_id)
        if not messages:
            return ""

        # Take the last `limit` messages
        recent = messages[-limit:]

        parts = ["## Prior Conversation Context\n"]
        for msg in recent:
            if msg["role"] == "user":
                parts.append(f"**User**: {msg['content']}")
            elif msg["role"] == "assistant":
                # Only include a summary of the answer, not the full thing
                answer = msg["content"]
                if len(answer) > 500:
                    answer = answer[:500] + "..."
                parts.append(f"**Assistant**: {answer}")
            parts.append("")

        return "\n".join(parts)

    def auto_title(self, session_id: str, first_query: str):
        """Generate a short title from the first query."""
        # Simple: take first 50 chars of the query
        title = first_query[:50].strip()
        if len(first_query) > 50:
            title += "..."
        self.rename_session(session_id, title)
        return title
