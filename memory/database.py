"""
Database — Shared SQLite connection and table initialization.

Provides a centralized database layer for persistent storage.
Uses WAL mode for concurrent read support.
"""

import sqlite3
import os

from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get a connection to the SQLite database. Creates tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    conn.execute("PRAGMA journal_mode=WAL")  # Concurrent read support
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection):
    """Create tables if they don't exist (idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim TEXT NOT NULL,
            mechanism TEXT,
            confidence REAL,
            source_id INTEGER,
            query_context TEXT,
            timestamp TEXT
        );

        CREATE TABLE IF NOT EXISTS research_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            claims_found INTEGER,
            verdict TEXT,
            timestamp TEXT
        );

        CREATE TABLE IF NOT EXISTS evidence_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            data_json TEXT,
            timestamp TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New Research',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
