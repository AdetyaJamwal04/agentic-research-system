"""
SQLite Memory Tests — Verifies the migration to SQLite storage.

Tests LongTermMemory, ResearchHistory, and EvidencePool against
a temporary SQLite database (no real data touched).

Run: python -m pytest tests/test_sqlite_memory.py -v
"""

import os
import sys
import tempfile
import sqlite3

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def setup_test_db():
    """Create a temporary SQLite DB and point config to it."""
    import config
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    config.DB_PATH = tmp.name
    return tmp.name


def teardown_test_db(db_path):
    """Remove temporary DB."""
    try:
        os.unlink(db_path)
    except Exception:
        pass


def test_long_term_memory():
    """Test LongTermMemory: save, load, retrieve, dedup, quality gate."""
    db_path = setup_test_db()
    try:
        from memory.long_term_memory import LongTermMemory

        ltm = LongTermMemory()

        # Initially empty
        assert ltm._claims == [], "Should start empty"
        assert ltm.summary() == "Long-term memory: empty"

        # Quality gate: insufficient verdict → no saves
        claims = [{"claim": "Test claim", "mechanism": "Test", "confidence": 0.9, "source_id": 1}]
        saved = ltm.save_verified_claims(claims, "test query", verdict="insufficient")
        assert saved == 0, "Should not save when verdict is insufficient"

        # Quality gate: sufficient verdict → saves high confidence only
        claims = [
            {"claim": "High confidence claim", "mechanism": "Proven", "confidence": 0.9, "source_id": 1},
            {"claim": "Low confidence claim", "mechanism": "Guess", "confidence": 0.3, "source_id": 2},
        ]
        saved = ltm.save_verified_claims(claims, "test query", verdict="sufficient")
        assert saved == 1, "Should save only high-confidence claim"

        # Deduplication
        saved = ltm.save_verified_claims(claims, "test query", verdict="sufficient")
        assert saved == 0, "Should not save duplicate claims"

        # Relevance retrieval
        relevant = ltm.retrieve_relevant("high confidence")
        assert len(relevant) >= 1, "Should find relevant claims"

        # Irrelevant query
        irrelevant = ltm.retrieve_relevant("completely unrelated banana smoothie")
        assert len(irrelevant) == 0, "Should not return irrelevant claims"

        # Persistence: new instance should load the same data
        ltm2 = LongTermMemory()
        assert len(ltm2._claims) == 1, "Should persist and reload claims"

        print("  [OK] LongTermMemory: all tests passed")
    finally:
        teardown_test_db(db_path)


def test_research_history():
    """Test ResearchHistory: log, dedup check, history retrieval."""
    db_path = setup_test_db()
    try:
        from memory.research_history import ResearchHistory

        history = ResearchHistory()

        # Initially empty
        assert history.get_history() == [], "Should start empty"
        assert not history.has_prior_research("test query")

        # Log a query
        history.log_query("Test Query", claims_found=5, verdict="sufficient")
        assert len(history.get_history()) == 1

        # Dedup check (case insensitive)
        assert history.has_prior_research("test query"), "Should find case-insensitive match"
        assert not history.has_prior_research("different query")

        # Persistence
        history2 = ResearchHistory()
        assert len(history2.get_history()) == 1, "Should persist and reload"

        print("  [OK] ResearchHistory: all tests passed")
    finally:
        teardown_test_db(db_path)


def test_evidence_pool_snapshots():
    """Test EvidencePool: save/load snapshots via SQLite."""
    db_path = setup_test_db()
    try:
        from memory.evidence_pool import EvidencePool

        pool = EvidencePool()

        # Add documents and claims
        docs = [
            {"content": "Document about AI", "url": "http://example.com", "title": "AI Doc", "source_type": "web", "task_id": 1, "query": "AI"},
            {"content": "Document about ML", "url": "http://example2.com", "title": "ML Doc", "source_type": "web", "task_id": 1, "query": "ML"},
        ]
        added = pool.add(docs)
        assert added == 2, "Should add 2 documents"

        claims = [
            {"claim": "AI is transformative", "mechanism": "Innovation", "confidence": 0.9, "source_id": 1},
        ]
        pool.add_claims(claims)

        # Save snapshot
        pool.save(query="test snapshot")

        # Load into fresh pool
        pool2 = EvidencePool()
        pool2.load()
        assert len(pool2._documents) == 2, "Should load 2 documents from snapshot"
        assert len(pool2._claims) == 1, "Should load 1 claim from snapshot"

        print("  [OK] EvidencePool snapshots: all tests passed")
    finally:
        teardown_test_db(db_path)


def test_clear_operations():
    """Test that DELETE operations actually clear data."""
    db_path = setup_test_db()
    try:
        from memory.database import get_connection
        from memory.long_term_memory import LongTermMemory
        from memory.research_history import ResearchHistory

        # Insert some data
        ltm = LongTermMemory()
        ltm.save_verified_claims(
            [{"claim": "Test", "mechanism": "Test", "confidence": 0.9, "source_id": 1}],
            "test", verdict="sufficient"
        )

        history = ResearchHistory()
        history.log_query("test", 1, "sufficient")

        # Clear via SQL (same as API endpoints)
        conn = get_connection()
        conn.execute("DELETE FROM claims")
        conn.execute("DELETE FROM research_history")
        conn.commit()
        conn.close()

        # Verify cleared
        ltm2 = LongTermMemory()
        assert len(ltm2._claims) == 0, "Claims should be cleared"

        history2 = ResearchHistory()
        assert len(history2.get_history()) == 0, "History should be cleared"

        print("  [OK] Clear operations: all tests passed")
    finally:
        teardown_test_db(db_path)


if __name__ == "__main__":
    print("=" * 50)
    print("SQLite Memory Tests")
    print("=" * 50)

    test_long_term_memory()
    test_research_history()
    test_evidence_pool_snapshots()
    test_clear_operations()

    print()
    print("All tests passed! [OK]")
    print("=" * 50)
