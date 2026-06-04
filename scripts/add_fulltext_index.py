#!/usr/bin/env python3
"""
Add PostgreSQL full-text search (tsvector) index to documents table.
This enables fast keyword/phrase search over all 500K documents.
"""
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/app/backend")

from sqlalchemy import text
from app.core.database import engine


def add_fulltext_index():
    """Add tsvector column and GIN index for full-text search."""
    with engine.connect() as conn:
        # Check if tsvector column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'documents' AND column_name = 'tsv'
        """))
        if result.fetchone():
            logger.info("tsv column already exists. Skipping.")
            return

        logger.info("Adding tsvector column to documents...")
        conn.execute(text("""
            ALTER TABLE documents
            ADD COLUMN tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', COALESCE(text, '')))
            STORED
        """))
        conn.commit()
        logger.info("tsvector column added.")

        logger.info("Creating GIN index on tsvector...")
        conn.execute(text("""
            CREATE INDEX idx_documents_tsv
            ON documents
            USING GIN (tsv)
        """))
        conn.commit()
        logger.info("GIN index created.")

        # Analyze for query planner
        conn.execute(text("ANALYZE documents"))
        conn.commit()
        logger.info("Table analyzed.")


def test_fulltext_search():
    """Test the full-text search."""
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        query = "multipart & upload & limits"
        start = time.time()
        result = db.execute(text("""
            SELECT id, text
            FROM documents
            WHERE tsv @@ to_tsquery('english', :query)
            LIMIT 10
        """), {"query": query})
        rows = result.fetchall()
        elapsed = time.time() - start
        logger.info(f"FTS query '{query}' returned {len(rows)} docs in {elapsed*1000:.1f}ms")
        for row in rows[:3]:
            logger.info(f"  {row[0][:50]}...")
    finally:
        db.close()


if __name__ == "__main__":
    add_fulltext_index()
    test_fulltext_search()
