"""
Tests for document deduplication via content hash.
Covers the _content_hash function, ingestion-time dedup, and the /dedup endpoint.
"""
import hashlib
from unittest.mock import patch, MagicMock

import pytest

from app.api.routes import _content_hash
from tests.conftest import Document


class TestContentHash:
    """Tests for the _content_hash helper function."""

    def test_returns_sha256_hex(self):
        """Test that _content_hash returns a SHA-256 hex digest."""
        text = "Hello, world!"
        result = _content_hash(text)
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()

        assert result == expected
        assert len(result) == 64

    def test_deterministic(self):
        """Test that the same input always produces the same hash."""
        text = "Consistent document text"
        assert _content_hash(text) == _content_hash(text)

    def test_different_text_different_hash(self):
        """Test that different texts produce different hashes."""
        hash_a = _content_hash("Text A")
        hash_b = _content_hash("Text B")
        assert hash_a != hash_b

    def test_empty_string(self):
        """Test hashing an empty string."""
        result = _content_hash("")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected
        assert len(result) == 64

    def test_unicode_content(self):
        """Test hashing text with unicode characters."""
        text = "Document with unicode: éèê 日本語"
        result = _content_hash(text)
        assert len(result) == 64

        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected

    def test_whitespace_matters(self):
        """Test that whitespace differences produce different hashes."""
        hash1 = _content_hash("Hello world")
        hash2 = _content_hash("Hello  world")  # double space
        hash3 = _content_hash("Hello world ")   # trailing space
        hash4 = _content_hash(" Hello world")   # leading space

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash1 != hash4

    def test_case_sensitive(self):
        """Test that case differences produce different hashes."""
        hash_lower = _content_hash("hello world")
        hash_upper = _content_hash("HELLO WORLD")
        hash_mixed = _content_hash("Hello World")

        assert hash_lower != hash_upper
        assert hash_lower != hash_mixed
        assert hash_upper != hash_mixed

    def test_long_text(self):
        """Test hashing a very long document."""
        text = "A" * 1_000_000
        result = _content_hash(text)
        assert len(result) == 64

    def test_special_characters(self):
        """Test hashing text with special characters."""
        text = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?\n\t\r"
        result = _content_hash(text)
        assert len(result) == 64


class TestIngestionDedup:
    """Tests for deduplication during the ingestion flow."""

    @patch("app.tasks.celery_app.celery_app")
    def test_exact_duplicate_skipped(self, mock_celery, client, db_session_for_api):
        """Test that documents with identical content are marked as duplicates."""
        session, _ = db_session_for_api
        mock_celery.send_task = MagicMock(return_value=MagicMock())

        payload = {
            "documents": [
                {"id": "doc-1", "text": "This is unique content"},
                {"id": "doc-2", "text": "This is unique content"},  # exact dup
            ]
        }
        response = client.post("/api/v1/ingest", json=payload)

        data = response.json()
        assert data["new_docs"] == 1
        assert data["duplicate_docs"] == 1
        assert data["total_docs"] == 2

    @patch("app.tasks.celery_app.celery_app")
    def test_same_id_different_content_updates(self, mock_celery, client, db_session_for_api):
        """Test that re-ingesting with same ID but different text updates the doc."""
        session, _ = db_session_for_api
        mock_celery.send_task = MagicMock(return_value=MagicMock())

        # First ingestion
        payload1 = {
            "documents": [
                {"id": "doc-1", "text": "Original content"},
            ]
        }
        client.post("/api/v1/ingest", json=payload1)

        # Second ingestion with same ID, different content
        payload2 = {
            "documents": [
                {"id": "doc-1", "text": "Updated content"},
            ]
        }
        response = client.post("/api/v1/ingest", json=payload2)

        data = response.json()
        assert data["new_docs"] == 1  # Updated, not duplicate

        # Verify the document text was updated
        doc = session.query(Document).filter(Document.id == "doc-1").first()
        assert doc.text == "Updated content"

    @patch("app.tasks.celery_app.celery_app")
    def test_mixed_new_and_duplicate(self, mock_celery, client, db_session_for_api):
        """Test ingestion with a mix of new and duplicate documents."""
        session, _ = db_session_for_api
        mock_celery.send_task = MagicMock(return_value=MagicMock())

        payload = {
            "documents": [
                {"id": "doc-1", "text": "Unique document one"},
                {"id": "doc-2", "text": "Unique document two"},
                {"id": "doc-3", "text": "Unique document one"},   # dup of doc-1
                {"id": "doc-4", "text": "Unique document three"},
                {"id": "doc-5", "text": "Unique document two"},   # dup of doc-2
            ]
        }
        response = client.post("/api/v1/ingest", json=payload)

        data = response.json()
        assert data["total_docs"] == 5
        assert data["new_docs"] == 3
        assert data["duplicate_docs"] == 2

    @patch("app.tasks.celery_app.celery_app")
    def test_duplicate_detection_via_source_hash(self, mock_celery, client, db_session_for_api):
        """Test that dedup works by storing hash in source field."""
        session, _ = db_session_for_api
        mock_celery.send_task = MagicMock(return_value=MagicMock())

        text = "Content to verify hash storage"
        payload = {
            "documents": [
                {"id": "doc-h1", "text": text},
            ]
        }
        client.post("/api/v1/ingest", json=payload)

        # Check that the stored document has hash: prefix in source
        doc = session.query(Document).filter(Document.id == "doc-h1").first()
        assert doc.source.startswith("hash:")
        expected_hash = _content_hash(text)
        assert doc.source == f"hash:{expected_hash}"


class TestDedupEndpoint:
    """Tests for the POST /api/v1/dedup cleanup endpoint."""

    def test_dedup_removes_extra_copies(self, client, db_session_for_api):
        """Test that /dedup removes documents with identical content, keeping first."""
        session, _ = db_session_for_api

        # Manually insert docs with duplicate content
        text = "Duplicate content for dedup test"
        doc1 = Document(id="keep-me", text=text, source="hash:abc")
        doc2 = Document(id="remove-me", text=text, source="hash:abc")
        doc3 = Document(id="also-keep", text="Different content", source="hash:xyz")
        session.add_all([doc1, doc2, doc3])
        session.flush()

        response = client.post("/api/v1/dedup")

        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 3
        assert data["duplicates_removed"] == 1
        assert data["unique_documents"] == 2

        # Verify the right document was kept
        remaining = session.query(Document).all()
        remaining_ids = [d.id for d in remaining]
        assert "keep-me" in remaining_ids
        assert "also-keep" in remaining_ids
        assert "remove-me" not in remaining_ids

    def test_dedup_no_duplicates(self, client, db_session_for_api):
        """Test /dedup when there are no duplicates."""
        session, _ = db_session_for_api

        session.add(Document(id="d1", text="Content A", source="hash:a"))
        session.add(Document(id="d2", text="Content B", source="hash:b"))
        session.flush()

        response = client.post("/api/v1/dedup")

        data = response.json()
        assert data["duplicates_removed"] == 0
        assert data["unique_documents"] == 2

    def test_dedup_empty_database(self, client):
        """Test /dedup when the database has no documents."""
        response = client.post("/api/v1/dedup")

        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 0
        assert data["duplicates_removed"] == 0

    def test_dedup_multiple_copies(self, client, db_session_for_api):
        """Test dedup when there are more than 2 copies of the same content."""
        session, _ = db_session_for_api

        text = "Triple duplicate text"
        session.add(Document(id="d1", text=text, source="hash:x"))
        session.add(Document(id="d2", text=text, source="hash:x"))
        session.add(Document(id="d3", text=text, source="hash:x"))
        session.add(Document(id="d4", text="Unique", source="hash:y"))
        session.flush()

        response = client.post("/api/v1/dedup")

        data = response.json()
        assert data["duplicates_removed"] == 2
        assert data["unique_documents"] == 2

    def test_dedup_preserves_first_occurrence(self, client, db_session_for_api):
        """Test that dedup always keeps the first inserted document."""
        session, _ = db_session_for_api

        text = "First one stays"
        doc1 = Document(id="first", text=text, source="hash:keep")
        doc2 = Document(id="second", text=text, source="hash:keep")
        session.add(doc1)
        session.flush()
        session.add(doc2)
        session.flush()

        response = client.post("/api/v1/dedup")

        remaining = session.query(Document).filter(Document.text == text).all()
        assert len(remaining) == 1
        assert remaining[0].id == "first"
