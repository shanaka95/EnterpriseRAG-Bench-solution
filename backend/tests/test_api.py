"""
Tests for FastAPI endpoints using TestClient.
Mocks Celery tasks and embedding models so tests run without GPU/Redis.
"""
import json
import uuid

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from tests.conftest import ClusterNode, Document, IngestionJob


# ---------------------------------------------------------------------------
# Helper: patch celery_app.send_task as a no-op for all ingest tests
# ---------------------------------------------------------------------------
def _mock_celery_send_task(*args, **kwargs):
    """No-op Celery task sender for tests."""
    return MagicMock()


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_ok(self, client):
        """Test that health check returns 200 and status ok."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "database" in data
        assert "timestamp" in data

    def test_health_database_ok(self, client):
        """Test that database status is reported."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "database" in data


class TestIngestEndpoint:
    """Tests for POST /api/v1/ingest."""

    @patch("app.tasks.celery_app.celery_app")
    def test_ingest_empty_documents(self, mock_celery, client):
        """Test that empty document list returns 400."""
        mock_celery.send_task = _mock_celery_send_task
        response = client.post("/api/v1/ingest", json={"documents": []})
        assert response.status_code == 400

    @patch("app.tasks.celery_app.celery_app")
    def test_ingest_documents_success(self, mock_celery, client):
        """Test successful document ingestion."""
        mock_celery.send_task = _mock_celery_send_task
        payload = {
            "documents": [
                {"id": "doc-1", "text": "First document"},
                {"id": "doc-2", "text": "Second document"},
            ]
        }
        response = client.post("/api/v1/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["total_docs"] == 2
        assert data["new_docs"] == 2
        assert data["duplicate_docs"] == 0

    @patch("app.tasks.celery_app.celery_app")
    def test_ingest_creates_job_in_db(self, mock_celery, client, db_session_for_api):
        """Test that ingestion creates an IngestionJob record."""
        mock_celery.send_task = _mock_celery_send_task
        session, _ = db_session_for_api
        payload = {
            "documents": [
                {"id": "doc-a", "text": "Test document"},
            ]
        }
        response = client.post("/api/v1/ingest", json=payload)
        data = response.json()
        job_id = data["job_id"]

        # Verify job was created in DB (test model uses string IDs)
        job = session.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        assert job is not None
        assert job.total_docs == 1

    @patch("app.tasks.celery_app.celery_app")
    def test_ingest_dedup_same_content(self, mock_celery, client):
        """Test that documents with same content are deduped."""
        mock_celery.send_task = _mock_celery_send_task
        # First ingestion
        payload1 = {
            "documents": [
                {"id": "doc-1", "text": "Identical content"},
            ]
        }
        client.post("/api/v1/ingest", json=payload1)

        # Second ingestion with same content
        payload2 = {
            "documents": [
                {"id": "doc-2", "text": "Identical content"},
            ]
        }
        response = client.post("/api/v1/ingest", json=payload2)
        data = response.json()
        assert data["duplicate_docs"] == 1
        assert data["new_docs"] == 0


class TestDedupEndpoint:
    """Tests for POST /api/v1/dedup."""

    def test_content_hash_deterministic(self):
        """Test content hash produces consistent results."""
        from app.api.routes import _content_hash
        hash1 = _content_hash("Test content")
        hash2 = _content_hash("Test content")
        assert hash1 == hash2

    def test_content_hash_differs(self):
        """Test different texts produce different hashes."""
        from app.api.routes import _content_hash
        hash1 = _content_hash("Document A")
        hash2 = _content_hash("Document B")
        assert hash1 != hash2

    def test_dedup_removes_duplicates(self, client, db_session_for_api):
        """Test that dedup endpoint removes duplicate documents."""
        session, _ = db_session_for_api

        # Insert docs directly with same content
        doc1 = Document(id="doc-1", text="Same content", source="hash:abc")
        doc2 = Document(id="doc-2", text="Same content", source="hash:abc")
        doc3 = Document(id="doc-3", text="Different content", source="hash:xyz")
        session.add_all([doc1, doc2, doc3])
        session.flush()

        response = client.post("/api/v1/dedup")
        data = response.json()
        assert data["duplicates_removed"] == 1
        assert data["unique_documents"] == 2

    def test_dedup_no_duplicates(self, client, db_session_for_api):
        """Test dedup when there are no duplicates."""
        session, _ = db_session_for_api
        session.add(Document(id="d1", text="Content A", source="hash:a"))
        session.add(Document(id="d2", text="Content B", source="hash:b"))
        session.flush()

        response = client.post("/api/v1/dedup")
        data = response.json()
        assert data["duplicates_removed"] == 0

    def test_dedup_empty_database(self, client):
        """Test dedup when the database has no documents."""
        response = client.post("/api/v1/dedup")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 0
        assert data["duplicates_removed"] == 0


class TestTreeEndpoint:
    """Tests for GET /api/v1/tree."""

    def test_empty_tree(self, client):
        """Test tree endpoint with no nodes."""
        response = client.get("/api/v1/tree")
        assert response.status_code == 200
        data = response.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_tree_with_nodes(self, client, db_session_for_api):
        """Test tree endpoint returns nodes and edges."""
        session, _ = db_session_for_api

        parent_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())

        root = ClusterNode(
            id=parent_id,
            parent_id=None,
            doc_count=10,
            keywords=json.dumps(["test"]),
            is_leaf=False,
            depth=0,
        )
        child = ClusterNode(
            id=child_id,
            parent_id=parent_id,
            doc_count=5,
            keywords=json.dumps(["child"]),
            is_leaf=True,
            doc_ids=json.dumps(["doc-1"]),
            depth=1,
        )
        session.add_all([root, child])
        session.flush()

        response = client.get("/api/v1/tree")
        data = response.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["edges"][0]["source"] == parent_id
        assert data["edges"][0]["target"] == child_id


class TestStatsEndpoint:
    """Tests for GET /api/v1/stats."""

    def test_stats_empty(self, client):
        """Test stats with empty database."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_nodes"] == 0
        assert data["total_documents"] == 0

    def test_stats_with_data(self, client, db_session_for_api):
        """Test stats with data."""
        session, _ = db_session_for_api

        root = ClusterNode(
            id=str(uuid.uuid4()),
            parent_id=None,
            doc_count=10,
            keywords=json.dumps(["test"]),
            is_leaf=False,
            depth=0,
        )
        doc = Document(id="doc-1", text="Test document", source="test")
        session.add_all([root, doc])
        session.flush()

        response = client.get("/api/v1/stats")
        data = response.json()
        assert data["total_nodes"] == 1
        assert data["total_documents"] == 1


class TestNodeEndpoint:
    """Tests for GET /api/v1/nodes/{node_id}."""

    def test_get_node_invalid_uuid(self, client):
        """Test that invalid UUID returns 400."""
        response = client.get("/api/v1/nodes/not-a-uuid")
        assert response.status_code == 400

    def test_get_node_not_found(self, client):
        """Test that missing node returns 404."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/nodes/{fake_id}")
        assert response.status_code == 404

    def test_get_existing_node(self, client, db_session_for_api):
        """Test getting an existing node."""
        session, _ = db_session_for_api

        node_id = str(uuid.uuid4())
        node = ClusterNode(
            id=node_id,
            parent_id=None,
            doc_count=10,
            keywords=json.dumps(["test", "keyword"]),
            is_leaf=True,
            doc_ids=json.dumps(["doc-1"]),
            depth=0,
        )
        session.add(node)
        session.flush()

        response = client.get(f"/api/v1/nodes/{node_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == node_id
        assert data["doc_count"] == 10


class TestIngestStatusEndpoint:
    """Tests for GET /api/v1/ingest/{job_id}."""

    def test_get_job_status(self, client, db_session_for_api):
        """Test fetching ingestion job status."""
        session, _ = db_session_for_api

        job_id = str(uuid.uuid4())
        job = IngestionJob(
            id=job_id,
            status="completed",
            total_docs=5,
            processed_docs=5,
        )
        session.add(job)
        session.flush()

        response = client.get(f"/api/v1/ingest/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["status"] == "completed"

    def test_get_nonexistent_job(self, client):
        """Test that a 404 is returned for a nonexistent job."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/ingest/{fake_id}")
        assert response.status_code == 404

    def test_get_job_invalid_uuid(self, client):
        """Test that an invalid UUID format returns 400."""
        response = client.get("/api/v1/ingest/invalid")
        assert response.status_code == 400


class TestDocumentsEndpoint:
    """Tests for GET /api/v1/documents."""

    def test_list_documents_empty(self, client):
        """Test listing with no documents."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_documents_with_data(self, client, db_session_for_api):
        """Test listing documents."""
        session, _ = db_session_for_api

        for i in range(3):
            session.add(Document(id=f"doc-{i}", text=f"Document {i}"))
        session.flush()

        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_list_documents_pagination(self, client, db_session_for_api):
        """Test document list pagination."""
        session, _ = db_session_for_api

        for i in range(10):
            session.add(Document(id=f"page-doc-{i}", text=f"Doc {i}"))
        session.flush()

        response = client.get("/api/v1/documents?limit=3&offset=0")
        data = response.json()
        assert len(data) == 3

        response = client.get("/api/v1/documents?limit=3&offset=3")
        data = response.json()
        assert len(data) == 3

    def test_get_existing_document(self, client, db_session_for_api):
        """Test fetching a specific document."""
        session, _ = db_session_for_api
        session.add(Document(id="specific-doc", text="Specific content", source="test"))
        session.flush()

        response = client.get("/api/v1/documents/specific-doc")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "specific-doc"
        assert data["text"] == "Specific content"

    def test_get_nonexistent_document(self, client):
        """Test that a 404 is returned for a nonexistent document."""
        response = client.get("/api/v1/documents/nonexistent")
        assert response.status_code == 404


class TestQueryEndpoint:
    """Tests for POST /api/v1/query."""

    def test_query_no_tree(self, client):
        """Test query when no tree exists."""
        response = client.post("/api/v1/query", json={"query": "test"})
        assert response.status_code == 404

    @patch("app.api.routes.cross_encoder_score")
    def test_query_with_tree(self, mock_ce_score, client, db_session_for_api):
        """Test query with existing tree and mocked cross-encoder."""
        session, _ = db_session_for_api

        root_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())

        root = ClusterNode(
            id=root_id,
            parent_id=None,
            doc_count=3,
            keywords=json.dumps(["test"]),
            is_leaf=False,
            depth=0,
        )
        child = ClusterNode(
            id=child_id,
            parent_id=root_id,
            medoid_doc_id="doc-1",
            doc_count=3,
            keywords=json.dumps(["keyword"]),
            is_leaf=True,
            doc_ids=json.dumps(["doc-1", "doc-2", "doc-3"]),
            depth=1,
        )
        doc = Document(id="doc-1", text="Machine learning is a subset of AI.")
        session.add_all([root, child, doc])
        session.flush()

        # Mock cross-encoder to return a high score for the child
        mock_ce_score.return_value = np.array([0.9], dtype=np.float32)

        response = client.post("/api/v1/query", json={"query": "machine learning"})
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"
        assert "doc_ids" in data
        assert "paths_traversed" in data

    @patch("app.api.routes.cross_encoder_score")
    def test_query_with_custom_threshold(self, mock_ce_score, client, db_session_for_api):
        """Test query with custom threshold parameter."""
        session, _ = db_session_for_api

        root_id = str(uuid.uuid4())
        root = ClusterNode(
            id=root_id,
            parent_id=None,
            doc_count=1,
            keywords=json.dumps(["test"]),
            is_leaf=True,
            doc_ids=json.dumps(["doc-1"]),
            depth=0,
        )
        session.add(root)
        session.flush()

        # Low cross-encoder score won't pass a high threshold
        mock_ce_score.return_value = np.array([0.5], dtype=np.float32)

        response = client.post("/api/v1/query", json={"query": "test", "threshold": 0.9})
        assert response.status_code == 200
