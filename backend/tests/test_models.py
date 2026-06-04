"""
Tests for SQLAlchemy models: ClusterNode, Document, IngestionJob.
Uses SQLite-compatible models defined in conftest.py.
"""
import json
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select

from tests.conftest import ClusterNode, Document, IngestionJob


class TestClusterNode:
    """Tests for the ClusterNode model."""

    def test_create_root_node(self, db_session):
        """Test creating a root cluster node (no parent)."""
        node = ClusterNode(
            id=str(uuid.uuid4()),
            parent_id=None,
            doc_count=10,
            keywords=json.dumps(["ml", "ai", "deep-learning"]),
            is_leaf=False,
            doc_ids=None,
            depth=0,
        )
        db_session.add(node)
        db_session.flush()

        result = db_session.query(ClusterNode).first()
        assert result is not None
        assert result.parent_id is None
        assert result.doc_count == 10
        assert result.is_leaf is False
        assert result.depth == 0
        assert json.loads(result.keywords) == ["ml", "ai", "deep-learning"]

    def test_create_leaf_node(self, db_session):
        """Test creating a leaf cluster node with doc_ids."""
        node = ClusterNode(
            id=str(uuid.uuid4()),
            parent_id=None,
            doc_count=3,
            keywords=json.dumps(["nlp", "text"]),
            is_leaf=True,
            doc_ids=json.dumps(["doc1", "doc2", "doc3"]),
            depth=1,
        )
        db_session.add(node)
        db_session.flush()

        result = db_session.query(ClusterNode).first()
        assert result.is_leaf is True
        assert json.loads(result.doc_ids) == ["doc1", "doc2", "doc3"]

    def test_parent_child_relationship(self, db_session):
        """Test that parent-child relationships work correctly."""
        parent_id = str(uuid.uuid4())
        parent = ClusterNode(
            id=parent_id,
            parent_id=None,
            doc_count=20,
            keywords=json.dumps(["root"]),
            is_leaf=False,
            depth=0,
        )
        db_session.add(parent)
        db_session.flush()

        child1 = ClusterNode(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            doc_count=10,
            keywords=json.dumps(["child1"]),
            is_leaf=True,
            doc_ids=json.dumps(["d1", "d2"]),
            depth=1,
        )
        child2 = ClusterNode(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            doc_count=10,
            keywords=json.dumps(["child2"]),
            is_leaf=True,
            doc_ids=json.dumps(["d3", "d4"]),
            depth=1,
        )
        db_session.add(child1)
        db_session.add(child2)
        db_session.flush()

        parent_result = db_session.query(ClusterNode).filter(
            ClusterNode.id == parent_id
        ).first()
        assert parent_result is not None
        assert len(parent_result.children) == 2

        # Verify children reference parent correctly
        for child in parent_result.children:
            assert child.parent_id == parent_id
            assert child.depth == 1

    def test_to_dict(self, db_session):
        """Test the to_dict serialization method."""
        node_id = str(uuid.uuid4())
        # Create the parent first to satisfy FK constraint
        parent_id = str(uuid.uuid4())
        parent = ClusterNode(
            id=parent_id,
            parent_id=None,
            doc_count=0,
            keywords=json.dumps([]),
            is_leaf=True,
            depth=0,
        )
        db_session.add(parent)
        db_session.flush()

        node = ClusterNode(
            id=node_id,
            parent_id=parent_id,
            medoid_doc_id="doc-42",
            doc_count=5,
            keywords=json.dumps(["keyword1", "keyword2"]),
            is_leaf=True,
            doc_ids=json.dumps(["d1", "d2", "d3"]),
            depth=2,
        )
        db_session.add(node)
        db_session.flush()

        result = db_session.query(ClusterNode).filter(ClusterNode.id == node_id).first()
        d = result.to_dict()

        assert d["id"] == node_id
        assert d["parent_id"] == parent_id
        assert d["medoid_doc_id"] == "doc-42"
        assert d["doc_count"] == 5
        assert d["keywords"] == ["keyword1", "keyword2"]
        assert d["is_leaf"] is True
        assert d["doc_ids"] == ["d1", "d2", "d3"]
        assert d["depth"] == 2
        assert d["created_at"] is not None

    def test_default_values(self, db_session):
        """Test that default values are set correctly."""
        node = ClusterNode(id=str(uuid.uuid4()))
        db_session.add(node)
        db_session.flush()

        result = db_session.query(ClusterNode).first()
        assert result.doc_count == 0
        assert result.is_leaf is True
        assert result.depth == 0
        assert result.parent_id is None
        assert result.medoid_doc_id is None
        assert result.doc_ids is None

    def test_query_root_nodes(self, db_session):
        """Test querying for root nodes (parent_id is None)."""
        root1 = ClusterNode(id=str(uuid.uuid4()), parent_id=None, depth=0)
        root2 = ClusterNode(id=str(uuid.uuid4()), parent_id=None, depth=0)
        child = ClusterNode(id=str(uuid.uuid4()), parent_id=root1.id, depth=1)
        db_session.add_all([root1, root2, child])
        db_session.flush()

        roots = db_session.query(ClusterNode).filter(
            ClusterNode.parent_id == None  # noqa: E711
        ).all()
        assert len(roots) == 2

    def test_query_by_depth(self, db_session):
        """Test querying nodes by depth level."""
        n0 = ClusterNode(id=str(uuid.uuid4()), depth=0)
        n1a = ClusterNode(id=str(uuid.uuid4()), depth=1)
        n1b = ClusterNode(id=str(uuid.uuid4()), depth=1)
        n2 = ClusterNode(id=str(uuid.uuid4()), depth=2)
        db_session.add_all([n0, n1a, n1b, n2])
        db_session.flush()

        depth1_nodes = db_session.query(ClusterNode).filter(
            ClusterNode.depth == 1
        ).all()
        assert len(depth1_nodes) == 2


class TestDocument:
    """Tests for the Document model."""

    def test_create_document(self, db_session):
        """Test creating a basic document."""
        doc = Document(
            id="doc-001",
            text="This is a sample document about machine learning.",
            source="hash:abc123",
        )
        db_session.add(doc)
        db_session.flush()

        result = db_session.query(Document).first()
        assert result.id == "doc-001"
        assert result.text == "This is a sample document about machine learning."
        assert result.source == "hash:abc123"

    def test_document_with_embedding(self, db_session):
        """Test storing document with embedding as JSON string."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        doc = Document(
            id="doc-002",
            text="Document with embedding",
            source=None,
            embedding=json.dumps(embedding),
        )
        db_session.add(doc)
        db_session.flush()

        result = db_session.query(Document).filter(Document.id == "doc-002").first()
        stored_embedding = json.loads(result.embedding)
        assert len(stored_embedding) == 5
        assert abs(stored_embedding[0] - 0.1) < 1e-6

    def test_document_to_dict(self, db_session):
        """Test Document.to_dict() method."""
        doc = Document(
            id="doc-003",
            text="Short text.",
            source="upload",
        )
        db_session.add(doc)
        db_session.flush()

        result = db_session.query(Document).filter(Document.id == "doc-003").first()
        d = result.to_dict()

        assert d["id"] == "doc-003"
        assert d["text"] == "Short text."
        assert d["source"] == "upload"
        assert d["created_at"] is not None

    def test_to_dict_truncates_long_text(self, db_session):
        """Test that to_dict truncates text longer than 200 characters."""
        long_text = "A" * 300
        doc = Document(
            id="doc-004",
            text=long_text,
        )
        db_session.add(doc)
        db_session.flush()

        result = db_session.query(Document).filter(Document.id == "doc-004").first()
        d = result.to_dict()

        assert len(d["text"]) == 203  # 200 + "..."
        assert d["text"].endswith("...")

    def test_duplicate_id_raises_error(self, db_session):
        """Test that inserting a document with duplicate ID raises an error."""
        doc1 = Document(id="same-id", text="First document")
        db_session.add(doc1)
        db_session.flush()

        doc2 = Document(id="same-id", text="Second document")
        db_session.add(doc2)

        with pytest.raises(Exception):
            db_session.flush()

    def test_query_document_by_source(self, db_session):
        """Test querying documents by source (content hash)."""
        hash_val = "hash:deadbeef"
        doc1 = Document(id="d1", text="Text 1", source=hash_val)
        doc2 = Document(id="d2", text="Text 2", source="hash:other")
        db_session.add_all([doc1, doc2])
        db_session.flush()

        results = db_session.query(Document).filter(
            Document.source == hash_val
        ).all()
        assert len(results) == 1
        assert results[0].id == "d1"


class TestIngestionJob:
    """Tests for the IngestionJob model."""

    def test_create_job(self, db_session):
        """Test creating an ingestion job with defaults."""
        job = IngestionJob(
            id=str(uuid.uuid4()),
            total_docs=10,
        )
        db_session.add(job)
        db_session.flush()

        result = db_session.query(IngestionJob).first()
        assert result.status == "pending"
        assert result.total_docs == 10
        assert result.processed_docs == 0
        assert result.error_message is None

    def test_update_job_status(self, db_session):
        """Test updating job status through the pipeline."""
        job_id = str(uuid.uuid4())
        job = IngestionJob(id=job_id, status="pending", total_docs=5)
        db_session.add(job)
        db_session.flush()

        # Simulate pipeline stages
        job.status = "processing"
        db_session.flush()

        job.processed_docs = 3
        db_session.flush()

        job.status = "completed"
        job.processed_docs = 5
        db_session.flush()

        result = db_session.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        assert result.status == "completed"
        assert result.processed_docs == 5

    def test_job_failure(self, db_session):
        """Test recording a job failure."""
        job = IngestionJob(
            id=str(uuid.uuid4()),
            status="failed",
            total_docs=10,
            error_message="CUDA out of memory",
        )
        db_session.add(job)
        db_session.flush()

        result = db_session.query(IngestionJob).first()
        assert result.status == "failed"
        assert "CUDA" in result.error_message

    def test_to_dict(self, db_session):
        """Test IngestionJob.to_dict() serialization."""
        job_id = str(uuid.uuid4())
        job = IngestionJob(
            id=job_id,
            status="completed",
            total_docs=15,
            processed_docs=15,
        )
        db_session.add(job)
        db_session.flush()

        result = db_session.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        d = result.to_dict()

        assert d["id"] == job_id
        assert d["status"] == "completed"
        assert d["total_docs"] == 15
        assert d["processed_docs"] == 15
        assert d["error_message"] is None
        assert d["created_at"] is not None

    def test_query_by_status(self, db_session):
        """Test querying jobs by status."""
        j1 = IngestionJob(id=str(uuid.uuid4()), status="pending", total_docs=1)
        j2 = IngestionJob(id=str(uuid.uuid4()), status="completed", total_docs=2)
        j3 = IngestionJob(id=str(uuid.uuid4()), status="completed", total_docs=3)
        db_session.add_all([j1, j2, j3])
        db_session.flush()

        completed = db_session.query(IngestionJob).filter(
            IngestionJob.status == "completed"
        ).all()
        assert len(completed) == 2
