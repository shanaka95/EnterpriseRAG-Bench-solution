"""
Pytest fixtures for the Hierarchical Soft-Clustering RAG Pipeline tests.
Uses in-memory SQLite and mocks heavy ML models.
"""
import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine, event, text, String, Text as SAText, Float, Boolean, Integer, DateTime, ForeignKey, TypeDecorator
from sqlalchemy.orm import sessionmaker, DeclarativeBase, relationship
from sqlalchemy import Column


# ---------------------------------------------------------------------------
# SQLite-compatible UUID type that accepts both str and uuid.UUID in queries
# ---------------------------------------------------------------------------

class SQLiteUUID(TypeDecorator):
    """A UUID type that stores values as strings in SQLite but accepts
    uuid.UUID objects in filter expressions, converting them to strings
    automatically so that queries like ``ClusterNode.id == uuid.UUID(...)``
    work correctly against the string-backed column."""
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return str(value)
        return value


# ---------------------------------------------------------------------------
# SQLite-compatible declarative base and models
# ---------------------------------------------------------------------------
# The production models use PG_UUID and PG_ARRAY which are PostgreSQL-only.
# We recreate equivalent models using portable SQLAlchemy types for SQLite tests.
# ---------------------------------------------------------------------------

class TestBase(DeclarativeBase):
    """SQLAlchemy declarative base for test models (SQLite compatible)."""
    pass


class ClusterNode(TestBase):
    """SQLite-compatible ClusterNode model."""
    __tablename__ = "cluster_nodes"

    id = Column(SQLiteUUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    parent_id = Column(SQLiteUUID(), ForeignKey("cluster_nodes.id"), nullable=True)
    medoid_doc_id = Column(String(255), nullable=True)
    doc_count = Column(Integer, nullable=False, default=0)
    keywords = Column(SAText, nullable=False, default="[]")  # JSON string
    is_leaf = Column(Boolean, nullable=False, default=True)
    doc_ids = Column(SAText, nullable=True)  # JSON string
    depth = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    parent = relationship("ClusterNode", remote_side=[id], backref="children")

    def to_dict(self):
        import json
        return {
            "id": str(self.id),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "medoid_doc_id": self.medoid_doc_id,
            "doc_count": self.doc_count,
            "keywords": json.loads(self.keywords) if self.keywords else [],
            "is_leaf": self.is_leaf,
            "doc_ids": json.loads(self.doc_ids) if self.doc_ids else [],
            "depth": self.depth,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Document(TestBase):
    """SQLite-compatible Document model."""
    __tablename__ = "documents"

    id = Column(String(255), primary_key=True)
    text = Column(SAText, nullable=False)
    source = Column(String(512), nullable=True)
    embedding = Column(SAText, nullable=True)  # JSON string of floats
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class IngestionJob(TestBase):
    """SQLite-compatible IngestionJob model."""
    __tablename__ = "ingestion_jobs"

    id = Column(SQLiteUUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(50), nullable=False, default="pending")
    total_docs = Column(Integer, nullable=False, default=0)
    processed_docs = Column(Integer, nullable=False, default=0)
    error_message = Column(SAText, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": str(self.id),
            "status": self.status,
            "total_docs": self.total_docs,
            "processed_docs": self.processed_docs,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Engine and Session Fixtures
# ---------------------------------------------------------------------------

SQLALCHEMY_DATABASE_URL = "sqlite:///file::memory:?cache=shared&uri=true"


@pytest.fixture(scope="session")
def engine():
    """Create a SQLAlchemy engine for in-memory SQLite (shared across session)."""
    eng = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    # Enable WAL mode for better concurrency with shared cache
    @event.listens_for(eng, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    TestBase.metadata.create_all(bind=eng)
    yield eng
    TestBase.metadata.drop_all(bind=eng)
    eng.dispose()


@pytest.fixture()
def db_session(engine):
    """Create a fresh database session for each test, rolled back afterwards."""
    connection = engine.connect()
    transaction = connection.begin()
    TestSession = sessionmaker(bind=connection)
    session = TestSession()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def db_session_for_api(engine):
    """
    Create a database session for use in FastAPI TestClient dependency overrides.
    Returns (session, override_generator) so the test can install the override.
    """
    connection = engine.connect()
    transaction = connection.begin()
    TestSession = sessionmaker(bind=connection)
    session = TestSession()

    def override_get_db():
        yield session

    yield session, override_get_db

    session.close()
    transaction.rollback()
    connection.close()


# ---------------------------------------------------------------------------
# Mock Embedding Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_bi_encoder():
    """Mock SentenceTransformer bi-encoder that returns random embeddings."""
    mock_model = MagicMock()
    embedding_dim = 128  # Small dim for tests
    mock_model.encode.return_value = np.random.rand(5, embedding_dim).astype(np.float32)
    mock_model.to.return_value = mock_model
    return mock_model


@pytest.fixture()
def mock_cross_encoder():
    """Mock CrossEncoder that returns random scores."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.rand(3).astype(np.float32)
    return mock_model


@pytest.fixture()
def mock_embedding_functions(mock_bi_encoder, mock_cross_encoder):
    """
    Patch the embedding module so encode_documents and cross_encoder_score
    use mocked models instead of loading real GPU models.
    """
    with patch("app.ml.embedding.get_bi_encoder", return_value=mock_bi_encoder), \
         patch("app.ml.embedding.get_cross_encoder", return_value=mock_cross_encoder):
        yield {
            "bi_encoder": mock_bi_encoder,
            "cross_encoder": mock_cross_encoder,
        }


# ---------------------------------------------------------------------------
# Synthetic Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_embeddings():
    """Generate synthetic embeddings for clustering tests.

    Creates 3 well-separated Gaussian blobs in 50-dimensional space,
    suitable for testing UMAP + HDBSCAN with small data.
    """
    rng = np.random.RandomState(42)
    n_per_cluster = 20
    dim = 50

    # Cluster 1: centered with positive offset
    offset1 = np.zeros(dim)
    offset1[:dim // 2] = 5.0
    c1 = rng.randn(n_per_cluster, dim) * 0.3 + offset1

    # Cluster 2: centered with negative offset
    offset2 = np.zeros(dim)
    offset2[:dim // 2] = -5.0
    c2 = rng.randn(n_per_cluster, dim) * 0.3 + offset2

    # Cluster 3: offset along first feature to separate from c1 and c2
    c3 = rng.randn(n_per_cluster, dim) * 0.3
    c3[:, 0] += 10.0

    embeddings = np.vstack([c1, c2, c3]).astype(np.float32)
    labels_true = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)
    return embeddings, labels_true


@pytest.fixture()
def sample_documents():
    """Generate sample document data for API and dedup tests."""
    return [
        {"id": "doc-1", "text": "Machine learning models process data efficiently.", "source": None},
        {"id": "doc-2", "text": "Neural networks learn patterns from training examples.", "source": None},
        {"id": "doc-3", "text": "Deep learning uses multiple layers of abstraction.", "source": None},
        {"id": "doc-4", "text": "Natural language processing handles text data.", "source": None},
        {"id": "doc-5", "text": "Computer vision interprets visual information.", "source": None},
    ]


@pytest.fixture()
def duplicate_documents():
    """Generate documents with duplicate content for dedup tests."""
    return [
        {"id": "doc-a", "text": "This is a unique document about cats.", "source": None},
        {"id": "doc-b", "text": "This is a unique document about dogs.", "source": None},
        {"id": "doc-c", "text": "This is a unique document about cats.", "source": None},  # duplicate of doc-a
        {"id": "doc-d", "text": "This is a unique document about birds.", "source": None},
        {"id": "doc-e", "text": "This is a unique document about dogs.", "source": None},  # duplicate of doc-b
    ]


@pytest.fixture()
def sample_cluster_texts():
    """Sample cluster texts for c-TF-IDF tests."""
    return {
        "cluster_0": "machine learning neural network deep learning model training data prediction accuracy gradient descent",
        "cluster_1": "database query optimization index SQL relational schema transaction performance",
        "cluster_2": "neural network training data machine learning prediction accuracy model",
    }


# ---------------------------------------------------------------------------
# FastAPI TestClient Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(db_session_for_api):
    """Create a FastAPI TestClient with database dependency override.

    Patches:
    - The production models (ClusterNode, Document, IngestionJob) in the
      routes module to use SQLite-compatible test models.
    - The Celery app to avoid Redis connections.
    - The database session via FastAPI dependency override.
    - Skips the lifespan (which tries to connect to PostgreSQL).
    """
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fastapi.middleware.cors import CORSMiddleware

    from app.core.database import get_db
    from app.api.routes import router as api_router

    session, override_get_db = db_session_for_api

    # Build a test-only FastAPI app that includes the API router but skips
    # the lifespan (which tries to connect to PostgreSQL) and the SSE/Redis
    # router (which requires a running Redis server).
    @asynccontextmanager
    async def _noop_lifespan(app: FastAPI):
        yield

    test_app = FastAPI(lifespan=_noop_lifespan)
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    test_app.include_router(api_router)

    test_app.dependency_overrides[get_db] = override_get_db

    # Patch the production model classes in the routes module with our
    # SQLite-compatible test models so that all DB operations use the
    # correct column types.
    with patch("app.api.routes.ClusterNode", ClusterNode), \
         patch("app.api.routes.Document", Document), \
         patch("app.api.routes.IngestionJob", IngestionJob):
        with TestClient(test_app) as c:
            yield c

    test_app.dependency_overrides.clear()
