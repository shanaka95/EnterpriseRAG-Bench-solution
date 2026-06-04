"""
SQLAlchemy models for the cluster tree and document metadata.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, ForeignKey, Text, ARRAY, Float,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY as PG_ARRAY
from sqlalchemy.orm import relationship
from app.core.database import Base


class ClusterNode(Base):
    """Represents a node in the hierarchical cluster tree."""
    __tablename__ = "cluster_nodes"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id = Column(PG_UUID(as_uuid=True), ForeignKey("cluster_nodes.id"), nullable=True)
    medoid_doc_id = Column(String(255), nullable=True)
    doc_count = Column(Integer, nullable=False, default=0)
    keywords = Column(PG_ARRAY(String), nullable=False, default=list)
    is_leaf = Column(Boolean, nullable=False, default=True)
    doc_ids = Column(PG_ARRAY(String), nullable=True)  # Only populated if is_leaf=True
    depth = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    parent = relationship("ClusterNode", remote_side=[id], backref="children")

    def to_dict(self):
        return {
            "id": str(self.id),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "medoid_doc_id": self.medoid_doc_id,
            "doc_count": self.doc_count,
            "keywords": self.keywords or [],
            "is_leaf": self.is_leaf,
            "doc_ids": self.doc_ids or [],
            "depth": self.depth,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Document(Base):
    """Stores document text and metadata."""
    __tablename__ = "documents"

    id = Column(String(255), primary_key=True)
    text = Column(Text, nullable=False)
    source = Column(String(512), nullable=True)
    embedding = Column(PG_ARRAY(Float), nullable=True)  # Stored as float array for fallback
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class IngestionJob(Base):
    """Tracks ingestion job status."""
    __tablename__ = "ingestion_jobs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed
    total_docs = Column(Integer, nullable=False, default=0)
    processed_docs = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": str(self.id),
            "status": self.status,
            "total_docs": self.total_docs,
            "processed_docs": self.processed_docs,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
