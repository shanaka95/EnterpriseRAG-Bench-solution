"""Add performance indexes for 512K document scale.

Revision ID: 20260531_205758
Revises:
Create Date: 2026-05-31 20:57:58
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260531_205758"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Documents: dedup lookup by source (content hash)
    op.create_index("idx_documents_source", "documents", ["source"])

    # Documents: partial index for FAISS build (only docs with embeddings)
    op.execute(
        "CREATE INDEX idx_documents_embedding_not_null ON documents(id) WHERE embedding IS NOT NULL"
    )

    # Cluster nodes: leaf queries (Phase 2 retrieval)
    op.create_index("idx_cluster_nodes_is_leaf", "cluster_nodes", ["is_leaf"])

    # Cluster nodes: parent traversal (tree path reconstruction)
    op.create_index("idx_cluster_nodes_parent_id", "cluster_nodes", ["parent_id"])

    # Cluster nodes: depth ordering (tree display)
    op.create_index("idx_cluster_nodes_depth", "cluster_nodes", ["depth"])


def downgrade() -> None:
    op.drop_index("idx_documents_source", table_name="documents")
    op.drop_index("idx_documents_embedding_not_null", table_name="documents")
    op.drop_index("idx_cluster_nodes_is_leaf", table_name="cluster_nodes")
    op.drop_index("idx_cluster_nodes_parent_id", table_name="cluster_nodes")
    op.drop_index("idx_cluster_nodes_depth", table_name="cluster_nodes")
