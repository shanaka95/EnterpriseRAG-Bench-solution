"""
Core configuration for the Hierarchical Soft-Clustering RAG Pipeline.
All settings are loaded from environment variables with sensible defaults.
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Application ──
    APP_NAME: str = "Hierarchical RAG Pipeline"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, alias="DEBUG")

    # ── Database ──
    DATABASE_URL: str = Field(
        default="postgresql://rag:ragpass@localhost:5432/ragdb",
        alias="DATABASE_URL",
    )

    # ── Redis / Celery ──
    REDIS_URL: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", alias="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND")

    # ── ML Models ──
    BI_ENCODER_MODEL: str = Field(default="BAAI/bge-m3", alias="BI_ENCODER_MODEL")
    CROSS_ENCODER_MODEL: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        alias="CROSS_ENCODER_MODEL",
    )
    EMBEDDING_DIM: int = Field(default=1024, alias="EMBEDDING_DIM")  # bge-m3 output dim
    EMBEDDING_BATCH_SIZE: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")

    # ── UMAP ──
    UMAP_N_NEIGHBORS: int = Field(default=15, alias="UMAP_N_NEIGHBORS")
    UMAP_MIN_DIST: float = Field(default=0.1, alias="UMAP_MIN_DIST")
    UMAP_N_COMPONENTS: int = Field(default=32, alias="UMAP_N_COMPONENTS")
    UMAP_METRIC: str = Field(default="cosine", alias="UMAP_METRIC")

    # ── HDBSCAN ──
    HDBSCAN_MIN_CLUSTER_SIZE: int = Field(default=5, alias="HDBSCAN_MIN_CLUSTER_SIZE")
    HDBSCAN_MIN_SAMPLES: int = Field(default=3, alias="HDBSCAN_MIN_SAMPLES")
    HDBSCAN_METRIC: str = Field(default="euclidean", alias="HDBSCAN_METRIC")

    # ── Soft Assignment ──
    SOFT_ASSIGNMENT_THRESHOLD: float = Field(default=0.5, alias="SOFT_ASSIGNMENT_THRESHOLD")
    BORDERLINE_MARGIN: float = Field(default=0.05, alias="BORDERLINE_MARGIN")

    # ── Cross-Encoder Query Routing ──
    CROSS_ENCODER_THRESHOLD: float = Field(default=1.0, alias="CROSS_ENCODER_THRESHOLD")

    # ── Recursive Clustering ──
    MIN_DOCS_FOR_SPLIT: int = Field(default=100, alias="MIN_DOCS_FOR_SPLIT")
    MAX_TREE_DEPTH: int = Field(default=8, alias="MAX_TREE_DEPTH")
    CTFIDF_TOP_K: int = Field(default=10, alias="CTFIDF_TOP_K")

    # ── GPU ──
    USE_GPU: bool = Field(default=True, alias="USE_GPU")
    GPU_DEVICE: str = Field(default="cuda:0", alias="GPU_DEVICE")

    # ── ColBERT Reranker (Jina-ColBERT-v2) ──
    # When enabled, replaces the cross-encoder in Phase 5 of /api/v1/query
    # with multi-vector late-interaction reranking against a local LanceDB
    # of pre-computed int8-quantized embeddings.
    COLBERT_RERANK_ENABLED: bool = Field(default=False, alias="COLBERT_RERANK_ENABLED")
    COLBERT_RERANK_POOL:    int  = Field(default=1000, alias="COLBERT_RERANK_POOL")
    COLBERT_INDEX_PATH:     str  = Field(
        default="/data/projects/rag/data/colbert_index/db", alias="COLBERT_INDEX_PATH",
    )
    COLBERT_MODEL_NAME:     str  = Field(
        default="jinaai/jina-colbert-v2", alias="COLBERT_MODEL_NAME",
    )
    COLBERT_QUERY_PREFIX:   str  = Field(default="[QueryMarker]", alias="COLBERT_QUERY_PREFIX")
    COLBERT_DOC_PREFIX:     str  = Field(default="[DocumentMarker]", alias="COLBERT_DOC_PREFIX")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
