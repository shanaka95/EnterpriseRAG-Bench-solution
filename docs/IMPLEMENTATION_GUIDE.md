# Hierarchical Soft-Clustering RAG Pipeline

## Complete Implementation & Deployment Guide

> **Status: FULLY OPERATIONAL** | Last Updated: 2026-05-31
> **Public URL**: http://92.43.29.102:16492
> **API Docs**: http://92.43.29.102:16492/docs

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Database Schema](#database-schema)
5. [ML Pipeline Details](#ml-pipeline-details)
6. [API Reference](#api-reference)
7. [Deployment on vast.ai](#deployment-on-vastai)
8. [Configuration Reference](#configuration-reference)
9. [Testing](#testing)
10. [Known Issues & Solutions](#known-issues--solutions)
11. [Performance Metrics](#performance-metrics)
12. [RAG Evaluation Pipeline](#rag-evaluation-pipeline)
13. [Reimplementation Checklist](#reimplementation-checklist)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│         (React Flow Tree Visualization + Query Panel)    │
│            http://92.43.29.102:16492/                    │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                         │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐  │
│  │ /ingest  │ │  /query   │ │  /tree   │ │  /stream  │  │
│  │  POST    │ │  POST     │ │  GET     │ │  SSE      │  │
│  └────┬─────┘ └─────┬─────┘ └──────────┘ └─────┬─────┘  │
│       │             │                           │        │
│  ┌────▼─────┐ ┌────▼──────────────────┐ ┌──────▼──────┐  │
│  │ Document │ │ Cross-Encoder Query   │ │ Redis       │  │
│  │ Dedup    │ │ Multi-Path + Top-K    │ │ Pub/Sub     │  │
│  │ (SHA256) │ │ Reranking             │ │             │  │
│  └────┬─────┘ └──────────────────────┘ └─────────────┘  │
└───────┼──────────────────────────────────────────────────┘
        │ Celery Task Queue
┌───────▼──────────────────────────────────────────────────┐
│               Celery Worker (GPU Queue)                   │
│  ┌──────────────────────────────────────────────────────┐ │
│  │         Recursive Clustering Engine                  │ │
│  │  1. Bi-Encoder Embedding (BAAI/bge-m3, FP16)       │ │
│  │  2. UMAP Dimensionality Reduction (cuML GPU)        │ │
│  │  3. HDBSCAN Soft-Assignment Clustering (cuML GPU)   │ │
│  │  4. c-TF-IDF Keyword Extraction                     │ │
│  │  5. Cross-Encoder Borderline Refinement             │ │
│  │  6. Recursive Sub-tree Building (max depth 8)       │ │
│  └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
        │                           │
┌───────▼───────┐          ┌────────▼────────┐
│  PostgreSQL   │          │     Redis       │
│  (Tree + Docs)│          │  (Queue + Cache) │
└───────────────┘          └─────────────────┘
```

---

## Technology Stack

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| Backend | FastAPI | 0.115.6 | Python 3.11+ |
| Database | PostgreSQL | 16 | SQLAlchemy 2.0 ORM |
| Task Queue | Celery | 5.4.0 | Redis backend |
| Bi-Encoder | BAAI/bge-m3 | - | 1024-dim embeddings, FP16 |
| Cross-Encoder | ms-marco-MiniLM-L-6-v2 | - | Query routing + reranking |
| UMAP | RAPIDS cuML | 26.4 | GPU-accelerated (CPU fallback) |
| HDBSCAN | RAPIDS cuML | 26.4 | GPU-accelerated (CPU fallback) |
| c-TF-IDF | scikit-learn | 1.6.1 | Custom implementation |
| Frontend | React + TypeScript | 18.3 | React Flow for graph viz |
| Infrastructure | Supervisor | - | Process management |
| GPU | 2x NVIDIA Tesla T4 | 15GB each | vast.ai instance |

---

## Configuration Reference

### Critical Settings (Must Get Right)

| Variable | Default | For 371 docs | For 500K docs | Notes |
|----------|---------|-------------|---------------|-------|
| MIN_DOCS_FOR_SPLIT | 30 | 30 | 100-200 | Controls tree depth. Too small = excessive depth. Too large = shallow tree |
| MAX_TREE_DEPTH | 8 | 8 | 8-10 | Hard cap prevents infinite recursion |
| SOFT_ASSIGNMENT_THRESHOLD | 0.5 | 0.5 | 0.5 | Higher = less multi-assignment = smaller tree |
| CROSS_ENCODER_THRESHOLD | 1.0 | 1.0 | 0.5-1.0 | Controls query path count. Higher = fewer paths = fewer docs |
| EMBEDDING_BATCH_SIZE | 32 | 32 | 64 | Per 15GB VRAM: ~32 for bge-m3 FP16 |
| HDBSCAN_MIN_CLUSTER_SIZE | 5 | 5 | 10-20 | Minimum docs per cluster |

### All Settings

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | postgresql://rag:ragpass@localhost:5432/ragdb | PostgreSQL connection |
| REDIS_URL | redis://localhost:6379/0 | Redis for Celery + pub/sub |
| CELERY_BROKER_URL | redis://localhost:6379/0 | Celery broker |
| CELERY_RESULT_BACKEND | redis://localhost:6379/1 | Celery result backend |
| BI_ENCODER_MODEL | BAAI/bge-m3 | Sentence transformer model |
| CROSS_ENCODER_MODEL | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder model |
| EMBEDDING_BATCH_SIZE | 32 | GPU batch size (32 for 15GB VRAM) |
| UMAP_N_NEIGHBORS | 15 | UMAP neighbor count |
| UMAP_N_COMPONENTS | 32 | UMAP output dimensions |
| HDBSCAN_MIN_CLUSTER_SIZE | 5 | Min cluster size |
| HDBSCAN_MIN_SAMPLES | 3 | Min samples for HDBSCAN |
| SOFT_ASSIGNMENT_THRESHOLD | 0.5 | Min probability for multi-membership |
| BORDERLINE_MARGIN | 0.05 | Margin for Cross-Encoder refinement |
| CROSS_ENCODER_THRESHOLD | 1.0 | Score threshold for query routing |
| MIN_DOCS_FOR_SPLIT | 30 | Min docs to attempt splitting |
| MAX_TREE_DEPTH | 8 | Maximum tree depth |
| CTFIDF_TOP_K | 10 | Keywords per cluster |
| USE_GPU | true | Enable GPU acceleration |
| GPU_DEVICE | cuda:0 | CUDA device |

---

## API Reference

### POST /api/v1/ingest
Ingest documents with SHA-256 content-hash deduplication.

### POST /api/v1/query
Multi-path Cross-Encoder query with top-k reranking.
- **top_k** (default: 10): Max documents to return after Cross-Encoder reranking
- **threshold**: Cross-Encoder score threshold for path traversal

### POST /api/v1/rebuild
Rebuild the entire cluster tree from all stored documents.

### GET /api/v1/tree
Full tree structure (nodes + edges).

### GET /api/v1/stream
SSE real-time tree building updates.

### GET /api/v1/health
Health check with GPU status.

### GET /api/v1/stats
Tree statistics.

### POST /api/v1/dedup
Scan and remove duplicate documents.

### GET /api/v1/documents
List stored documents.

---

## Deployment on vast.ai

### Instance Details
- **Host**: 92.43.29.102
- **SSH Port**: 27846
- **Public URL**: http://92.43.29.102:16492
- **GPU**: 2x NVIDIA Tesla T4 (15GB each, 30GB total)
- **RAM**: 377 GB
- **OS**: Ubuntu 24.04 LTS

### Step-by-Step Deployment

```bash
# 1. Install system dependencies
apt-get update && apt-get install -y postgresql postgresql-contrib redis-server

# 2. Start services
pg_ctlcluster 16 main start
redis-server --daemonize yes

# 3. Create database
su - postgres -c "psql -c \"CREATE USER rag WITH PASSWORD 'ragpass';\""
su - postgres -c "psql -c \"CREATE DATABASE ragdb OWNER rag;\""
su - postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE ragdb TO rag;\""

# 4. Create conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda create -y -n rag python=3.11
conda activate rag

# 5. Install Python dependencies (CRITICAL: order matters!)
pip install fastapi==0.115.6 uvicorn[standard]==0.34.0 pydantic==2.10.4 pydantic-settings==2.7.1
pip install sqlalchemy==2.0.36 psycopg2-binary==2.9.10 alembic==1.14.1
pip install celery[redis]==5.4.0 redis==5.2.1
pip install scikit-learn==1.6.1 scipy hdbscan umap-learn
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124  # MUST be ≥2.6!
pip install sentence-transformers==3.3.1
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
pip install pytest sse-starlette==2.2.1 python-multipart httpx requests

# 6. Copy project files
mkdir -p /app && cd /app
# Upload tarball and extract

# 7. Create .env (see Configuration Reference)

# 8. Create DB tables
cd /app/backend
python3 -c "from app.core.database import Base, engine; from app.models.schemas import *; Base.metadata.create_all(bind=engine)"

# 9. Build frontend
source /opt/nvm/nvm.sh
cd /app/frontend && npm install && npm run build

# 10. Stop Jupyter on port 8080
kill $(lsof -t -i:8080)

# 11. Configure supervisor (see below)

# 12. Start services
supervisorctl reread && supervisorctl update

# 13. Ingest documents
python3 /app/ingest_local.py /app/finance-and-legal http://localhost:8080 50 0

# 14. Rebuild tree from all docs
curl -X POST http://localhost:8080/api/v1/rebuild
```

### Supervisor Config (/etc/supervisor/conf.d/rag-app.conf)

```ini
[program:rag-api]
command=/venv/rag/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
directory=/app/backend
environment=PATH="/venv/rag/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",FRONTEND_BUILD_DIR="/app/frontend/build"
autostart=true
autorestart=true
stdout_logfile=/var/log/rag-api.log
stderr_logfile=/var/log/rag-api-error.log

[program:rag-celery-worker]
command=/venv/rag/bin/celery -A worker.celery_app worker --loglevel=info --queues=gpu --concurrency=1
directory=/app/backend
environment=PATH="/venv/rag/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",FRONTEND_BUILD_DIR="/app/frontend/build"
autostart=true
autorestart=true
stdout_logfile=/var/log/rag-celery.log
stderr_logfile=/var/log/rag-celery-error.log
```

---

## Known Issues & Solutions

### 1. PyTorch CVE-2025-32434
**Problem**: PyTorch < 2.6 blocks torch.load. Transformers enforces this.
**Solution**: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`

### 2. cuML / cupy / pytest Import Error
**Problem**: cupy.testing imports pytest but it's not installed.
**Solution**: `pip install pytest`

### 3. CUDA OOM on 12GB VRAM
**Problem**: bge-m3 uses ~2.3GB, batch encoding of 64 docs overflows.
**Solution**: EMBEDDING_BATCH_SIZE=32, FP16 precision, text truncation to 2000 chars, auto-recovery.

### 4. Celery 5.x Queue Flag
**Problem**: Celery 5 uses `--queues` not `--queue`.
**Solution**: `celery -A worker.celery_app worker --queues=gpu`

### 5. Cross-Encoder Score Range
**Problem**: MiniLM outputs raw logits (-10 to +5), NOT probabilities.
**Solution**: CROSS_ENCODER_THRESHOLD=1.0 (not 0.6).

### 6. Excessive Tree Depth (70+ levels)
**Problem**: MIN_DOCS_FOR_SPLIT=10 + low soft_threshold=0.2 = massive fanout.
**Solution**: MIN_DOCS_FOR_SPLIT=30, MAX_TREE_DEPTH=8, SOFT_ASSIGNMENT_THRESHOLD=0.5, conservative multi-assignment (secondary must be ≥80% of primary probability).

### 7. Too Many Docs Retrieved (100+)
**Problem**: Low Cross-Encoder threshold + deep tree = too many leaf paths.
**Solution**: Cross-Encoder reranking in query endpoint with top_k=10.

### 8. Multiple Root Nodes from Batch Ingestion
**Problem**: Each batch creates its own root node via separate Celery task.
**Solution**: Use /api/v1/rebuild endpoint to build single tree from all docs at once.

### 9. MiniMax-M2.7 API Response Format
**Problem**: Returns content array with "thinking" and "text" type blocks.
**Solution**: Extract only blocks with type="text", not type="thinking".

### 10. NVIDIA nvjitlink Version Conflict
**Problem**: cuML requires nvjitlink ≥12.9, PyTorch 2.6 ships 12.4.
**Impact**: Warnings only. Both work correctly at runtime.

---

## RAG Evaluation Pipeline

### Pipeline Architecture

1. **Generate 100 Q&A pairs** from random documents using MiniMax-M2.7 LLM
2. **Retrieve** documents via Cross-Encoder multi-path traversal + top-10 reranking
3. **Generate grounded answers** from top-10 docs using MiniMax-M2.7
4. **Evaluate** with retrieval recall and LLM-as-judge metrics

### Evaluation Scripts

- `scripts/generate_qa_pairs.py` - Full pipeline: QA generation → retrieval → answer generation → evaluation
- `scripts/run_llm_eval.py` - LLM-as-judge evaluation on existing results

### LLM Configuration

- **API URL**: https://api.minimax.io/anthropic/v1/messages
- **Model**: MiniMax-M2.7
- **Response format**: Content array with "thinking" and "text" blocks

---

## Reimplementation Checklist

- [ ] Install system packages: PostgreSQL 16, Redis, Supervisor
- [ ] Create PostgreSQL database: user `rag`, password `ragpass`, database `ragdb`
- [ ] Create conda environment: Python 3.11
- [ ] Install PyTorch 2.6.0+ with CUDA support (**CRITICAL: ≥ 2.6**)
- [ ] Install sentence-transformers + download models
- [ ] Install RAPIDS cuML for GPU acceleration
- [ ] Install pytest (required by cupy)
- [ ] Copy backend code to /app/backend/
- [ ] Create database tables via SQLAlchemy
- [ ] Build React frontend (`npm install && npm run build`)
- [ ] Configure Supervisor for rag-api and rag-celery-worker
- [ ] Stop Jupyter on port 8080
- [ ] Set .env with correct settings for your dataset size
- [ ] Start services via supervisorctl
- [ ] Ingest documents using ingest_local.py
- [ ] Rebuild tree via /api/v1/rebuild (NOT individual batch ingestion)
- [ ] Verify health endpoint, tree stats, query endpoint
- [ ] Test frontend loads and displays tree
- [ ] Run dedup to check for duplicates
- [ ] Test SSE real-time updates
- [ ] Test query endpoint with various queries

### Critical Gotchas

1. **PyTorch version must be ≥ 2.6** or transformers refuses to load models
2. **EMBEDDING_BATCH_SIZE must match VRAM** (32 for 15GB, 64 for 30GB+)
3. **CROSS_ENCODER_THRESHOLD must be ~1.0** (raw logits, not probabilities)
4. **Celery 5 uses `--queues`** not `--queue`
5. **pytest must be installed** for cupy to import correctly
6. **Kill Jupyter first** before binding to port 8080
7. **Use /api/v1/rebuild** not individual batch ingestion for tree building
8. **MIN_DOCS_FOR_SPLIT** must be appropriate for dataset size (30 for 371 docs, 100-200 for 500K)
9. **MAX_TREE_DEPTH=8** prevents runaway recursion
10. **SOFT_ASSIGNMENT_THRESHOLD=0.5** with 80% margin prevents tree fanout
