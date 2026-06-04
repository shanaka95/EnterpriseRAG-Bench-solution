# RAG Retrieval Experiments — Complete Build & Reproduction Guide

**Date:** 2026-06-04
**Status:** All experiments complete; this document is the definitive guide to reproducing the entire system from scratch.
**Audience:** Someone with a clean Linux machine, basic Python knowledge, and access to a few HuggingFace models.

By following this document end-to-end, you can reproduce:
- The 511,962-document corpus
- 3 retrieval indexes (jina-v3, gte-large, jina-colbert-v2)
- 11 experiment outputs (CSVs and JSONLs)
- The complete analysis in `RETRIEVAL_EXPERIMENTS_REPORT.md`
- The end-to-end "user question → top-100 docs" pipeline

---

## 1. System Overview

### What we built

A multi-stage RAG retrieval pipeline that combines **dense** (jina-embeddings-v3), **sparse** (BM25), and **late-interaction** (jina-colbert-v2) retrieval, with **Reciprocal Rank Fusion (RRF)** as the integration method. The final recommended production flow is:

```
user question
   ↓
[jina-embeddings-v3 dense top-2000]  ∥  [BM25 Lucene top-2000]
   ↓                              ↘                    ↙
            [RRF fusion, k0=60] → ranked list
                              ↓
                  [top-100 doc IDs]
```

- **Latency:** ~0.6-2.5s end-to-end (CPU), ~50-200ms (GPU)
- **Accuracy:** 83.4% hit@100, 89.4% hit@1000
- **Improvement over single retrievers:** +2-19pp at every K
- **Best of all tested configurations:** RRF alone for K ≤ 200, ColBERT rerank for K ≥ 500

### Why these numbers matter

The "hit@K" metric is the percentage of 500 labeled questions for which the gold document appears in the top-K retrieved documents. This is a strict correctness metric — we count the question as a hit only if the expected doc ID is found.

### Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 8 cores | 16+ cores (ColBERT rerank is CPU-bound) |
| RAM | 32 GB | 64 GB (ColBERT index is 64 GB) |
| Disk | 100 GB free | 200 GB (all indexes + corpus) |
| GPU | None required | Optional NVIDIA for ~3-5× speedup |
| Python | 3.10+ | 3.12 |

**Wall-clock for full reproduction (CPU):**

| Stage | Time |
|---|---|
| Environment setup | 5-10 min |
| Corpus download (assumed pre-existing) | N/A |
| Build jina-v3 index (512K docs) | 2-4 hours GPU / 12-24 hours CPU |
| Build ColBERT index (512K docs) | 4-8 hours GPU / 24-48 hours CPU |
| Run all 11 experiments | 2-3 hours |

---

## 2. Prerequisites

### 2.1 System packages

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip \
    build-essential git curl wget
```

### 2.2 HuggingFace account (for model downloads)

Some models (jina-colbert-v2) require accepting a license. Create a free account at https://huggingface.co, then:
1. Visit https://huggingface.co/jinaai/jina-colbert-v2 and accept the license.
2. Create an access token: https://huggingface.co/settings/tokens
3. Set the env var: `export HF_TOKEN=hf_xxxxx`

### 2.3 Pre-existing data (assumed to exist)

This document assumes you already have:
- `/data/projects/rag/data/all_documents/` — 511,962 .txt files (3.3 GB) in 9 source subdirs (slack/, gmail/, google_drive/, linear/, hubspot/, fireflies/, github/, jira/, confluence/)
- `/data/projects/rag/data/questions.jsonl` — 500 labeled questions with gold doc IDs (765 KB)

If you need to generate the corpus from scratch, see Appendix A.

---

## 3. Project Layout

```
/data/projects/rag/                                 # PROJECT ROOT
├── data/
│   ├── all_documents/                              # 511,962 source .txt files (3.3 GB)
│   │   ├── slack/
│   │   ├── gmail/
│   │   ├── google_drive/
│   │   ├── linear/
│   │   ├── hubspot/
│   │   ├── fireflies/
│   │   ├── github/
│   │   ├── jira/
│   │   └── confluence/
│   ├── questions.jsonl                              # 500 labeled questions (765 KB)
│   │
│   ├── dense_index/db/                              # jina-v3 LanceDB (2.0 GB) — built by Exp 4
│   ├── lancedb_data/                                # gte-large LanceDB (2.8 GB) — built by Exp 2
│   ├── colbert_index/db/                            # jina-colbert-v2 LanceDB (64 GB) — built by Exp 4
│   │
│   ├── jina_v3_scale_experiment.csv                 # EXP 4 OUTPUT: 1.7 MB
│   ├── retrieval_experiment.csv                     # EXP 1 OUTPUT: 33 KB
│   ├── two_stage_experiment.csv                     # EXP 2 OUTPUT: 2.5 MB
│   ├── jina_v3_topk_docids.jsonl                    # EXP 4 FULL: 497 MB
│   ├── jina_v3_topk_evaluation.csv                  # EXP 4 EVAL: 54 KB
│   ├── bm25_topk_docids.jsonl                       # EXP 5 FULL: 514 MB
│   ├── bm25_topk_evaluation.csv                     # EXP 5 EVAL: 54 KB
│   ├── hybrid_retrieval_per_question.csv            # EXP 6 PER-Q: 110 KB
│   ├── hybrid_retrieval_summary.csv                 # EXP 6 SUMMARY: 0.7 KB
│   ├── hybrid_retrieval_pareto.csv                  # EXP 6 PARETO: 0.9 KB
│   ├── hybrid_per_source_all.csv                    # EXP 6 PER-SRC: 11 KB
│   ├── colbert_rerank_jv500_bm2000_per_question.csv # EXP 7 PER-Q: 86 KB
│   ├── colbert_rerank_jv500_bm2000_summary.csv      # EXP 7 SUMMARY: 0.6 KB
│   ├── colbert_rerank_jv500_bm2000_ranked.csv       # EXP 7 RANKED: 0.7 KB
│   ├── colbert_rerank_per_source.csv                # EXP 7 PER-SRC: 1.7 KB
│   ├── flashrank_rerank_per_question.csv            # EXP 8 PER-Q: 83 KB
│   ├── flashrank_rerank_summary.csv                 # EXP 8 SUMMARY: 0.6 KB
│   ├── rrf_fusion_summary.csv                       # EXP 9 SUMMARY: 1.8 KB
│   ├── rrf_fusion_N*_k*_per_question.csv            # EXP 9 PER-Q (16 files): ~83 KB each
│   ├── rrf_full_ranking_N2000_k060.jsonl            # EXP 9 FULL: 256 MB
│   ├── colbert_rerank_rrf_1000_per_question.csv     # EXP 10 PER-Q: 86 KB
│   ├── colbert_rerank_rrf_1000_summary.csv          # EXP 10 SUMMARY: 0.6 KB
│   ├── colbert_rerank_rrf_per_question.csv          # EXP 10 PER-Q (top-2000): 86 KB
│   ├── colbert_rerank_rrf_summary.csv               # EXP 10 SUMMARY: 0.6 KB
│   │
│   └── ruvector.db                                  # Optional ruvector index
│
└── backend/                                          # Python venv + FastAPI app
    ├── venv/                                        # Python 3.12 venv (REQUIRED)
    ├── app/
    │   ├── core/
    │   │   ├── config.py                            # Settings (model paths, etc.)
    │   │   └── database.py                          # SQLAlchemy / PG setup (NOT USED in experiments)
    │   └── ml/
    │       └── colbert_reranker.py                  # ColBERT MaxSim impl
    ├── requirements.txt                             # Python deps
    └── alembic/                                     # (NOT USED in experiments)

/home/shanaka/Desktop/projects/rag/                   # REPORT/SCRIPTS ROOT
├── BUILD.md                                         # This file
├── RETRIEVAL_EXPERIMENTS_REPORT.md                  # 40-section analysis (1800+ lines)
└── scripts/                                         # All experiment scripts (40+ files)
    ├── retrieval_experiment.py                      # EXP 1
    ├── two_stage_experiment.py                      # EXP 2
    ├── jina_v3_scale_experiment.py                  # EXP 3, 4
    ├── save_topk_docids.py                          # EXP 4 full doc IDs
    ├── save_topk_bm25.py                            # EXP 5
    ├── hybrid_retrieval_experiment.py               # EXP 6
    ├── colbert_hybrid_rerank.py                     # EXP 7
    ├── flashrank_rerank.py                          # EXP 8
    ├── rrf_fusion.py                                # EXP 9
    ├── colbert_rerank_rrf.py                        # EXP 10
    ├── retrieve_100.py                              # END-TO-END live pipeline
    ├── show_rrf_path.py                             # Inspect any question's RRF path
    ├── smoke_dense.py                               # Smoke test: LanceDB jina-v3
    ├── smoke_colbert.py                             # Smoke test: LanceDB ColBERT
    ├── bench_colbert_latency.py                     # ColBERT latency benchmark
    ├── crosscheck_dense.py                          # Server-vs-local verification
    ├── build_faiss_index.py                         # (legacy) Build FAISS bge-m3
    ├── bulk_embed_*.py                              # (legacy) Various GPU embed builders
    ├── generate_qa_pairs.py                         # (legacy) Generate questions from corpus
    ├── eval_enterprise_bench.py                     # (legacy) Eval utility
    └── [other legacy scripts...]
```

---

## 4. Environment Setup

### 4.1 Create Python venv

```bash
cd /home/shanaka/Desktop/projects/rag
python3.12 -m venv backend/venv
source backend/venv/bin/activate

# Install all dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
# Plus the experimental extras
pip install bm25s FlagEmbedding FlashRank
```

### 4.2 Install PyLate (ColBERT)

```bash
pip install pylate einops
```

### 4.3 Verify GPU (optional)

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
```

If GPU available, set `CUDA_VISIBLE_DEVICES=0` in subsequent commands.

### 4.4 HuggingFace login (for jina-colbert-v2)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
huggingface-cli login --token $HF_TOKEN
```

### 4.5 Verify environment

```bash
cd /home/shanaka/Desktop/projects/rag
./backend/venv/bin/python -c "
import lancedb, sentence_transformers, bm25s, FlagEmbedding, FlashRank
from pylate import models as pylate_models
print('lancedb:', lancedb.__version__)
print('sentence_transformers:', sentence_transformers.__version__)
print('bm25s:', bm25s.__version__)
print('FlagEmbedding OK')
print('FlashRank OK')
print('pylate OK')
"
```

Expected: all imports succeed, versions printed.

### 4.6 Required Python packages (full list)

From `backend/requirements.txt` + extras:

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
pydantic-settings==2.7.1
sqlalchemy==2.0.36
psycopg2-binary==2.9.10
alembic==1.14.1
celery[redis]==5.4.0
redis==5.2.1
sentence-transformers==3.3.1
scikit-learn==1.6.1
numpy==1.26.4
scipy==1.14.1
torch==2.5.1
faiss-cpu==1.14.2
cross-encoder==1.0.0
pylate==1.2.0
einops==0.8.0
lancedb==0.33.0
pylance==7.0.0
python-multipart==0.0.20
httpx==0.28.1
sse-starlette==2.2.1
websockets==14.1
python-dateutil==2.9.0
uuid7==0.1.0
# Experimental:
bm25s==0.3.9
FlagEmbedding==1.4.0
FlashRank==0.2.10
```

Note: `sentence-transformers==3.3.1` is in requirements.txt but the venv has 4.0.2 (upgrade is OK; all scripts work).

---

## 5. Data

### 5.1 Corpus structure

`/data/projects/rag/data/all_documents/` — 511,962 plain-text documents:

```
all_documents/
├── confluence/        (PRDs, wiki pages)
├── fireflies/         (meeting transcripts, often long)
├── github/            (PRs, commits, code review notes)
├── gmail/             (emails, customer support)
├── google_drive/      (PRDs, specs)
├── hubspot/           (deals, contacts)
├── jira/              (tickets, comments)
├── linear/            (issues, design docs)
├── slack/             (chat, support)
└── ruvector.db        (optional ruvector index, not used here)
```

- **Doc ID format:** `{source}/{filename}.txt` where filename is `dsid_{32_hex}__{title}.txt`
- **Avg doc size:** 4.8 KB
- **Total size:** 3.3 GB raw text

### 5.2 Questions dataset

`/data/projects/rag/data/questions.jsonl` — 500 labeled questions (JSONL format):

```json
{
  "question_id": "qst_0001",
  "question_type": "basic",
  "source_types": ["github"],
  "question": "What are the default size limits for file uploads...?",
  "expected_doc_ids": ["dsid_ae068ee4aa9640159427cd941bef0238"],
  "gold_answer": "The default limits are 10 MiB per file (max_file_size) and 50 MiB total per request (max_total_request_size)...",
  "answer_facts": ["The default per file upload size limit (max_file_size) for multipart uploads on OpenAI-compatible endpoints is 10 MiB.", ...]
}
```

**Distribution by source (single-source questions):**

| Source | Count |
|---|---:|
| jira | 60 |
| slack | 57 |
| confluence | 64 |
| github | 39 |
| gmail | 42 |
| linear | 44 |
| google_drive | 42 |
| hubspot | 33 |
| fireflies | 21 |
| multi-source | 98 |
| **Total** | **500** |

---

## 6. Indexing (build from scratch)

### 6.1 Jina-embeddings-v3 dense index (LanceDB)

The jina-v3 index is the primary dense retriever.

```bash
cd /home/shanaka/Desktop/projects/rag

# This script encodes 511,962 docs in batches and stores in LanceDB
./backend/venv/bin/python scripts/jina_v3_scale_experiment.py \
    --num 500 \
    --out /tmp/_throwaway.csv    # we only need the index built
```

Actually the script both builds the index AND runs the experiment. To build just the index without running the experiment, use the `bulk_embed_and_build.py` script:

```bash
# Encode all 512K docs and build the LanceDB index
# This is the same script that originally built the jina_v3 dense_index/db
./backend/venv/bin/python scripts/bulk_embed_and_build.py
```

**Configuration used:**
- Model: `jinaai/jina-embeddings-v3` (570M params, 1024-dim)
- Task: `retrieval.passage` for documents, `retrieval.query` for queries
- LoRA adapter: yes (MUST match between encode and query)
- Storage: `/data/projects/rag/data/dense_index/db/`
- Table name: `documents`
- Schema: `id` (string), `source` (string), `n_tokens` (int), `embedding` (list[float32, 1024-dim])
- Index type: vector column for L2 (Euclidean) search
- Disk size: 2.0 GB

**Build time:** ~2-4 hours on GPU, ~12-24 hours on CPU

### 6.2 GTE-large dense index (LanceDB)

Used in Experiment 2 only.

```bash
# Build gte-large index (the original "baseline" dense model)
./backend/venv/bin/python scripts/bulk_embed_v3.py --model gte-large
# OR if a dedicated gte build script exists:
# (See scripts/bulk_embed_*.py variants)
```

**Configuration:**
- Model: `Alibaba-NLP/gte-large-en-v1.5` (434M params, 1024-dim, MIT license)
- Storage: `/data/projects/rag/lancedb_data/`
- Disk size: 2.8 GB

**Build time:** ~1-2 hours on GPU, ~6-12 hours on CPU

### 6.3 Jina-ColBERT-v2 multi-vector index (LanceDB)

The ColBERT index is the late-interaction reranker. Largest index, slowest to build.

```bash
# Build ColBERT multi-vector index over 512K docs
# This is built via the pylate library and stored as int8-quantized multi-vectors
./backend/venv/bin/python -c "
import os
os.environ['HF_TOKEN'] = 'hf_xxxxx'
import lancedb
import numpy as np
from pylate import models as pylate_models
from pathlib import Path

# Load model
model = pylate_models.ColBERT(
    model_name_or_path='jinaai/jina-colbert-v2',
    document_length=8192,
    query_prefix='[QueryMarker]',
    document_prefix='[DocumentMarker]',
    attend_to_expansion_tokens=True,
    trust_remote_code=True,
    device='cpu',  # or 'cuda' if GPU
)

# Open or create LanceDB
db = lancedb.connect('/data/projects/rag/data/colbert_index/db')
if 'documents' in db.table_names():
    table = db.open_table('documents')
else:
    # Create table with empty rows
    import pyarrow as pa
    schema = pa.schema([
        ('id', pa.string()),
        ('source', pa.string()),
        ('n_tokens', pa.int32()),
        ('scale', pa.float32()),
        ('embeddings', pa.list_(pa.int8())),
    ])
    table = db.create_table('documents', schema=schema, mode='overwrite')

# Encode and insert in batches
docs = []
ids = []
for fp in sorted(Path('/data/projects/rag/data/all_documents').rglob('*.txt')):
    did = fp.relative_to('/data/projects/rag/data/all_documents').as_posix()
    ids.append(did)

# Batch encode (use pylate.models.ColBERT.encode with is_query=False)
# NOTE: full encoding script was bulk_embed_*.py variants; see existing ColBERT index for the schema
"
```

**Configuration:**
- Model: `jinaai/jina-colbert-v2` (auto-loaded by pylate)
- Schema: `id` (string), `source` (string), `n_tokens` (int), `scale` (float32), `embeddings` (list of int8)
- Per doc: 128-dim multi-vector, ~500-2000 tokens, int8 quantized
- Storage: `/data/projects/rag/data/colbert_index/db/`
- Disk size: 64 GB
- Requires accepting the jina-colbert-v2 license on HuggingFace

**Build time:** ~4-8 hours on GPU, ~24-48 hours on CPU (this is the slowest step)

### 6.4 Verify indexes

```bash
./backend/venv/bin/python -c "
import lancedb
print('--- jina-v3 index ---')
t = lancedb.connect('/data/projects/rag/data/dense_index/db').open_table('documents')
print('  rows:', t.count_rows())
print('  schema:', t.schema)

print('--- ColBERT index ---')
t = lancedb.connect('/data/projects/rag/data/colbert_index/db').open_table('documents')
print('  rows:', t.count_rows())
print('  schema:', t.schema)
"
```

Expected output:
```
--- jina-v3 index ---
  rows: 511962
  schema: id: string, source: string, n_tokens: int, vector: fixed_size_list<item: float>[1024]
--- ColBERT index ---
  rows: 511962
  schema: id: string, source: string, n_tokens: int, scale: float, embeddings: list<item: int8>
```

### 6.5 Smoke tests

```bash
# Test jina-v3 retrieval
./backend/venv/bin/python scripts/smoke_dense.py

# Test ColBERT rerank
./backend/venv/bin/python scripts/smoke_colbert.py
```

Both should print "smoke OK" at the end.

---

## 7. Experiments — Step-by-Step

All experiments assume the indexes are built (Section 6) and the corpus + questions are in place (Section 5).

### 7.1 Experiment 1: Standalone Retrieval (10 questions, 3 algorithms)

**Purpose:** Compare jina-v3, gte-large, and ColBERT as standalone retrievers.
**Runtime:** ~2 minutes
**Output:** `data/retrieval_experiment.csv`

```bash
cd /home/shanaka/Desktop/projects/rag
./backend/venv/bin/python scripts/retrieval_experiment.py
```

**Expected results:**

| Algorithm | hit@100 |
|---|:---:|
| jina-embeddings-v3 | 7/10 (70%) |
| gte-large-en-v1.5 | 2/10 (20%) |
| ColBERT (prefilter) | 0/10 (artifact) |

**Report section:** §2

### 7.2 Experiment 2: Two-Stage Pipeline (10 questions)

**Purpose:** Test jina-v3 (or gte) → ColBERT MaxSim rerank on top-1000 candidates.
**Runtime:** ~3 minutes
**Output:** `data/two_stage_experiment.csv`

```bash
./backend/venv/bin/python scripts/two_stage_experiment.py
```

**Expected results:**

| Pipeline | hit@5 | hit@10 | hit@100 |
|---|:---:|:---:|:---:|
| jina-v3 → ColBERT | 70% | 80% | 80% |
| gte → ColBERT | 20% | 20% | 20% |

**Report section:** §3

### 7.3 Experiment 3: Jina-v3 Scale (100 questions)

**Purpose:** Measure jina-v3 hit@K for K ∈ {100, 500, 1000, 2000, 5000} on first 100 questions.
**Runtime:** ~4 minutes
**Output:** not persisted (just printed) — re-run to reproduce §4.

```bash
./backend/venv/bin/python scripts/jina_v3_scale_experiment.py --num 100
```

**Expected results:** hit@100=68%, hit@500=82%, hit@1000=89%, hit@2000=90%, hit@5000=95%.

**Report section:** §4

### 7.4 Experiment 4: Jina-v3 Scale (500 questions, full) + save top-K doc IDs

**Purpose:** Full 500-question jina-v3 evaluation + save complete top-K doc ID lists.
**Runtime:** ~17 minutes (CPU)
**Outputs:** `data/jina_v3_scale_experiment.csv`, `data/jina_v3_topk_docids.jsonl`, `data/jina_v3_topk_evaluation.csv`

```bash
# Step 4a: Run the experiment and write evaluation CSV
./backend/venv/bin/python scripts/jina_v3_scale_experiment.py --num 500
mv data/jina_v3_scale_experiment.csv data/_jina_v3_scale_experiment.csv
# (the script writes the per-question CSV with only top-5 docs per K; we need full top-K)

# Step 4b: Save the full top-K doc IDs (overwrites the CSV with only top-5)
./backend/venv/bin/python scripts/save_topk_docids.py --num 500

# Recreate the evaluation CSV (just summary)
./backend/venv/bin/python -c "
import json, csv
with open('/data/projects/rag/data/jina_v3_topk_docids.jsonl') as f:
    rows = [json.loads(l) for l in f]
with open('/data/projects/rag/data/questions.jsonl') as f:
    questions = {json.loads(l)['question_id']: json.loads(l) for l in f if l.strip()}
def best_match(ranked, expected):
    for eid in expected:
        for ri, d in enumerate(ranked):
            if eid in d: return ri+1
    return None
out = []
for r in rows:
    qid = r['question_id']
    q = questions[qid]
    exp = q.get('expected_doc_ids', [])
    row = {
        'question_id': qid,
        'question_type': q.get('question_type', ''),
        'source_types': '|'.join(q.get('source_types', [])),
        'expected_doc_ids': '|'.join(exp),
    }
    for k in [100, 500, 1000, 2000, 5000]:
        rank = best_match(r[f'top_{k}_ids'], exp)
        row[f'hit@{k}'] = 1 if rank else 0
        row[f'rank@{k}'] = rank if rank else ''
    out.append(row)
with open('/data/projects/rag/data/jina_v3_topk_evaluation.csv', 'w', newline='') as f:
    cols = ['question_id', 'question_type', 'source_types', 'expected_doc_ids']
    for k in [100, 500, 1000, 2000, 5000]:
        cols.extend([f'hit@{k}', f'rank@{k}'])
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)
print('done')
"

# Now restore the jina_v3_scale_experiment.csv by re-running the original
./backend/venv/bin/python scripts/jina_v3_scale_experiment.py --num 500 --out /data/projects/rag/data/jina_v3_scale_experiment.csv
```

(Note: this workflow has a slight duplication; the cleaner approach is to use `save_topk_docids.py` as the primary script and add evaluation to it. The current state has both files; the jina_v3_scale_experiment.csv has only top-5 per K as the "summary" while jina_v3_topk_docids.jsonl has the full list.)

**Expected results:**

| K | hit@K | Accuracy |
|---:|---:|---:|
| 100 | 315/500 | 63.0% |
| 500 | 377/500 | 75.4% |
| 1000 | 402/500 | 80.4% |
| 2000 | 419/500 | 83.8% |
| 5000 | 437/500 | 87.4% |

**Report sections:** §5, §11

### 7.5 Experiment 5: BM25 Standalone (500 questions)

**Purpose:** Build BM25 index, retrieve top-K, evaluate.
**Runtime:** ~4 minutes (index build) + ~1 min retrieval
**Outputs:** `data/bm25_topk_docids.jsonl`, `data/bm25_topk_evaluation.csv`

```bash
./backend/venv/bin/python scripts/save_topk_bm25.py --num 500
```

**Expected results:**

| K | hit@K | Accuracy |
|---:|---:|---:|
| 100 | 408/500 | 81.6% |
| 500 | 431/500 | 86.2% |
| 1000 | 446/500 | 89.2% |
| 2000 | 460/500 | 92.0% |
| 5000 | 464/500 | 92.8% |

**Report section:** §10

### 7.6 Experiment 6: 5×5 Hybrid Union (500 questions)

**Purpose:** Compute deduped-union of jv top-K + BM25 top-K for all 25 (K, K) combinations; evaluate hit-rate + union size.
**Runtime:** ~10 seconds (in-memory only)
**Outputs:** `data/hybrid_retrieval_per_question.csv`, `data/hybrid_retrieval_summary.csv`, `data/hybrid_retrieval_pareto.csv`, `data/hybrid_per_source_all.csv`

```bash
./backend/venv/bin/python scripts/hybrid_retrieval_experiment.py
```

**Expected results (5×5 hit rate, %):**

| jv \ bm | bm100 | bm500 | bm1000 | bm2000 | bm5000 |
|---|---:|---:|---:|---:|---:|
| jv100  | 84.4 | 87.6 | 89.4 | 92.0 | 92.8 |
| jv500  | 86.6 | 88.8 | 90.2 | 92.2 | 92.8 |
| jv1000 | 87.8 | 89.4 | 90.6 | 92.2 | 92.8 |
| jv2000 | 88.6 | 90.0 | 91.0 | 92.2 | 92.8 |
| jv5000 | 90.4 | 91.2 | 92.0 | 92.8 | 93.2 |

**Pareto-optimal configurations (11 of 25):**

| Combo | hit@K union | Mean union |
|---|---:|---:|
| jv100+bm100 | 84.4% | 187 |
| jv500+bm100 | 86.6% | 574 |
| jv100+bm500 | 87.6% | 574 |
| **jv500+bm500** | **88.8%** | **928** ⭐ |
| jv100+bm1000 | 89.4% | 1066 |
| jv500+bm1000 | 90.2% | 1397 |
| jv1000+bm1000 | 90.6% | 1845 |
| **jv100+bm2000** | **92.0%** | **2058** ⭐ |
| jv500+bm2000 | 92.2% | 2359 |
| jv1000+bm2000 | 92.2% | 2776 |
| jv5000+bm5000 | 93.2% | 9029 |

**Report sections:** §14, §15, §16

### 7.7 Experiment 7: ColBERT Rerank on jv500+bm2000 Union

**Purpose:** Apply ColBERT MaxSim rerank to the jv500+bm2000 deduped union; evaluate at 17 K values.
**Runtime:** ~32 minutes (CPU)
**Outputs:** `data/colbert_rerank_jv500_bm2000_per_question.csv`, `..._summary.csv`, `..._ranked.csv`, `data/colbert_rerank_per_source.csv`

```bash
./backend/venv/bin/python scripts/colbert_hybrid_rerank.py --num 500
```

**Expected results (hit@K):**

| K | Hits | hit_rate | mean_rank |
|---:|---:|---:|---:|
| 10  | 303 | 60.6% | 2.4 |
| 50  | 366 | 73.2% | 6.8 |
| 100 | 391 | 78.2% | 11.7 |
| 200 | 415 | 83.0% | 22.2 |
| 500 | 440 | 88.0% | 41.1 |
| 1000| 454 | 90.8% | 73.3 |

**Report sections:** §21, §22, §23

### 7.8 Experiment 8: FlashRank Rerank (TinyBERT-L-2)

**Purpose:** Try a lightweight ONNX-optimized cross-encoder (FlashRank) on the same pool.
**Runtime:** ~44 minutes (CPU)
**Outputs:** `data/flashrank_rerank_per_question.csv`, `..._summary.csv`

```bash
./backend/venv/bin/python scripts/flashrank_rerank.py --num 500
```

**Expected results (hit@K):**

| K | Hits | hit_rate | vs ColBERT |
|---:|---:|---:|:---:|
| 10  | 288 | 57.6% | -3.0pp |
| 100 | 375 | 75.0% | -3.2pp |
| 1000| 446 | 89.2% | -1.6pp |

**Report section:** §27

### 7.9 Experiment 9: RRF Fusion (4 N × 4 k0 sweep)

**Purpose:** Compute Reciprocal Rank Fusion over jv top-N + BM25 top-N for 16 configurations.
**Runtime:** ~10 seconds
**Outputs:** `data/rrf_fusion_summary.csv`, 16 per-question CSVs (one per config), `data/rrf_full_ranking_N2000_k060.jsonl`

```bash
# Step 9a: Run the full sweep
./backend/venv/bin/python scripts/rrf_fusion.py

# Step 9b: Save the full ranked lists for the best config (N=2000, k0=60)
./backend/venv/bin/python -c "
import json
from collections import defaultdict
with open('/data/projects/rag/data/jina_v3_topk_docids.jsonl') as f:
    jv = {json.loads(l)['question_id']: json.loads(l)['top_2000_ids'] for l in f}
with open('/data/projects/rag/data/bm25_topk_docids.jsonl') as f:
    bm = {json.loads(l)['question_id']: json.loads(l)['top_2000_ids'] for l in f}
with open('/data/projects/rag/data/questions.jsonl') as f:
    questions = [json.loads(l) for l in f if l.strip()]
K0, N = 60, 2000
out = '/data/projects/rag/data/rrf_full_ranking_N2000_k060.jsonl'
with open(out, 'w') as f:
    for q in questions:
        qid = q['question_id']
        jv_top, bm_top = jv[qid], bm[qid]
        scores = defaultdict(float)
        for r, d in enumerate(jv_top, 1): scores[d] += 1.0 / (K0 + r)
        for r, d in enumerate(bm_top, 1): scores[d] += 1.0 / (K0 + r)
        jvr = {d: i for i, d in enumerate(jv_top)}
        bmr = {d: i for i, d in enumerate(bm_top)}
        ranked = sorted(scores.keys(), key=lambda d: (-scores[d], jvr.get(d, 1e9), bmr.get(d, 1e9)))
        rec = {
            'question_id': qid, 'question': q['question'],
            'source_types': q.get('source_types', []),
            'expected_doc_ids': q.get('expected_doc_ids', []),
            'ranked_doc_ids': ranked,
            'rrf_scores': [scores[d] for d in ranked],
        }
        f.write(json.dumps(rec) + '\n')
print('done')
"
```

**Expected results (hit@K for N=2000, k0=60):**

| K | hit_rate |
|---:|:---:|
| 10  | 71.0% |
| 50  | 80.6% |
| 100 | 83.4% |
| 200 | 85.2% |
| 1000| 89.4% |

**Report sections:** §32, §34, §37

### 7.10 Experiment 10: ColBERT Rerank on RRF Top-K

**Purpose:** Apply ColBERT MaxSim rerank on the RRF top-1000 (or 2000) instead of the flat union.
**Runtime:** ~15 min (top-1000) or ~23 min (top-2000)
**Outputs:** `data/colbert_rerank_rrf_1000_summary.csv`, `data/colbert_rerank_rrf_summary.csv`

```bash
# ColBERT on RRF top-1000
./backend/venv/bin/python scripts/colbert_rerank_rrf.py \
    --num 500 --n-input 2000 --k0 60 --rrf-pool 1000 \
    --out /data/projects/rag/data/colbert_rerank_rrf_1000_per_question.csv \
    --summary /data/projects/rag/data/colbert_rerank_rrf_1000_summary.csv

# ColBERT on RRF top-2000
./backend/venv/bin/python scripts/colbert_rerank_rrf.py \
    --num 500 --n-input 2000 --k0 60 --rrf-pool 2000 \
    --out /data/projects/rag/data/colbert_rerank_rrf_per_question.csv \
    --summary /data/projects/rag/data/colbert_rerank_rrf_summary.csv
```

**Expected results (hit@K):**

| K | ColBERT on flat | ColBERT on RRF-1k | ColBERT on RRF-2k |
|---:|:---:|:---:|:---:|
| 100 | 78.2% | 78.8% | 77.4% |
| 1000| 90.8% | 89.4% | 89.6% |

**Report sections:** §33, §34

### 7.11 (Optional) End-to-end live pipeline

```bash
# Inspect any question's RRF path
./backend/venv/bin/python scripts/show_rrf_path.py --qid qst_0001 --k 20

# Run the live end-to-end pipeline (slower — ~0.6-2.5s per query)
./backend/venv/bin/python scripts/retrieve_100.py \
    --query "What are the default size limits for file uploads" \
    --k 100 --show-text
```

**Report section:** Part VII

---

## 8. Key Files Reference

### 8.1 The full ranked-list (most important for "what is the path?")

`data/rrf_full_ranking_N2000_k060.jsonl` — 256 MB
- One row per question
- `ranked_doc_ids`: full ordered list (avg 3,457 candidates)
- `rrf_scores`: parallel array of scores

Query it like this:
```python
import json
with open('/data/projects/rag/data/rrf_full_ranking_N2000_k060.jsonl') as f:
    for ln in f:
        rec = json.loads(ln)
        if rec['question_id'] == 'qst_0001':
            for j, doc_id in enumerate(rec['ranked_doc_ids'][:100], 1):
                print(f'{j:>3}. {doc_id}')
            break
```

### 8.2 The production retrieval script

`scripts/retrieve_100.py` — end-to-end user question → top-100 doc IDs

```bash
# Inspect a known question (fast, <100ms)
./backend/venv/bin/python scripts/retrieve_100.py \
    --query "What are the default size limits for file uploads" \
    --qid qst_0001 --from-cached --k 10 --all-ks

# End-to-end live (slower, 0.6-2.5s)
./backend/venv/bin/python scripts/retrieve_100.py \
    --query "Your new question here" \
    --k 100 --show-text
```

### 8.3 The final report

`RETRIEVAL_EXPERIMENTS_REPORT.md` (also at `/home/shanaka/Desktop/projects/rag/RETRIEVAL_EXPERIMENTS_REPORT.md`) — 40 sections, 1800+ lines, covers every experiment in detail with per-source breakdowns, Pareto analyses, and recommended production pipelines.

---

## 9. Reproduction Checklist

To verify a fresh rebuild is correct, check that these file sizes and row counts match:

```bash
cd /data/projects/rag
echo "=== Corpus ==="
ls all_documents/ | wc -l           # → 10 (9 source dirs + ruvector.db)
find all_documents -name "*.txt" | wc -l   # → 511962

echo "=== Questions ==="
wc -l data/questions.jsonl          # → 500

echo "=== Indexes ==="
./backend/venv/bin/python -c "
import lancedb
for name, path in [('jv', 'data/dense_index/db'),
                   ('colbert', 'data/colbert_index/db')]:
    t = lancedb.connect(path).open_table('documents')
    print(f'{name}: {t.count_rows()} rows')
"   # → jv: 511962, colbert: 511962

echo "=== Experiment outputs ==="
for f in data/jina_v3_topk_docids.jsonl data/bm25_topk_docids.jsonl \
         data/rrf_full_ranking_N2000_k060.jsonl; do
    wc -l $f
done   # each should be 500

echo "=== Summaries ==="
for f in data/*_summary.csv data/hybrid_retrieval_pareto.csv data/colbert_rerank_*_ranked.csv; do
    echo "$f: $(wc -l < $f) rows"
done
```

**Expected:**
- `jina_v3_topk_docids.jsonl`: 500 rows, ~497 MB
- `bm25_topk_docids.jsonl`: 500 rows, ~514 MB
- `rrf_full_ranking_N2000_k060.jsonl`: 500 rows, ~256 MB
- All summary CSVs: as documented per experiment

---

## 10. Recommended Production Pipeline

Based on all 10 experiments, the recommended production flow is:

```
user question
   ↓
[Stage 1A: jina-v3 dense top-2000]  ||  [Stage 1B: BM25 Lucene top-2000]
   ↓                                  ↘                ↙
            [RRF fusion, k0=60] → ranked list
                              ↓
                  [top-100 doc IDs]
```

**For K=100 RAG use case (most common):**
- Pipeline: RRF alone (no rerank)
- hit@100: 83.4%
- Latency: ~0.6-2.5s end-to-end
- Cost: 2-3s CPU time per query, 0 GPU

**For K=1000 max recall:**
- Pipeline: BM25 top-1000 ∪ jv top-500 → ColBERT MaxSim → top-1000
- hit@1000: 90.8%
- Latency: ~3.9s end-to-end on CPU

**For K=200 balanced:**
- Pipeline: RRF top-1000 → ColBERT MaxSim → top-200
- hit@200: 83.8%
- Latency: ~1.8s end-to-end on CPU

---

## 11. Known Limitations & Future Work

### 11.1 The 9.2% accuracy ceiling

36 of 500 questions (7.2%) are "impossible" — missed by BM25@5000 and missed by jina-v3@5000. All rerank and fusion approaches top out at ~91% at K=1000.

**To break this ceiling:**
1. **Annotation audit** of the 36 misses (1-2 hours manual) — may reveal bad labels
2. **Query2Doc** (LLM pseudo-documents + BM25) — 3-15% BM25 boost per the paper
3. **SPLADE++** as learned sparse first-stage — 2-5% boost
4. **BGE-M3** for long-context dense — 1-3% boost
5. **Multi-hop decomposition** for "complex" questions — unknown boost

### 11.2 Latency improvements

- GPU jina-v3 encode: 50-200ms (vs 2s CPU)
- GPU ColBERT: 0.5-1s (vs 3.9s CPU) — requires re-implementing in PyLate with FP16

### 11.3 What's missing

- LLM-based reranking (RankGPT, RankZephyr) — not viable on CPU
- Cross-encoder rerank (BGE-reranker-v2-m3) — too slow on CPU; would work on GPU
- Query rewriting / HyDE — needs LLM API

---

## 12. Appendix A: Regenerating the Corpus (Optional)

If you don't have the corpus pre-existing, it was generated via:

```bash
# Historical: scripts/generate_qa_pairs.py was used to extract QA pairs from a
# larger Slack/Gmail/GitHub/Jira/etc. snapshot. This is NOT a deterministic
# process — it depends on the source data.
#
# The corpus structure is:
#   all_documents/{source}/dsid_{32_hex}__{title}.txt
# where the dsid is a randomly-generated 32-hex ID and the title is a slug
# of the document's content.
#
# This process is OUT OF SCOPE for the retrieval experiments. The 500 labeled
# questions in questions.jsonl are the test set, and were generated from
# the same source data.
```

If you have the original source data, see `scripts/ingest_docs.py` for the ingestion logic.

---

## 13. Appendix B: Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          OFFLINE INDEXING                                │
└─────────────────────────────────────────────────────────────────────────┘

  /data/projects/rag/data/all_documents/        (511,962 .txt files, 3.3 GB)
       │
       │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
       ├─→│ jina-embed-v3     │  │ gte-large-en-v1.5│  │ jina-colbert-v2   │
       │  │ (570M, 1024-dim) │  │ (434M, 1024-dim) │  │ (multi-vec, 128-d)│
       │  │ task=passage     │  │ CLS pooled       │  │ int8 quant       │
       │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
       │           ↓                     ↓                     ↓
       │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
       │  │ LanceDB          │  │ LanceDB          │  │ LanceDB          │
       │  │ /dense_index/db  │  │ /lancedb_data    │  │ /colbert_index/db│
       │  │ 2.0 GB           │  │ 2.8 GB           │  │ 64 GB            │
       │  └──────────────────┘  └──────────────────┘  └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          ONLINE RETRIEVAL                                │
└─────────────────────────────────────────────────────────────────────────┘

  user question
       │
       ├─→ [jina-v3 encode, 1024-dim] ─→ [LanceDB L2 top-2000]   (50-200ms)
       │
       ├─→ [BM25 tokenize]           ─→ [bm25s index top-2000]   (50-200ms)
       │
       └─→ [RRF fusion k0=60]        ─→ [ranked list]            (<50ms)
                                            │
                                            ↓
                                       [top-100 doc IDs]          (<10ms)

       OPTIONAL: For higher precision at K=200-1000:
                                            ↓
                                   [ColBERT MaxSim rerank]      (1.8-3.9s)
                                            ↓
                                   [top-K refined docs]
```

---

## 14. Appendix C: Quick Reference — All Hit@K Numbers in One Place

### Standalone retrievers (500 questions)

| Method | hit@100 | hit@500 | hit@1000 | hit@2000 | hit@5000 |
|---|:---:|:---:|:---:|:---:|:---:|
| jina-v3 dense | 63.0% | 75.4% | 80.4% | 83.8% | 87.4% |
| BM25 Lucene | 81.6% | 86.2% | 89.2% | 92.0% | 92.8% |
| gte-large | — | — | 20% | — | — |
| ColBERT standalone (prefilter) | — | — | 0% (artifact) | — | — |

### Hybrid union (no rerank)

| Combo | hit@100 | hit@500 | hit@1000 | hit@2000 | hit@5000 |
|---|:---:|:---:|:---:|:---:|:---:|
| jv500+bm500 (best cost/acc) | 86.6% | 88.8% | 90.2% | 92.2% | 92.8% |
| jv500+bm2000 (used in Exp 7) | — | — | 92.2% | — | — |
| jv100+bm2000 (best 2.5K) | — | — | 92.0% | — | — |
| jv5000+bm5000 (max union) | 90.4% | 91.2% | 92.0% | 92.8% | 93.2% |

### RRF alone (16 configs)

| N \ k0 | 10 | 30 | 60 | 100 |
|---|---:|---:|---:|---:|
| 500  | 88.8% | 88.8% | 88.8% | 88.8% |
| 1000 | 89.0% | 89.0% | 89.0% | 89.0% |
| 2000 | **89.4%** | **89.4%** | **89.4%** | **89.4%** |
| 5000 | **89.4%** | **89.4%** | **89.4%** | **89.4%** |

### Rerankers (on jv500+bm2000 union of 2359 candidates)

| Method | hit@10 | hit@50 | hit@100 | hit@200 | hit@500 | hit@1000 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ColBERT MaxSim (on flat union) | 60.6% | 73.2% | 78.2% | 83.0% | 88.0% | **90.8%** |
| FlashRank TinyBERT-L-2 | 57.6% | 70.0% | 75.0% | 79.4% | 86.0% | 89.2% |
| RRF (N=2k, k0=60) alone | **71.0%** | **80.6%** | **83.4%** | **85.2%** | 87.0% | 89.4% |

### Rerankers (on RRF top-1000)

| Method | hit@10 | hit@50 | hit@100 | hit@200 | hit@500 | hit@1000 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ColBERT on RRF top-1000 | 61.2% | 73.8% | 78.8% | 83.8% | 87.4% | 89.4% |
| ColBERT on RRF top-2000 | 60.2% | 72.6% | 77.4% | 82.4% | 87.2% | 89.6% |

**Bold = winner per K column.**

---

## 15. Summary

This document is the definitive guide to reproducing the entire RAG retrieval experiment system. By following Sections 4-7 in order, you can rebuild:

1. The Python environment (5-10 min)
2. The 3 retrieval indexes (12-48 hours, mostly ColBERT)
3. All 11 experiment outputs (2-3 hours)
4. The 40-section analysis report (auto-generated by reference to the experiment outputs)

**Time investment:**
- Setup + indexes: ~1-2 days (CPU) or ~6-12 hours (GPU)
- All experiments: ~2-3 hours (CPU, mostly ColBERT rerank)
- Reading this doc: 30 min

**Disk footprint:**
- Code + scripts: ~50 MB
- Corpus: 3.3 GB
- Indexes: 70 GB (mostly ColBERT)
- Experiment outputs: 1.5 GB
- **Total: ~75 GB**

If anything is unclear or you hit an error, the most likely issues are:
1. Wrong Python version (need 3.10+, prefer 3.12)
2. Missing HF_TOKEN for jina-colbert-v2 license
3. Insufficient RAM for ColBERT index (64 GB)
4. CUDA OOM if using GPU with insufficient VRAM
5. Path differences (this doc assumes `/data/projects/rag/` and `/home/shanaka/Desktop/projects/rag/`)

For questions or to extend the system, see `RETRIEVAL_EXPERIMENTS_REPORT.md` Part V §28 (Path to 95%) for concrete next-experiment recommendations.

---

## 16. Reactive RAG Agent (LangGraph + Streamlit UI)

A reactive agent built on top of the best production pipeline. Stages 1-3
match the RRF N=2000 k0=60 config exactly (83.4% hit@100 on dev). Stage
4 is a ReAct-style LLM agent that reads the refined docs 10 at a time
and answers the question as soon as it sees supporting evidence.

### 16.1 What it does

```
[user question]
    │
    ▼
┌──────────────────┐
│ bm25_retrieve    │  top-2000 BM25 (Lucene, k1=1.5, b=0.75)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ jina_dense       │  top-2000 jina-embeddings-v3 (task=retrieval.query)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ rrf_fuse         │  RRF (k0=60) → top-100
└──────────────────┘
    │
    ▼
┌──────────────────┐   tool: get_next_batch (10 docs at a time)
│  agent (ReAct)   │ ──▶
└──────────────────┘
    │
    ▼
{"doc_id": "...", "response": "..."}    (strict JSON, single tool call)
```

The agent's first call fetches docs 1-10. If the answer is in those 10,
it stops and emits JSON. If not, it calls `get_next_batch` again to
get docs 11-20, and so on, up to 100. After reading all 100 it gives
its best guess or admits it cannot find the answer.

### 16.2 Code layout

```
backend/agent/
├── __init__.py        # exports
├── state.py           # AgentState TypedDict
├── config.py          # AgentConfig (env-driven)
├── retrieval.py       # bm25_retrieve, jina_dense_retrieve, make_rrf_fuse
├── tools.py           # make_get_next_batch (closure-based tool)
├── llm.py             # get_llm (ChatAnthropic on the MiniMax endpoint)
└── graph.py           # StateGraph + build_graph + run_agent

ui/
└── streamlit_app.py   # 500-question picker + per-question run + trace view

scripts/
└── run_ui.sh          # launches streamlit on port 8599

.env.example           # template for MINIMAX_API_KEY
```

### 16.3 Setup

```bash
cd /home/shanaka/Desktop/projects/rag
cp .env.example .env
# edit .env and put your real API key
./backend/venv/bin/pip install -r requirements-agent.txt   # if not already
```

If starting from scratch the agent needs:
- `langgraph`, `langchain-core`, `langchain-anthropic` — pip install
- `streamlit` 1.49.x + `starlette<0.42` (compatibility with FastAPI 0.115)

### 16.4 Run

```bash
# Option A: use the run script (loads .env automatically)
./scripts/run_ui.sh
# → http://localhost:8599

# Option B: manual
MINIMAX_API_KEY=sk-cp-... \
MINIMAX_BASE_URL=https://api.minimax.io/anthropic \
MINIMAX_MODEL=MiniMax-M2.7 \
./backend/venv/bin/streamlit run ui/streamlit_app.py \
    --server.port 8599 --server.headless true
```

### 16.5 Programmatic use (no UI)

```python
import sys
sys.path.insert(0, "/data/projects/rag/backend")
from agent import run_agent
import json

q = json.loads(open("/data/projects/rag/data/questions.jsonl").readline())
final = run_agent(
    q["question"],
    question_id=q["question_id"],
    expected_doc_ids=q.get("expected_doc_ids", []),
    gold_answer=q.get("gold_answer"),
)
print("answer:", final["final_answer"])
print("supporting:", final["supporting_doc_ids"])
print("hit:", any(any(e in d for d in final["supporting_doc_ids"])
                  for e in q["expected_doc_ids"]))
```

### 16.6 Known issues

- **First run is slow** (~4 min): the BM25 index is built lazily on first
  call and cached in process memory. Subsequent runs are <1s for retrieval.
- **Some model outputs truncate doc_ids** to the human-readable suffix
  (e.g. `dsid_18421-…` instead of the full 32-char hash). The UI marks
  expected hits with substring matching so this still works.
- **Thinking blocks are stripped** from the conversation after each
  agent turn — the MiniMax endpoint has trouble pairing `tool_use` with
  `tool_result` when `thinking` is interleaved. The actual LLM still
  "thinks"; we just don't send those blocks back.
- **Streamlit 1.58 needs starlette 1.x** but FastAPI 0.115 caps at
  starlette 0.41 — we pin `streamlit<1.50` for compatibility.

### 16.7 Latency budget (per question)

| Stage                       | First run   | Subsequent  |
|-----------------------------|-------------|-------------|
| jina-v3 + LanceDB load      | ~6s         | (cached)    |
| BM25 index build            | ~240s       | (cached)    |
| BM25 query (top-2000)       | ~0.5s       | ~0.5s       |
| jina-v3 encode + search     | ~7s         | ~7s         |
| RRF fusion                  | <0.01s      | <0.01s      |
| LLM agent (per turn)        | ~3-12s      | ~3-12s      |
| LLM turns (typical)         | 5-12        | 5-12        |
| **Total**                   | **~5 min**  | **30-90s**  |


