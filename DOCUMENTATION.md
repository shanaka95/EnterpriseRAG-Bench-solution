# RAG System — Full Documentation

**Status:** Production retrieval indexes are built, deployed, and verified. Four distinct embedding layers exist over the same **511,962 documents**:

1. **FAISS bi-encoder** (`BAAI/bge-m3`) — live in the FastAPI query pipeline (Phase 1)
2. **ColBERT late-interaction reranker** (`jinaai/jina-colbert-v2`) — wired into `routes.py`, gated behind `COLBERT_RERANK_ENABLED` (Phase 5)
3. **Dense jina-v3 single-vector** (`jinaai/jina-embeddings-v3`) — built, verified, **not yet wired** (candidate Phase 1 replacement or RRF addition)
4. **Dense gte-large single-vector** (`Alibaba-NLP/gte-large-en-v1.5`) — built earlier, stored in a separate LanceDB with a rich metadata schema (path, size, mtime, sha256, text snippet, etc.) — usable for semantic search

---

## 1. Corpus

- **Source:** `/home/shanaka/Desktop/projects/rag/data/all_documents/` (3.3 GB, 511,962 `.txt` files)
- **Doc ID convention:** the **relative path** under `all_documents/` (e.g. `slack/eng-oncall/dsid_xxxx__20260321-kms-rotation.txt`). This is the primary key used across every index, the Postgres `documents` table, and the FastAPI surface — it is stable across every system component.
- **Manifest:** `data/{dense_index,colbert_index}/_build/manifest.txt` — 511,962 sorted relative paths.

### Corpus Composition (exact counts)

| Source | Documents | % of Corpus |
|---|---:|---:|
| `slack` | 285,605 | 55.8% |
| `gmail` | 121,390 | 23.7% |
| `linear` | 35,308 | 6.9% |
| `google_drive` | 25,108 | 4.9% |
| `hubspot` | 15,017 | 2.9% |
| `fireflies` | 10,173 | 2.0% |
| `github` | 8,052 | 1.6% |
| `jira` | 6,120 | 1.2% |
| `confluence` | 5,189 | 1.0% |
| **Total** | **511,962** | **100%** |

### File size distribution

- Median: **4,295 bytes**
- P99: **11,947 bytes**
- Max: **42,281 bytes**
- All fit under the 8,192-token context window.

---

## 2. Indexes (what we have)

### 2.1 FAISS bi-encoder (existing — first-stage retrieval)

| | |
|---|---|
| **Location** | `/app/backend/faiss_index` (inside container path; see `backend/app/ml/faiss_index.py:INDEX_DIR`) |
| **Build script** | `scripts/build_faiss_index.py` |
| **Model** | `BAAI/bge-m3` (`backend/.env:BI_ENCODER_MODEL`) |
| **Dim** | 1024 |
| **Metric** | cosine (vectors L2-normalized at encode time) |
| **Wrapper** | `backend/app/ml/faiss_index.py` — lazy singleton, returns `(doc_id, score)` |
| **Used at query time** | Phase 1 of `/api/v1/query` — top-1,000 bi-encoder candidates |
| **Embedder** | `backend/app/ml/embedding.py:encode_documents()` — BF16 on GPU, OOM-retry halves batch size |

### 2.2 ColBERT multi-vector reranker (deployed — Phase 5 rerank)

| | |
|---|---|
| **LanceDB** | `/data/projects/rag/data/colbert_index/db` |
| **Model** | `jinaai/jina-colbert-v2` (`backend/.env:COLBERT_MODEL_NAME`) |
| **Storage** | int8-quantized per-doc, 128-dim per token, ~150 KB/doc → **64 GB total** |
| **Schema** | `id (string PK) \| source (string) \| n_tokens (int32) \| scale (float32) \| embeddings (large_binary)` |
| **Build artifacts** | `data/colbert_index/_build/server_scripts/embed_colbert.py` + `data/colbert_index/_build/RUNBOOK.md` |
| **Wrapper** | `backend/app/ml/colbert_reranker.py` — lazy singleton model + LanceDB, MaxSim via `np.einsum("qd,nkd->nqk", ...)` in chunks of 100 |
| **Activated by** | `backend/.env:COLBERT_RERANK_ENABLED=true` (default **false**); falls back to cross-encoder on any error |
| **License** | cc-by-nc-4.0 (non-commercial use OK) |

### 2.3 Dense jina-v3 single-vector (built — for future first-stage replacement)

| | |
|---|---|
| **LanceDB** | `/data/projects/rag/data/dense_index/db` |
| **Model** | `jinaai/jina-embeddings-v3` (570 M params, 1024-dim, 8K ctx, Matryoshka-friendly) |
| **Storage** | float32, 1024-dim, 4 KB/doc → **2.0 GB total** |
| **Schema** | `id (string PK) \| source (string) \| n_tokens (int32) \| embedding (fixed_size_list<float32, 1024>)` |
| **Build artifacts** | `data/dense_index/_build/server_scripts/embed_dense_v2.py` |
| **Embedded with** | `task="retrieval.passage"` LoRA adapter, BF16 + flash-attn 2 on RTX 3060, 9.18 h total |
| **NOT yet wired into backend** | needs a query-time loader + a `routes.py` toggle; see §5 |
| **License** | cc-by-nc-4.0 (non-commercial use OK) |

### 2.4 Dense gte-large single-vector (legacy — built earlier, rich metadata)

| | |
|---|---|
| **LanceDB** | `/data/projects/rag/lancedb_data/` |
| **Model** | `Alibaba-NLP/gte-large-en-v1.5` (~434 M, 1024-dim, 8K ctx, RoPE, CLS-pooled) |
| **Server** | vLLM `--runner pooling` on GPU (was `localhost:18000` via SSH tunnel) |
| **Storage** | float32, 1024-dim, ~5.7 KB/doc → **2.9 GB total** |
| **Schema** | `id (string PK) \| vector (list<float32, 1024>) \| path (string) \| source (string) \| name (string) \| size (int64) \| mtime (float64) \| mtime_iso (string) \| text (string, truncated to 2K chars) \| snippet (string, first 240 chars) \| sha256 (string) \| embedding_model (string)` |
| **Embedder** | `embed_corpus.py` (sequential batches of 8, calls vLLM `/v1/embeddings`) |
| **Performance** | ~4.3 emb/s sustained on RTX 2060; full 512K run took ~33 h |
| **License** | MIT (commercial-friendly) |

Key differences vs the jina-v3 index:
- **Rich metadata** — stores full file metadata (mtime, sha256, truncated text, snippet) so it is usable standalone for ad-hoc search without needing Postgres or the source filesystem.
- **vLLM-served** — embeddings were generated via a running vLLM server on GPU, not via local `sentence-transformers`.
- **Text truncated to 2,000 chars** — this gave a 3.7× speedup with minimal semantic loss; the jina-v3 pipeline (§2.3) reads full docs from disk instead and does not store text.

---

## 3. End-to-end query pipeline (current)

```
user query
   │
   ▼  Phase 1 ── FAISS bi-encoder (BAAI/bge-m3)          ── top-1000 candidates
   │
   ▼  Phase 2 ── BM25 keyword (Postgres FTS or local)    ── top-1000 candidates
   │
   ▼  Phase 3 ── Postgres FTS (text search)              ── top-1000 candidates
   │
   ▼  Phase 4 ── Per-cluster leaves (Hierarchical Soft-Clustering, where present)
   │
   ▼  RRF fusion over {bi, kw, fts, leaf} ranks  ────  1,000 final candidates
   │
   ▼  Phase 5 ── Reranker (TOGGLE)
   │              ├─ COLBERT_RERANK_ENABLED=false → BAAI/bge-reranker-v2-m3 cross-encoder
   │              └─ COLBERT_RERANK_ENABLED=true  → jinaai/jina-colbert-v2 MaxSim
   │
   ▼  Score fusion: 0.7·ce_norm + 0.3·fused_norm → final ranking
   │
   ▼  Hydrate documents + cluster context → /api/v1/query response
```

The ColBERT branch in `routes.py:633-660` is the only change needed to flip the reranker.

---

## 4. Loading each database from Python

### 4.1 Open a LanceDB and inspect schema

```python
import lancedb

# ColBERT
t_cb = lancedb.connect("/data/projects/rag/data/colbert_index/db").open_table("documents")
print(t_cb.count_rows())            # 511,962
print(t_cb.schema)

# Dense jina-v3
t_dn = lancedb.connect("/data/projects/rag/data/dense_index/db").open_table("documents")
print(t_dn.count_rows())            # 511,962
print(t_dn.schema)

# Dense gte-large (legacy)
t_gt = lancedb.connect("/data/projects/rag/lancedb_data").open_table("documents")
print(t_gt.count_rows())            # 511,962
print(t_gt.schema)
```

### 4.2 Read a ColBERT doc embedding (int8 → float32)

```python
import numpy as np
EMB_DIM = 128

r = t_cb.search().where("id = 'slack/eng-oncall/dsid_xxx__20260321-kms.txt'") \
                 .limit(1).to_arrow().to_pylist()[0]
q = np.frombuffer(r["embeddings"], dtype=np.int8).reshape(r["n_tokens"], EMB_DIM)
v = q.astype(np.float32) * r["scale"]      # (n_tokens, 128) L2-normalized per row
print(v.shape, np.linalg.norm(v[0]))       # (~1500, 128)  ~1.0
```

### 4.3 Read a dense jina-v3 doc embedding

```python
r = t_dn.search().where("id = 'gmail/ben_carter/dsid_xxx__20260821.txt'") \
                 .limit(1).to_arrow().to_pylist()[0]
v = np.asarray(r["embedding"], dtype=np.float32)        # (1024,) L2-normalized
print(v.shape, np.linalg.norm(v))                       # (1024,)  1.0000
```

### 4.4 Read a dense gte-large doc embedding

```python
r = t_gt.search().where("id = 'slack/eng-oncall/dsid_xxx__20260321-kms.txt'") \
                 .limit(1).to_arrow().to_pylist()[0]
v = np.asarray(r["vector"], dtype=np.float32)           # (1024,)
print(v.shape, np.linalg.norm(v))                       # (1024,)  ~1.0
```

### 4.5 Search the gte-large LanceDB (standalone semantic search)

You need a query embedding first. Two options:

**Option A — vLLM server** (fastest, recommended for repeated queries):

```bash
vllm serve Alibaba-NLP/gte-large-en-v1.5 \
    --runner pooling --port 18000 --max-model-len 8192 \
    --dtype float16 --enforce-eager --trust-remote-code \
    --hf-overrides '{"architectures":["GteNewModel"]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18000/v1", api_key="unused")
qvec = client.embeddings.create(
    model="gte-large-en-v1.5", input=["customer escalation private upgrade rollback"]
).data[0].embedding

results = (
    t_gt.search(qvec)
     .metric("cosine")     # lower = more similar
     .where("source IN ('confluence', 'linear')")
     .limit(10)
     .to_pandas()
)
for _, row in results.iterrows():
    print(f"  score={row['_distance']:.4f}  {row['id']}")
    print(f"     {row['snippet'][:120]}")
```

**Option B — sentence-transformers** (for one-off queries):

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
qvec = model.encode("your query here").tolist()
results = t_gt.search(qvec).metric("cosine").limit(10).to_pandas()
```

### 4.6 ColBERT rerank a candidate pool (already wired into the backend)

```python
from app.ml.colbert_reranker import colbert_rerank

ranked = colbert_rerank(
    query="KMS rotation broke MFA for the gmail integration",
    doc_ids=["slack/.../kms.txt", "gmail/.../mfa.txt", ...],   # ≤ COLBERT_RERANK_POOL (1000)
    top_k=10,
)
# → [("slack/.../kms.txt", 28.7), ("gmail/.../mfa.txt", 26.1), ...]
```

Internally:
1. Encodes the query on CPU with PyLate (`get_model()` — first call ~5-15 s, cached after).
2. Fetches doc embeddings in one filtered `to_lance().to_table(filter="id IN (...)")`.
3. Dequantizes int8 → float32 with the per-doc scale.
4. Pads and runs `np.einsum("qd,nkd->nqk", Q, D)` in chunks of 100.
5. Returns sorted `(doc_id, MaxSim_score)` tuples.

### 4.7 FAISS bi-encoder (read at query time)

```python
from app.ml.faiss_index import search_faiss
hits = search_faiss(query_vec, top_k=1000)   # [(doc_id, score), ...]
```

The FAISS index is opened by `faiss_index.py:get_index()` on first call. The model itself is loaded by `embedding.py:get_bi_encoder()`.

---

## 5. What we can do (next steps)

### 5.1 Flip ColBERT reranker ON (already wired, just toggle)

Edit `backend/.env`:
```
COLBERT_RERANK_ENABLED=true
```
Restart the backend. The cross-encoder path is preserved — flipping back to `false` restores it. To A/B:

```bash
# A: baseline cross-encoder
COLBERT_RERANK_ENABLED=false python scripts/eval_enterprise_bench.py 2>&1 | tee /tmp/eval_bge.log
# B: ColBERT rerank
COLBERT_RERANK_ENABLED=true  python scripts/eval_enterprise_bench.py 2>&1 | tee /tmp/eval_colbert.log
diff /tmp/eval_bge.log /tmp/eval_colbert.log | grep -E "recall@"
```

Hypothesis: ColBERT recall ≥ baseline + 3-7 pp.

### 5.2 Wire the dense jina-v3 index in as a NEW first-stage signal

**Option A — replace the FAISS bi-encoder:**
- Add `app/ml/dense_encoder.py` (lazy singleton: load `jinaai/jina-embeddings-v3`, encode with `task="retrieval.query"`, mean-pool, L2-normalize).
- Add `app/ml/dense_search.py` (LanceDB ANN or flat scan over the 2.0 GB index).
- Toggle in `routes.py` Phase 1: `if settings.DENSE_FIRST_STAGE: use dense_search() else: use search_faiss()`.

**Option B — add as a 5th RRF signal:**
- Inject `dn_rank` into the RRF formula alongside bi, kw, fts, leaf.

### 5.3 Add new documents

For the **gte-large pipeline** (rich-metadata LanceDB at `lancedb_data/`):
```bash
source /data/projects/rag/.venv/bin/activate
python3 /data/projects/rag/embed_corpus.py \
    --root /home/shanaka/Desktop/projects/rag/data/all_documents \
    --url  http://localhost:18000/v1 \
    --db-dir /data/projects/rag/lancedb_data
```
It is **resume-safe** (skips existing `id`s). To start fresh: `rm -rf /data/projects/rag/lancedb_data`.

For the **jina-v3 / ColBERT pipelines** (server-side batch embed):
- Both `embed_dense_v2.py` and `embed_colbert.py` are resumable.
- Ship the new corpus to a vast.ai box, run the script, rsync back. See §6 for the dense recipe; see `data/colbert_index/_build/RUNBOOK.md` for the ColBERT recipe.

### 5.4 Quantization / format migration

The ColBERT index uses **int8 + per-doc scale** (~64 GB). The dense index uses **float32** (~2 GB). If storage is tight, fp16/bf16 conversion of the dense index would halve it to ~1 GB with no ANN-quality loss.

### 5.5 Build an ANN index over the dense embeddings

LanceDB's `create_index(num_partitions=…, num_sub_vectors=…)` turns flat storage into IVF-PQ. Worth doing if QPS becomes a bottleneck. At 512 K × 1024 = 2 GB flat, brute-force is ~50 ms on CPU.

---

## 6. Reproducing the dense jina-v3 build (full recipe)

**Local (one-time prep):**
```bash
cd /data/projects/rag/data/dense_index/_build
find /home/shanaka/Desktop/projects/rag/data/all_documents -name '*.txt' \
    -printf '%P\n' | sort > manifest.txt                # 511,962 rows
tar --use-compress-program='zstd -T0 -3' \
    -cf corpus.tar.zst -C /home/shanaka/Desktop/projects/rag/data \
    all_documents/{confluence,fireflies,github,gmail,google_drive,hubspot,jira,linear,slack}
sha256sum corpus.tar.zst > corpus.tar.zst.sha256
```

**Server (vast.ai):**
```bash
ssh -p 31694 root@182.224.239.168
bash /workspace/build/setup_server.sh        # ~10 min
# unpack corpus, then:
tmux new -d -s embed "/workspace/venv/bin/python -u /workspace/build/embed_dense_v2.py \
        --batch-tokens 131072 --max-tokens 4096 2>&1 | tee /workspace/build/embed.log"
tail -F /workspace/build/embed.log
```

**Local (after pipeline):**
```bash
rsync -av --partial --progress -e 'ssh -p 31694' \
    root@182.224.239.168:/workspace/lancedb_dense/ /data/projects/rag/data/dense_index/db/
python scripts/smoke_dense.py
python scripts/crosscheck_dense.py
```

The `embed_dense_v2.py` script:
- Direct `AutoModel.forward(adapter_mask=…)` (bypasses sentence-transformers overhead).
- Char-based length estimate (`chars / 3.5`) — skips 1 h pre-tokenize pass.
- Token-budget batching (`--batch-tokens 131072`) — ~21 K real tokens per forward.
- Background LanceDB writer thread — GPU never blocks on disk.
- Stable 5.7 GB peak GPU memory, 0 leaks.

---

## 7. Key scripts and what they do

| Script | Purpose | Run with |
|---|---|---|
| `scripts/smoke_dense.py` | Local-side: 511,962 rows + 5-sample L2-norm check + source distribution | `python scripts/smoke_dense.py` |
| `scripts/crosscheck_dense.py` | Server-vs-local ID-based SHA256 cross-check | `python scripts/crosscheck_dense.py` |
| `scripts/smoke_colbert.py` | Local-side: 20 random doc IDs → ColBERT MaxSim ranking | `python scripts/smoke_colbert.py` |
| `scripts/check_quant_fidelity.py` | ColBERT int8 vs raw dequant MaxSim drift (< 0.5 %) | `python scripts/check_quant_fidelity.py` |
| `scripts/bench_colbert_latency.py` | p50/p95/p99 over 10 queries × 1000-doc pool | `python scripts/bench_colbert_latency.py --pool 1000 --queries 10` |
| `scripts/eval_enterprise_bench.py` | Recall@K vs gold pairs | `python scripts/eval_enterprise_bench.py` |
| `scripts/build_faiss_index.py` | Builds the bge-m3 FAISS index | `python scripts/build_faiss_index.py` |
| `embed_corpus.py` | Legacy vLLM-based pipeline for gte-large (resume-safe) | `python embed_corpus.py --root … --url … --db-dir …` |
| `data/dense_index/_build/server_scripts/embed_dense_v2.py` | Server-side jina-v3 batch embed | run on GPU box |
| `data/colbert_index/_build/server_scripts/embed_colbert.py` | Server-side ColBERT batch embed | run on GPU box |

---

## 8. Environment / dependencies

| | |
|---|---|
| **Python (local backend venv)** | 3.12 — `backend/venv/` |
| **Key local packages** | `fastapi 0.115.6`, `pydantic 2.10.4`, `sentence-transformers 3.3.1`, `torch 2.5.1`, `faiss-cpu 1.14.2`, `pylate 1.2.0`, `lancedb 0.33.0`, `transformers 4.45.2`, `einops 0.8.0`, `pyarrow 17.0.0`, `pylance 7.0.0` |
| **Legacy venv** | `/data/projects/rag/.venv/` — used by `embed_corpus.py`; has `lancedb`, `pylance`, `pandas`, `openai` |
| **Local models cached at** | `~/.cache/huggingface/hub/` (bge-m3, bge-reranker-v2-m3, jina-colbert-v2); jina-embeddings-v3 auto-downloaded on first use |
| **Server env (for re-running)** | torch 2.4.1+cu124, flash-attn 2.6.3, transformers 4.45.2, pylate 1.2.0, sentence-transformers 3.3.1, lancedb 0.33.0 — see `data/dense_index/_build/server_scripts/setup_server.sh` |

---

## 9. Storage footprint summary

| Index | Local path | Size | Rows | Notes |
|---|---|---|---:|---|
| **FAISS bi-encoder** (`bge-m3`) | `/app/backend/faiss_index` | ~2 GB | 487,249 | FastAPI Phase 1 |
| **ColBERT int8** (`jina-colbert-v2`) | `/data/projects/rag/data/colbert_index/db` | 64 GB | 511,962 | Phase 5 reranker |
| **Dense fp32** (`jina-embeddings-v3`) | `/data/projects/rag/data/dense_index/db` | 2.0 GB | 511,962 | Not yet wired |
| **Dense gte-large** (`Alibaba-NLP/gte-large-en-v1.5`) | `/data/projects/rag/lancedb_data` | 2.9 GB | 511,962 | Rich metadata + snippet |
| **Source corpus** | `/home/shanaka/Desktop/projects/rag/data/all_documents` | 3.3 GB | 511,962 | 9 source subtrees |

---

## 10. Troubleshooting

### "Connection refused" to vLLM (for gte-large queries)

```bash
vllm serve Alibaba-NLP/gte-large-en-v1.5 \
    --runner pooling --port 18000 --max-model-len 8192 \
    --dtype float16 --enforce-eager --trust-remote-code \
    --hf-overrides '{"architectures":["GteNewModel"]}'
```

The `--hf-overrides` is **mandatory** — the model declares architecture `NewModel` but vLLM's native `GteNewModel` matches the same weights.

### "No module named lance" / LanceDB import errors

```bash
# For the backend venv:
data/projects/rag/backend/venv/bin/pip install pylance

# For the legacy venv:
source /data/projects/rag/.venv/bin/activate
pip install pylance
```

### Slow searches on the gte-large LanceDB

- Build a vector index: `t.create_index(num_partitions=64, num_sub_vectors=64)` (only worth it for >1M rows; at 512 K, flat scan is ~50 ms)
- Pre-filter with `.where()` (by `source` or `mtime`) before vector search
- Reduce `.limit()` if you only need top-5 or top-10

### Re-embedding changed documents

The legacy `embed_corpus.py` checks path existence, not content. To re-embed a changed file:
1. Delete the old row manually: `db.open_table('documents').delete("id = 'path/to/file.txt'")`
2. Or rename the file (changes the `id`)
3. Re-run `embed_corpus.py` — it will pick up the new file.

### ColBERT reranker OOM

If `scripts/bench_colbert_latency.py` shows p95 > 1.2 s, lower `COLBERT_RERANK_POOL` to 500 in `backend/.env` and restart. The MaxSim computation is chunked at 100 docs per einsum call; memory is usually not the bottleneck, but latency on CPU for 1000 docs can be.

---

## 11. Open work / not yet done

- [ ] Run `scripts/eval_enterprise_bench.py --reranker=colbert` to measure ColBERT recall lift vs cross-encoder (the blocker for flipping `COLBERT_RERANK_ENABLED=true` in production).
- [ ] Run `scripts/bench_colbert_latency.py` to confirm p95 < 1.2 s for the 1000-doc pool.
- [ ] Implement the dense jina-v3 first-stage wiring (Option A or B in §5.2). The DB and the model are ready; only the loaders and a `routes.py` toggle are missing.
- [ ] Re-run a small recall A/B on the dense index after wiring it in.
- [ ] Build an IVF-PQ ANN index over the 2.0 GB dense LanceDB if QPS becomes a bottleneck (probably not needed for 512 K docs).
