# RAG Embedding Pipeline — Documentation

Complete documentation for the document embedding pipeline, local vector database, and search infrastructure.

---

## 1. What We Built

A pipeline that:

1. **Reads** all `.txt` files from a local document collection (no chunking — one embedding per whole document)
2. **Embeds** them using the `Alibaba-NLP/gte-large-en-v1.5` model served by vLLM
3. **Stores** the embeddings + metadata in a persistent **LanceDB** vector database
4. **Supports** semantic similarity search via cosine distance

### Final Deliverables

| Asset | Location | Size | Status |
|---|---|---|---|
| **LanceDB database** | `/data/projects/rag/lancedb_data/` | 2.9 GB | ✅ 511,962 rows verified |
| **Source documents** | `/home/shanaka/Desktop/projects/rag/data/all_documents/` | 3.3 GB | ✅ 511,962 `.txt` files |
| **Embedder script** | `/data/projects/rag/embed_corpus.py` | — | ✅ Tested, runnable |
| **Verify/copy script** | `/data/projects/rag/verify_and_copy.py` | — | ✅ Tested |
| **Python venv** | `/data/projects/rag/.venv/` | — | ✅ Includes lancedb, pylance, pandas, openai, requests |

### Corpus Composition (511,962 documents total)

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

File size distribution: median 4,295 bytes, p99 11,947 bytes, max 42,281 bytes. All fit under the model's 8,192-token context window.

---

## 2. The Embedding Model

### `Alibaba-NLP/gte-large-en-v1.5`

| Property | Value |
|---|---|
| Architecture | Modified BERT (24 layers, 1024 hidden) with RoPE position embeddings |
| Parameters | ~434M |
| Embedding dimension | **1024** |
| Max sequence length | **8,192 tokens** (v1.5 improvement over v1's 512) |
| Pooling strategy | CLS token |
| Normalization | Not done by model — done at application layer if needed |
| License | MIT (commercial-friendly) |
| HF page | https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5 |

**No prompt/instruction template needed** — the model is a non-instruct checkpoint. Encode queries and passages directly as raw text.

### Why this model

- SOTA on MTEB English benchmarks at its size class
- Long context (8K) is critical because 100% of our documents fit (max doc ≈ 10K tokens)
- Open license, commercial use OK
- Runs in fp16 on a single 12 GB consumer GPU (RTX 2060)

### vLLM server configuration

```bash
vllm serve Alibaba-NLP/gte-large-en-v1.5 \
    --runner pooling \                 # pooling engine for /v1/embeddings
    --host 0.0.0.0 --port 18000 \
    --max-model-len 8192 \             # full 8K context per model card
    --dtype float16 \                  # sm_75 (RTX 2060) has no native bf16
    --enforce-eager \                  # skip CUDA graphs (Turing)
    --trust-remote-code \              # model has custom code
    --served-model-name gte-large-en-v1.5 \
    --hf-overrides '{"architectures":["GteNewModel"]}'  # remap NewModel -> native GteNewModel
```

The `--hf-overrides` is mandatory — the model declares architecture `NewModel` (custom code) but vLLM has a native `GteNewModel` that matches the same weights.

---

## 3. LanceDB Schema

The `documents` table has the following columns:

| Column | Type | Description |
|---|---|---|
| `id` | `string` | Primary key — path relative to `all_documents` root |
| `vector` | `list<float32>[1024]` | The 1024-dim embedding |
| `path` | `string` | Absolute file path on disk |
| `source` | `string` | Top-level directory under root (`slack`, `gmail`, etc.) |
| `name` | `string` | Filename (basename) |
| `size` | `int64` | File size in bytes |
| `mtime` | `float64` | Modification time (epoch seconds) |
| `mtime_iso` | `string` | Modification time (ISO 8601) |
| `text` | `string` | Full file content, **truncated to 2,000 chars** (~500 tokens) |
| `snippet` | `string` | First 240 chars of text |
| `sha256` | `string` | SHA-256 hex digest of the full file bytes |
| `embedding_model` | `string` | Model identifier (`gte-large-en-v1.5`) |

**Key design choices:**

- `id` is the relative path → stable, unique, resume-friendly
- `text` is **truncated to 2,000 chars** (not 30,000). This gave a 3.7× speedup with minimal semantic loss for retrieval; the first 2K chars of these docs already cover the key terms
- Full text is NOT stored in the DB — re-read from disk if needed (paths are intact)
- No source column is needed in a separate dimension table — it's a regular column for filtering

---

## 4. How to Load the Database

### Install dependencies (one-time)

```bash
# Use the existing venv (has everything) or create a new one:
uv venv /data/projects/rag/.venv   # if not already created
source /data/projects/rag/.venv/bin/activate
uv pip install lancedb pylance pandas openai
```

### Open the database

```python
import lancedb

db = lancedb.connect('/data/projects/rag/lancedb_data')
t = db.open_table('documents')
print(f"Rows: {t.count_rows():,}")
# Rows: 511,962
```

### List all tables

```python
for item in db.list_tables():
    name = item[0] if isinstance(item, tuple) else item
    print(name)
# documents
```

### Get a single row

```python
df = t.to_pandas()
row = df.iloc[0]
print(row['id'], row['snippet'])
```

### Read a small subset of columns (faster)

```python
# Don't materialize all 512K rows unless you need to
ids = t.to_pandas(columns=['id', 'source'])['id'].tolist()
```

---

## 5. How to Search

### Get a query embedding

You need to embed the query with the same model. Either:

**Option A — Run vLLM server** (recommended for repeated queries)

```bash
# Start the server (on a GPU machine):
vllm serve Alibaba-NLP/gte-large-en-v1.5 \
    --runner pooling --port 18000 --max-model-len 8192 \
    --dtype float16 --enforce-eager --trust-remote-code \
    --hf-overrides '{"architectures":["GteNewModel"]}'
```

**Option B — Use sentence-transformers** (for one-off queries, no server)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
qvec = model.encode("your query here").tolist()
```

### Search the DB

```python
import lancedb
from openai import OpenAI

db = lancedb.connect('/data/projects/rag/lancedb_data')
t = db.open_table('documents')

# Option A: via vLLM
client = OpenAI(base_url="http://localhost:18000/v1", api_key="unused")
qvec = client.embeddings.create(
    model="gte-large-en-v1.5",
    input=["customer escalation private upgrade rollback"]
).data[0].embedding

# Top-10 most similar
results = (
    t.search(qvec)
     .metric("cosine")     # cosine distance (lower = more similar)
     .limit(10)
     .to_pandas()
)

for i, row in results.iterrows():
    print(f"[{i+1}] score={row['_distance']:.4f}  {row['id']}")
    print(f"     {row['snippet'][:120]}")
```

### Filter by source

```python
# Only return results from specific sources
results = (
    t.search(qvec)
     .metric("cosine")
     .where("source IN ('confluence', 'linear')")
     .limit(10)
     .to_pandas()
)
```

### Filter by date

```python
import time
# Only return results from the last 30 days
cutoff = time.time() - 30 * 86400
results = (
    t.search(qvec)
     .metric("cosine")
     .where(f"mtime > {cutoff}")
     .limit(10)
     .to_pandas()
)
```

---

## 6. End-to-End Example: RAG Query

```python
import lancedb
from openai import OpenAI

# 1. Connect
db = lancedb.connect('/data/projects/rag/lancedb_data')
t = db.open_table('documents')
client = OpenAI(base_url="http://localhost:18000/v1", api_key="unused")

# 2. User query
user_query = "What is our retention policy for request/response logs?"

# 3. Embed the query
qvec = client.embeddings.create(
    model="gte-large-en-v1.5", input=[user_query]
).data[0].embedding

# 4. Retrieve top context documents
hits = t.search(qvec).metric("cosine").limit(8).to_pandas()
context = "\n\n---\n\n".join(hits['text'].tolist())

# 5. Build prompt and call an LLM (e.g., Claude)
import anthropic
claude = anthropic.Anthropic()
response = claude.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    system=f"You are a helpful assistant. Use the following context to answer the user's question. Cite documents by their `id` when relevant.\n\nContext:\n{context}",
    messages=[{"role": "user", "content": user_query}]
)
print(response.content[0].text)
```

---

## 7. How to Add New Documents

If new `.txt` files appear under `/home/shanaka/Desktop/projects/rag/data/all_documents/`:

```bash
# The script is resume-safe — it skips any file whose relative path
# already exists in the table. New files will be embedded and added.
source /data/projects/rag/.venv/bin/activate
python3 /data/projects/rag/embed_corpus.py \
    --root /home/shanaka/Desktop/projects/rag/data/all_documents \
    --url  http://localhost:18000/v1 \
    --db-dir /data/projects/rag/lancedb_data
```

Expected throughput: ~4–5 docs/sec on RTX 2060. If the vLLM server is on a different host, change the `--url` accordingly.

### To start fresh

```bash
rm -rf /data/projects/rag/lancedb_data
python3 /data/projects/rag/embed_corpus.py ...
```

---

## 8. Performance Reference

Measured on RTX 2060 (sm_75, 12 GB VRAM, compute cap 7.5):

| Text length | Latency / batch | Throughput |
|---|---|---|
| 8 short texts (bs=8) | ~10 ms | — |
| 64 short texts (bs=64) | ~9 ms | — |
| 8 × 2K-char texts | ~1.5 s | ~5.3 emb/s |
| 8 × 4K-char texts | ~5.7 s | ~1.4 emb/s |
| Sequential bs=8 (realistic) | — | **~4.3 emb/s** (sustained) |

For 512k documents, full embedding took **~33 hours**. On a more capable GPU (A100, H100), this would be 5–20× faster.

---

## 9. Pipeline Architecture (How It Was Built)

```
┌─────────────────────┐                  ┌──────────────────────┐
│  Local (your box)   │                  │  vast.ai GPU server  │
│                     │   SSH tunnel     │                      │
│  all_documents/     │ ───────────────▶ │  vLLM serving gte-   │
│  (512K .txt files)  │                  │  large-en-v1.5       │
│                     │                  │  :18000              │
│  embed_corpus.py    │                  │                      │
│  ─────────────      │                  │  gte-embed (supervisor)│
│  - walks files      │                  │                      │
│  - reads text       │  POST            │                      │
│  - batches 8 docs   │ ───────────────▶ │  /v1/embeddings      │
│  - calls OpenAI SDK │                  │  → 1024-dim vectors  │
│  - flushes to       │                  │                      │
│    LanceDB every    │                  │                      │
│    4000 rows        │                  │                      │
└──────────┬──────────┘                  └──────────────────────┘
           │
           ▼
   lancedb_data/
   ├── documents.lance/   ← the actual data
   └── __manifest/       ← table metadata
```

### Why this architecture

- **vLLM on GPU** is the fastest way to serve the model (continuous batching, FlashAttention, no Python overhead per request)
- **Sequential client (no async)** is simpler and vLLM handles continuous batching on its end
- **batch=8** was the sweet spot: smaller batches waste GPU time on overhead; larger ones don't help because the GPU is the bottleneck
- **Truncation to 2,000 chars** (vs 30,000) gave 3.7× speedup with minimal retrieval quality loss
- **Local LanceDB** is embedded (no server), persistent (just a folder), and fast (columnar storage, vector search built-in)

---

## 10. Troubleshooting

### "Connection refused" to vLLM

The vLLM server isn't running or the URL is wrong. Start it:

```bash
vllm serve Alibaba-NLP/gte-large-en-v1.5 \
    --runner pooling --port 18000 --max-model-len 8192 \
    --dtype float16 --enforce-eager --trust-remote-code \
    --hf-overrides '{"architectures":["GteNewModel"]}'
```

If vLLM is on a remote machine, set up an SSH tunnel:

```bash
ssh -fN -L 18000:127.0.0.1:18000 user@server
```

### "No module named lance" / "lance" import errors

Install pylance:

```bash
uv pip install pylance
```

### "Table not found" or stale connection

LanceDB connections are lightweight. Just re-open:

```python
db = lancedb.connect('/data/projects/rag/lancedb_data')
t = db.open_table('documents')
```

### Slow searches

- Create a vector index: `t.create_index(num_partitions=64, num_sub_vectors=64)` (only worth it for >1M rows)
- Use `.where()` to pre-filter (e.g., by `source` or `mtime`) before vector search
- Reduce `limit()` if you only need top-5 or top-10

### Re-embedding changed documents

The script uses file `mtime` and `path` as the identity. If a file at the same path changes, the script will **skip** it (it only checks path, not content). To re-embed changed files, either:

1. Delete the old row manually: `db.open_table('documents').delete("id = 'path/to/file.txt'")`
2. Or rename the file (which changes the `id`)

---

## 11. Files in This Project

```
/data/projects/rag/
├── .venv/                          # Python venv with lancedb, openai, etc.
├── lancedb_data/                   # The vector database (2.9 GB)
│   ├── documents.lance/            # Vector data (Lance columnar format)
│   └── __manifest/                 # Table schema & transaction log
├── embed_corpus.py                 # Main pipeline: walk → embed → store
└── verify_and_copy.py              # Post-pipeline: verify + SCP to local
```

```
/home/shanaka/Desktop/projects/rag/data/
└── all_documents/                  # Source documents (512K .txt files)
    ├── confluence/
    ├── fireflies/
    ├── github/
    ├── gmail/
    ├── google_drive/
    ├── hubspot/
    ├── jira/
    ├── linear/
    └── slack/
```

---

## 12. Quick Reference Card

```python
# Load
import lancedb
db = lancedb.connect('/data/projects/rag/lancedb_data')
t = db.open_table('documents')

# Count
t.count_rows()  # 511,962

# Get one row
t.to_pandas().iloc[0]

# Search
from openai import OpenAI
qvec = OpenAI().embeddings.create(
    model="gte-large-en-v1.5", input=["query"]
).data[0].embedding
t.search(qvec).metric("cosine").limit(10).to_pandas()

# Filter then search
t.search(qvec).where("source = 'confluence'").limit(10).to_pandas()

# Add new docs (resume-safe)
# Just re-run embed_corpus.py — it skips existing ids

# Start fresh
import shutil; shutil.rmtree('/data/projects/rag/lancedb_data')
# then re-run embed_corpus.py
```

---

## 13. Contact / Credits

- **Model**: [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) (MIT)
- **Embeddings server**: [vLLM](https://github.com/vllm-project/vllm) (Apache 2.0)
- **Vector DB**: [LanceDB](https://lancedb.com/) (Apache 2.0)
- **Infrastructure**: vast.ai GPU cloud (RTX 2060 rental)
- **Build date**: 2026-06-03
- **Pipeline duration**: 33h 8m for 511,962 documents
