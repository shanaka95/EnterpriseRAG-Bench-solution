#!/usr/bin/env python3
"""
Build a RAPTOR-style hierarchical chunk graph for the RAPTOR-VIZ UI.

This is a smaller sibling of scripts/build_raptor_graph.py that:
  - Pulls the source corpus from HuggingFace (onyx-dot-app/EnterpriseRAG-Bench)
    instead of the on-disk data/all_documents/ tree, and gets structured
    per-doc metadata (title, source_type) for free
  - Limits the doc set to a subset (default: every doc referenced by at
    least one question's expected_doc_ids), so the graph is small enough
    to render fully in the browser
  - Writes to data/raptor_viz/chunks.lance with two extra columns:
    `title` (from HF) and `content_preview` (first 500 chars of the slice)

The chunking algorithm is imported unchanged from
backend.app.ml.raptor_chunking — the same one used by the production
build script — so the UI shows EXACTLY the same splits a real RAPTOR
build would produce.

Pipeline:
  1. (optional) download HF docs + questions -> data/raptor_viz/{docs,questions}.parquet
  2. Load questions -> set of expected_doc_ids -> filter HF docs
  3. For each selected doc:
     a. Tokenize with JinaBERT (return_offsets_mapping)
     b. Look up ColBERT embeddings (jina-colbert-v2) from the existing
        data/colbert_index/db/documents.lance
     c. Run the recursive chunker (recurse_chunk from raptor_chunking.py)
     d. Re-tokenize each leaf with Qwen3 (jina-v5 tokenizer) for n_tokens_v5
  4. Background writer thread flushes to data/raptor_viz/chunks.lance

Resumable: skips doc_ids already present in the output table.

Wall time for ~400 docs: 10-15 min on a local CPU box (ColBERT lookups dominate).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

# Make backend importable
BACKEND_ROOT = Path("/data/projects/rag/backend")
sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("HF_HOME", str(Path("/data/projects/rag/.cache/huggingface").resolve()))

import numpy as np
import pyarrow as pa
import lancedb

from app.ml.raptor_chunking import (
    recurse_chunk, recurse_chunk_semantic, make_node_id, ChunkNode,
    COLBERT_MAX_SEQ,
)


# ----------------------------- config ---------------------------------------

VIZ_DIR = Path("/data/projects/rag/data/raptor_viz")
DOCS_PARQUET = VIZ_DIR / "docs.parquet"
QUESTIONS_PARQUET = VIZ_DIR / "questions.parquet"
SELECTED_DOCS = VIZ_DIR / "selected_doc_ids.txt"

COLBERT_LANCE = "/data/projects/rag/data/colbert_index/db"
COLBERT_TABLE = "documents"
COLBERT_DIM = 128

CHUNKS_TABLE = "chunks"
PREVIEW_CHARS = 500

# Writer
WRITE_QUEUE_MAX = 4_000
WRITE_FLUSH_ROWS = 4_000  # smaller than production — viz graph is much smaller

# LanceDB schema (production schema + title + content_preview)
SCHEMA = pa.schema([
    pa.field("node_id", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("doc_name", pa.string()),
    pa.field("source", pa.string()),
    pa.field("title", pa.string()),
    pa.field("level", pa.int32()),
    pa.field("parent_id", pa.string()),
    pa.field("first_child_id", pa.string()),
    pa.field("next_sibling_id", pa.string()),
    pa.field("n_siblings", pa.int32()),
    pa.field("sibling_idx", pa.int32()),
    pa.field("start_char", pa.int32()),
    pa.field("end_char", pa.int32()),
    pa.field("n_chars", pa.int32()),
    pa.field("n_tokens_colbert", pa.int32()),
    pa.field("n_tokens_v5", pa.int32()),
    pa.field("is_leaf", pa.bool_()),
    pa.field("boundary_score", pa.float32()),
    pa.field("content_preview", pa.string()),
])


# ----------------------------- helpers --------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------- HF download ----------------------------------

def ensure_hf_parquet(out_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """Make sure docs.parquet + questions.parquet exist in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if DOCS_PARQUET.exists() and QUESTIONS_PARQUET.exists() and not force:
        log(f"Reusing existing HF cache: {DOCS_PARQUET}, {QUESTIONS_PARQUET}")
        return DOCS_PARQUET, QUESTIONS_PARQUET

    log("Downloading EnterpriseRAG-Bench (questions + documents) from HuggingFace ...")
    from datasets import load_dataset
    qs = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions", split="test")
    docs = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "documents", split="test")
    qs.to_parquet(str(QUESTIONS_PARQUET))
    docs.to_parquet(str(DOCS_PARQUET))
    log(f"  questions: {len(qs):,} -> {QUESTIONS_PARQUET}")
    log(f"  documents: {len(docs):,} -> {DOCS_PARQUET}")
    return DOCS_PARQUET, QUESTIONS_PARQUET


# ----------------------------- data loaders ---------------------------------

def load_questions() -> list[dict]:
    """Read questions.parquet -> list of dicts."""
    import pyarrow.parquet as pq
    t = pq.read_table(str(QUESTIONS_PARQUET))
    return t.to_pylist()


def load_docs_subset(expected_ids: set[str]) -> dict[str, dict]:
    """Read docs.parquet, filter to those with dsid in expected_ids.

    Returns: dict dsid -> {doc_id, source_type, title, content}
    """
    import pyarrow.parquet as pq
    t = pq.read_table(str(DOCS_PARQUET), columns=["doc_id", "source_type", "title", "content"])
    out: dict[str, dict] = {}
    rows = t.to_pylist()
    for r in rows:
        did = r["doc_id"]
        if did in expected_ids:
            out[did] = {
                "doc_id": did,
                "source_type": r["source_type"],
                "title": r.get("title", "") or "",
                "content": r["content"],
            }
    return out


# ----------------------------- tokenizers (singletons) -----------------------

_jina_tok = None
_jina_lock = threading.Lock()
_v5_tok = None
_v5_lock = threading.Lock()


def get_jina_tokenizer():
    global _jina_tok
    if _jina_tok is None:
        with _jina_lock:
            if _jina_tok is None:
                from transformers import AutoTokenizer
                log("Loading jina-colbert-v2 (JinaBERT) tokenizer (CPU) ...")
                _jina_tok = AutoTokenizer.from_pretrained(
                    "jinaai/jina-colbert-v2", trust_remote_code=True,
                )
    return _jina_tok


def get_v5_tokenizer():
    global _v5_tok
    if _v5_tok is None:
        with _v5_lock:
            if _v5_tok is None:
                from transformers import AutoTokenizer
                log("Loading jina-v5 (Qwen3) tokenizer (CPU) ...")
                _v5_tok = AutoTokenizer.from_pretrained(
                    "jinaai/jina-embeddings-v5-text-small", trust_remote_code=True,
                )
    return _v5_tok


# ----------------------------- colbert ---------------------------------------

_colbert_table = None
_colbert_lock = threading.Lock()


def get_colbert_table():
    global _colbert_table
    if _colbert_table is None:
        with _colbert_lock:
            if _colbert_table is None:
                db = lancedb.connect(COLBERT_LANCE)
                _colbert_table = db.open_table(COLBERT_TABLE)
                log(f"ColBERT index opened: {_colbert_table.count_rows():,} rows")
    return _colbert_table


def fetch_colbert_for_doc(doc_id: str) -> np.ndarray | None:
    """Fetch ColBERT for a doc whose id is the bare dsid_xxx.

    The ColBERT index uses full relative paths (e.g. 'slack/.../dsid_xxx__file.txt'),
    so we use a LIKE filter to find the row. There is at most one row per dsid.
    """
    safe = doc_id.replace("'", "''")
    t = get_colbert_table()
    arrow = (
        t.to_lance()
         .to_table(columns=["id", "n_tokens", "scale", "embeddings"],
                   filter=f"id LIKE '%{safe}%'")
    )
    if arrow.num_rows == 0:
        return None
    if arrow.num_rows > 1:
        log(f"  WARN: {arrow.num_rows} ColBERT rows for {doc_id}, using first")
    n = int(arrow.column("n_tokens")[0].as_py())
    scale = float(arrow.column("scale")[0].as_py())
    emb_blob = arrow.column("embeddings")[0].as_py()
    if n == 0 or not emb_blob:
        return None
    return (
        np.frombuffer(emb_blob, dtype=np.int8)
          .reshape(n, COLBERT_DIM)
          .astype(np.float32) * scale
    )


# ----------------------------- writer ----------------------------------------

class WriterThread(threading.Thread):
    def __init__(self, table, q: "queue.Queue"):
        super().__init__(daemon=True)
        self.table = table
        self.q = q
        self.rows_written = 0
        self.errors = 0
        self.stop = threading.Event()

    def run(self):
        buf = {f.name: [] for f in SCHEMA}

        def flush():
            n = len(buf["node_id"])
            if n == 0:
                return
            try:
                ab = pa.table({
                    k: pa.array(v, type=SCHEMA.field(k).type) for k, v in buf.items()
                }, schema=SCHEMA)
                self.table.add(ab)
                self.rows_written += n
            except Exception as e:
                self.errors += 1
                log(f"  WRITER ERROR: {e}")
            for v in buf.values():
                v.clear()

        while True:
            try:
                item = self.q.get(timeout=1.0)
            except queue.Empty:
                if self.stop.is_set():
                    break
                continue
            if item is None:
                break
            rows = item
            for r in rows:
                for k, v in r.items():
                    buf[k].append(v)
            if len(buf["node_id"]) >= WRITE_FLUSH_ROWS:
                flush()
        flush()
        log(f"  writer thread exit: {self.rows_written:,} rows, {self.errors} errors")


# ----------------------------- per-doc pipeline ------------------------------

def build_doc(doc_id: str, text: str, title: str, source_type: str,
              v5_tok, chunking: str = "semantic") -> list[dict] | None:
    """Build the chunk graph for one document; return list of row dicts.

    chunking: "token" (legacy MaxSim-on-token-windows) or "semantic"
              (sentence-level sliding-window cosine distance, percent
              breakpoints). Default: "semantic".
    """
    # Tokenize with JinaBERT
    jina_tok = get_jina_tokenizer()
    enc = jina_tok(
        text, return_offsets_mapping=True, add_special_tokens=False,
        truncation=True, max_length=COLBERT_MAX_SEQ,
    )
    offsets = enc["offset_mapping"]
    n_tokens = len(enc["input_ids"])
    if n_tokens == 0:
        return None

    # Fetch ColBERT
    colbert_vecs = fetch_colbert_for_doc(doc_id)
    if colbert_vecs is None or colbert_vecs.shape[0] == 0:
        return None
    n_colbert = colbert_vecs.shape[0]
    n_tokens = min(n_tokens, n_colbert)
    offsets = offsets[:n_tokens]

    # Build graph
    if chunking == "semantic":
        nodes = recurse_chunk_semantic(
            text=text,
            start_char=0, end_char=len(text),
            colbert_vecs=colbert_vecs,
            char_offsets=offsets,
            level=0, parent_id=None,
            sibling_idx=0, n_siblings=1,
            doc_id=doc_id,
        )
    else:
        nodes = recurse_chunk(
            start_tok=0, end_tok=n_tokens,
            colbert_vecs=colbert_vecs,
            char_offsets=offsets,
            start_char=0, end_char=len(text),
            level=0, parent_id=None,
            sibling_idx=0, n_siblings=1,
            doc_id=doc_id,
        )
    if not nodes:
        return None

    # Assign node_ids
    for nd in nodes:
        nd.node_id = make_node_id(doc_id, nd.level, nd.start_char, nd.end_char)

    # Wire parent/child links
    parents = [nd for nd in nodes if not nd.is_leaf]
    for p in parents:
        kids = [c for c in nodes if c.parent_id is None and c.level == p.level + 1
                and c.start_char >= p.start_char and c.end_char <= p.end_char]
        kids.sort(key=lambda x: x.start_char)
        if kids:
            p.first_child_id = kids[0].node_id
            for i, c in enumerate(kids):
                c.parent_id = p.node_id
                if i + 1 < len(kids):
                    c.next_sibling_id = kids[i + 1].node_id

    # Compute n_tokens_v5 for leaves
    leaves = [nd for nd in nodes if nd.is_leaf]
    if leaves:
        leaf_texts = [text[nd.start_char:nd.end_char] for nd in leaves]
        v5_enc = v5_tok(
            leaf_texts, add_special_tokens=False, truncation=True, max_length=COLBERT_MAX_SEQ,
        )
        v5_ntoks = [len(ids) for ids in v5_enc["input_ids"]]
    else:
        v5_ntoks = []

    leaf_idx = 0
    rows: list[dict] = []
    for nd in nodes:
        if nd.is_leaf:
            n_v5 = v5_ntoks[leaf_idx] if leaf_idx < len(v5_ntoks) else nd.n_tokens_colbert
            leaf_idx += 1
        else:
            n_v5 = -1
        # content_preview: first N chars of the slice, whitespace-stripped
        slice_text = text[nd.start_char:nd.end_char]
        preview = " ".join(slice_text.split())[:PREVIEW_CHARS]
        if len(slice_text) > PREVIEW_CHARS:
            preview += "..."
        rows.append({
            "node_id": nd.node_id,
            "doc_id": doc_id,
            "doc_name": doc_id,            # HF only gives the bare dsid
            "source": source_type,
            "title": title or "",
            "level": nd.level,
            "parent_id": nd.parent_id or "",
            "first_child_id": nd.first_child_id or "",
            "next_sibling_id": nd.next_sibling_id or "",
            "n_siblings": nd.n_siblings,
            "sibling_idx": nd.sibling_idx,
            "start_char": nd.start_char,
            "end_char": nd.end_char,
            "n_chars": nd.end_char - nd.start_char,
            "n_tokens_colbert": nd.n_tokens_colbert,
            "n_tokens_v5": n_v5,
            "is_leaf": nd.is_leaf,
            "boundary_score": -1.0 if nd.boundary_score is None else float(nd.boundary_score),
            "content_preview": preview,
        })
    return rows


# ----------------------------- resume ----------------------------------------

def existing_doc_ids(table) -> set[str]:
    n = table.count_rows()
    if n == 0:
        return set()
    log(f"Resuming: scanning {n:,} existing rows for unique doc_ids ...")
    t0 = time.time()
    rows = table.to_lance().to_table(columns=["doc_id"]).to_pylist()
    ids = {r["doc_id"] for r in rows}
    log(f"  found {len(ids):,} unique doc_ids in {time.time()-t0:.1f}s")
    return ids


# ----------------------------- main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-hf", action="store_true",
                    help="Re-download HF data (overwrite parquets)")
    ap.add_argument("--out", default=str(VIZ_DIR))
    ap.add_argument("--n-docs", type=int, default=0,
                    help="Limit to N docs (0 = all referenced).")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="Process at most N new docs this run (0 = unlimited).")
    ap.add_argument("--batch", type=int, default=200,
                    help="Per-doc batch size for tokenization roundtrips.")
    ap.add_argument("--colbert-cache-size", type=int, default=1000,
                    help="How many doc_ids to batch into one ColBERT SQL query.")
    ap.add_argument("--chunking", choices=["token", "semantic"], default="semantic",
                    help="Chunking algorithm: 'token' (legacy ColBERT MaxSim on "
                         "64-token windows) or 'semantic' (sentence-level "
                         "sliding-window cosine distance, 90th-percentile "
                         "breakpoints). Default: 'semantic'.")
    args = ap.parse_args()
    log(f"args: {vars(args)}")

    out_dir = Path(args.out)
    ensure_hf_parquet(out_dir, force=args.download_hf)

    # Load questions, collect referenced dsids
    questions = load_questions()
    expected_ids: set[str] = set()
    for q in questions:
        for did in q.get("expected_doc_ids", []):
            expected_ids.add(did)
    log(f"questions={len(questions)}  unique_expected_doc_ids={len(expected_ids)}")

    # Load HF docs, filter to referenced ids
    docs = load_docs_subset(expected_ids)
    log(f"docs with content from HF: {len(docs)}")

    if args.n_docs:
        # Take first N alphabetically (deterministic, reproducible) and write the list
        sorted_ids = sorted(docs.keys())[: args.n_docs]
        docs = {k: docs[k] for k in sorted_ids}
        log(f"  limited to first {len(docs)} (alphabetical)")
    SELECTED_DOCS.write_text("\n".join(sorted(docs.keys())) + "\n")
    log(f"wrote {SELECTED_DOCS} ({len(docs)} ids)")

    # Source distribution
    src_counts: dict[str, int] = {}
    for d in docs.values():
        src_counts[d["source_type"]] = src_counts.get(d["source_type"], 0) + 1
    log(f"source distribution: {src_counts}")

    # Open output LanceDB
    db = lancedb.connect(str(out_dir))
    if CHUNKS_TABLE in db.table_names():
        table = db.open_table(CHUNKS_TABLE)
        log(f"Opened existing {CHUNKS_TABLE}: {table.count_rows():,} rows")
    else:
        empty = pa.table({f.name: pa.array([], type=f.type) for f in SCHEMA}, schema=SCHEMA)
        table = db.create_table(CHUNKS_TABLE, empty, mode="create")
        log(f"Created fresh {CHUNKS_TABLE} at {out_dir}")

    done = existing_doc_ids(table)

    # Pre-load tokenizers
    get_jina_tokenizer()
    v5_tok = get_v5_tokenizer()

    # Open ColBERT
    get_colbert_table()

    # Writer thread
    write_q: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAX)
    writer = WriterThread(table, write_q)
    writer.start()

    # Stats
    stats = {
        "docs_total": len(docs),
        "docs_processed": 0,
        "docs_skipped": 0,
        "docs_missing_text": 0,
        "docs_missing_colbert": 0,
        "docs_error": 0,
        "nodes_written": 0,
        "leaves_written": 0,
        "max_level_seen": 0,
        "max_leaf_tokens": 0,
        "source_distribution": src_counts,
        "n_questions": len(questions),
        "n_unique_expected_ids": len(expected_ids),
    }
    start = time.time()
    last_log = start
    new_this_run = 0

    pending_ids = [did for did in sorted(docs.keys()) if did not in done]
    log(f"docs to process this run: {len(pending_ids)} (skipped: {len(docs) - len(pending_ids)})")

    for did in pending_ids:
        if args.max_docs and new_this_run >= args.max_docs:
            log(f"Reached --max-docs={args.max_docs}; stopping")
            break
        d = docs[did]
        text = d["content"]
        if not text or not text.strip():
            stats["docs_missing_text"] += 1
            continue
        try:
            rows = build_doc(did, text, d["title"], d["source_type"], v5_tok,
                             chunking=args.chunking)
        except Exception as e:
            stats["docs_error"] += 1
            log(f"  ERROR {did}: {e}")
            continue
        if rows is None:
            stats["docs_missing_colbert"] += 1
            continue

        write_q.put(rows)
        n_leaves = sum(1 for r in rows if r["is_leaf"])
        stats["docs_processed"] += 1
        stats["nodes_written"] += len(rows)
        stats["leaves_written"] += n_leaves
        stats["max_level_seen"] = max(stats["max_level_seen"], max(r["level"] for r in rows))
        stats["max_leaf_tokens"] = max(
            stats["max_leaf_tokens"],
            max((r["n_tokens_colbert"] for r in rows if r["is_leaf"]), default=0),
        )
        new_this_run += 1

        now = time.time()
        if now - last_log >= 15:
            rate = stats["docs_processed"] / max(now - start, 1e-3)
            remaining = (len(pending_ids) - stats["docs_processed"] - stats["docs_skipped"]) / max(rate, 1e-3)
            log(f"  docs {stats['docs_processed']:,}/{len(pending_ids)}  "
                f"({rate:.1f}/s)  nodes={stats['nodes_written']:,}  "
                f"leaves={stats['leaves_written']:,}  "
                f"max_level={stats['max_level_seen']}  "
                f"writeq={write_q.qsize()}  written={writer.rows_written:,}  "
                f"err={stats['docs_error']}  ETA {remaining/60:.1f}min")
            last_log = now

    log("Draining writer ...")
    write_q.put(None)
    writer.stop.set()
    writer.join()

    elapsed = time.time() - start
    log(f"DONE.  docs={stats['docs_processed']:,}  "
        f"nodes={stats['nodes_written']:,}  leaves={stats['leaves_written']:,}  "
        f"elapsed={elapsed/60:.1f}min  writer_errors={writer.errors}")
    log(f"Final table: {table.count_rows():,} rows")

    stats_path = out_dir / "build_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
