#!/usr/bin/env python3
"""
embed_corpus.py — sequential embedder for gte-large-en-v1.5 via the OpenAI client.

Key design decisions after extensive debugging:
  * Uses the `openai` Python SDK (battle-tested connection pooling, retries,
    timeout handling) instead of raw requests/aiohttp.
  * Sequential (one HTTP call at a time) with batch=64 texts per call.
    On-server probes showed vLLM serves bs=64 in ~10ms.  Concurrent requests
    caused vLLM to hang (its internal scheduler gets overwhelmed on this
    single-GPU RTX 2060).
  * Lazy file reading: walks the directory tree and builds path lists once,
    but only reads file content in batches right before embedding.  This
    avoids the 10-second startup cost of reading 512k files into RAM.
  * Flushes to lancedb every N rows; resume-safe (skips existing ids).

Throughput math:
  512k files / 64 per batch = 8000 HTTP calls
  8000 calls × ~10ms each = ~80s pure GPU time
  + file I/O + lancedb writes ≈ 5-10 minutes total
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DEFAULT_MODEL = "gte-large-en-v1.5"
DEFAULT_DIM = 1024
MAX_CHARS = 2_000  # ~500 tokens; 3.7x faster than 4K with minimal semantic loss
SNIPPET_CHARS = 240


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--table", default="documents")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=4000)
    ap.add_argument("--log-every", type=int, default=4000)
    ap.add_argument("--max-retries", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"root does not exist: {root}")
    Path(args.db_dir).mkdir(parents=True, exist_ok=True)

    # ---- 1. Walk the tree (just paths, no file reads yet) ----
    print(f"[setup] walking {root} ...", flush=True)
    t0 = time.time()
    all_paths: List[Path] = sorted(p for p in root.rglob("*.txt") if p.is_file())
    if args.limit:
        all_paths = all_paths[:args.limit]
    print(f"[setup] {len(all_paths):,} files found in {time.time()-t0:.1f}s", flush=True)

    # ---- 2. Open lancedb, filter against existing ids ----
    import lancedb
    import pyarrow as pa

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), DEFAULT_DIM)),
        pa.field("path", pa.string()),
        pa.field("source", pa.string()),
        pa.field("name", pa.string()),
        pa.field("size", pa.int64()),
        pa.field("mtime", pa.float64()),
        pa.field("mtime_iso", pa.string()),
        pa.field("text", pa.string()),
        pa.field("snippet", pa.string()),
        pa.field("sha256", pa.string()),
        pa.field("embedding_model", pa.string()),
    ])

    db = lancedb.connect(args.db_dir)
    existing: Optional[set] = None
    table = None
    if args.table in db.list_tables():
        table = db.open_table(args.table)
        existing = set(table.to_pandas(columns=["id"])["id"].tolist())
        print(f"[setup] table already has {len(existing):,} rows; will skip",
              flush=True)

    if existing:
        todo_paths = [p for p in all_paths
                      if str(p.relative_to(root)) not in existing]
    else:
        todo_paths = all_paths
    print(f"[setup] {len(todo_paths):,} files to embed (resume-safe)", flush=True)
    if not todo_paths:
        print("[done] nothing to do", flush=True)
        return

    total = len(todo_paths)
    print(f"[setup] sequential; batch={args.batch_size}; "
          f"flush every {args.flush_every}", flush=True)

    # ---- 3. OpenAI client ----
    from openai import OpenAI
    client = OpenAI(base_url=args.url, api_key="unused")

    # ---- 4. Main loop ----
    buf: list[dict] = []
    submitted = 0
    failed = 0
    started = time.time()

    from tqdm import tqdm
    with tqdm(total=total, desc="embedding", unit="doc", mininterval=5) as bar:
        i = 0
        while i < total:
            batch_paths = todo_paths[i:i + args.batch_size]
            i += len(batch_paths)

            # Read files in this batch
            records = []
            texts = []
            for p in batch_paths:
                try:
                    st = p.stat()
                    raw = p.read_bytes()
                except OSError:
                    continue
                text = raw.decode("utf-8", errors="replace")
                if not text.strip():
                    continue
                if len(text) > MAX_CHARS:
                    text = text[:MAX_CHARS]
                snippet = (text[:SNIPPET_CHARS] + "…") if len(text) > SNIPPET_CHARS else text
                rel = p.relative_to(root)
                records.append({
                    "id": str(rel),
                    "path": str(p),
                    "source": rel.parts[0] if len(rel.parts) > 1 else "(root)",
                    "name": p.name,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                               time.localtime(st.st_mtime)),
                    "text": text,
                    "snippet": snippet,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "embedding_model": DEFAULT_MODEL,
                })
                texts.append(text)

            if not texts:
                continue

            # Embed
            vectors = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    resp = client.embeddings.create(model=args.model, input=texts)
                    vectors = [d.embedding for d in resp.data]
                    break
                except Exception as e:
                    if attempt < args.max_retries:
                        wait = min(2 ** attempt, 16)
                        tqdm.write(f"[retry {attempt}] {e!r} — waiting {wait}s",
                                   file=sys.stderr)
                        time.sleep(wait)
                    else:
                        tqdm.write(f"[err] batch of {len(texts)} failed: {e!r}",
                                   file=sys.stderr)
                        failed += len(texts)

            if vectors is None:
                continue

            for rec, vec in zip(records, vectors):
                rec["vector"] = vec
                buf.append(rec)
                submitted += 1
                bar.update(1)

            if submitted % args.log_every < args.batch_size:
                elapsed = time.time() - started
                rate = submitted / elapsed if elapsed > 0 else 0
                eta = (total - submitted) / rate if rate > 0 else 0
                tqdm.write(f"[stats] {submitted:,}/{total:,} | "
                           f"{rate:.0f} emb/s | {elapsed/60:.1f}min elapsed | "
                           f"{eta/60:.1f}min ETA | {failed} failed",
                           file=sys.stderr)

            if len(buf) >= args.flush_every:
                if table is None:
                    table = db.create_table(args.table, buf, schema=schema,
                                            mode="create")
                else:
                    table.add(buf)
                buf.clear()

    # Final flush
    if buf:
        if table is None:
            table = db.create_table(args.table, buf, schema=schema, mode="create")
        else:
            table.add(buf)

    elapsed = time.time() - started
    print(f"\n[done] embedded {submitted:,} docs in {elapsed:.1f}s "
          f"({submitted/elapsed:.1f} emb/s) — {failed} failed", flush=True)
    print(f"[done] LanceDB at: {args.db_dir}  (table: {args.table})", flush=True)


if __name__ == "__main__":
    main()
