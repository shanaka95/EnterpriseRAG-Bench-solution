#!/usr/bin/env python3
"""
BM25 retrieval for all 500 questions.

Builds a BM25 index over the 511,962 documents under
/data/projects/rag/data/all_documents/ and retrieves top-100, 500, 1000,
2000, and 5000 doc IDs per question. Saves them in a JSONL file
mirroring the jina-v3 baseline format, plus a per-question evaluation
CSV.

Doc IDs use the same relative-path format as the LanceDB jina-v3 index
(e.g. "github/dsid_xxx__filename.txt") so jina-v3 and BM25 results can
be cross-compared one-for-one.

Approach:
  - Tokenizer: bm25s.tokenize with English stopwords, lowercased,
    punctuation-stripped, no stemming (the corpus is noisy multi-source
    text and stemming gave negligible gain on a quick pilot).
  - BM25: bm25s.BM25 with default Lucene method (k1=1.5, b=0.75).
  - Memory: corpus is 2.47 GB raw text. We stream-read files, build
    a list of (id, text), tokenize, then drop the raw text before
    scoring. bm25s uses scipy sparse matrices for the inverted index,
    so peak RAM stays well under the raw corpus size.

Outputs:
  - /data/projects/rag/data/bm25_topk_docids.jsonl  (one row per question)
  - /data/projects/rag/data/bm25_topk_evaluation.csv  (per-question hit/rank)
  - Stdout summary table comparing BM25 vs jina-v3 hit@K.
"""
import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import bm25s

# Stemmer is optional (provided by PyStemmer, installed with bm25s[core]).
try:
    import Stemmer  # type: ignore
    _HAS_STEMMER = True
except ImportError:
    _HAS_STEMMER = False

CORPUS_DIR  = "/data/projects/rag/data/all_documents"
QUESTIONS_PATH = "/data/projects/rag/data/questions.jsonl"
OUT_JSONL   = "/data/projects/rag/data/bm25_topk_docids.jsonl"
OUT_CSV     = "/data/projects/rag/data/bm25_topk_evaluation.csv"

K_VALUES = (100, 500, 1000, 2000, 5000)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_questions(path: str, n: int) -> list[dict]:
    out = []
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            out.append(json.loads(ln.strip()))
            if len(out) >= n:
                break
    return out


def discover_corpus(root: str) -> tuple[list[str], list[str]]:
    """Walk the corpus directory. Returns (ids, texts) — relative paths and raw content."""
    log(f"walking {root} …")
    ids: list[str] = []
    texts: list[str] = []
    root_path = Path(root)
    n = 0
    t0 = time.time()
    for fp in sorted(root_path.rglob("*.txt")):
        rel = fp.relative_to(root_path).as_posix()  # e.g. "github/dsid_xxx__file.txt"
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log(f"  failed to read {rel}: {e}")
            continue
        ids.append(rel)
        texts.append(txt)
        n += 1
        if n % 50000 == 0:
            log(f"  read {n:,} files  (elapsed {time.time()-t0:.0f}s)")
    log(f"  read {n:,} files total in {time.time()-t0:.0f}s")
    return ids, texts


def best_match_rank(ranked_ids: list[str], expected_ids: list[str]) -> int | None:
    for eid in expected_ids:
        for ri, doc_path in enumerate(ranked_ids):
            if eid in doc_path:
                return ri + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--out", default=OUT_JSONL)
    ap.add_argument("--csv", default=OUT_CSV)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b",  type=float, default=0.75)
    ap.add_argument("--method", default="lucene",
                    choices=["lucene", "robertson", "atire", "bm25l", "bm25+"])
    ap.add_argument("--stem", action="store_true",
                    help="Use English stemming (slower indexing, marginal retrieval gain).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optionally cap the corpus (for debugging).")
    args = ap.parse_args()

    log(f"args: {vars(args)}")

    questions = load_questions(args.questions, args.num)
    log(f"loaded {len(questions)} questions")

    # 1. Discover + read corpus
    ids, texts = discover_corpus(CORPUS_DIR)
    if args.limit:
        ids = ids[:args.limit]
        texts = texts[:args.limit]
        log(f"  capped corpus to {len(ids):,} docs (debug)")
    n_docs = len(ids)
    log(f"corpus: {n_docs:,} docs, total raw text: {sum(len(t) for t in texts)/1e9:.2f} GB")

    # 2. Tokenize
    log("tokenizing corpus …")
    t0 = time.time()
    stemmer = None
    if args.stem:
        if not _HAS_STEMMER:
            log("  --stem requested but PyStemmer not installed; proceeding without stemming")
        else:
            stemmer = Stemmer.Stemmer("english")
    stopwords = "en"
    corpus_tokens = bm25s.tokenize(
        texts,
        stopwords=stopwords,
        stemmer=stemmer,
        show_progress=True,
    )
    n_tokens = sum(len(t) for t in corpus_tokens.ids)
    log(f"  tokenized in {time.time()-t0:.0f}s  ({n_tokens:,} tokens, {n_docs:,} docs)")

    # Free raw text — tokenized form is what we need from here on
    del texts
    gc.collect()

    # 3. Build BM25 index
    log(f"building BM25 index  (method={args.method}, k1={args.k1}, b={args.b}) …")
    t0 = time.time()
    bm25 = bm25s.BM25(method=args.method, k1=args.k1, b=args.b)
    bm25.index(corpus_tokens, show_progress=True)
    log(f"  index built in {time.time()-t0:.0f}s")

    # 4. Tokenize queries
    log("tokenizing queries …")
    query_texts = [q["question"] for q in questions]
    query_tokens = bm25s.tokenize(
        query_texts,
        stopwords=stopwords,
        stemmer=stemmer,
        show_progress=False,
    )

    # 5. Retrieve top-K for ALL questions in one batched call
    # bm25s.retrieve returns a Results namedtuple (.documents, .scores) with
    # shape (n_queries, k). Pass corpus=ids to get the actual doc paths back.
    log(f"querying {len(questions)} questions × top-{max(K_VALUES)} …")
    t0 = time.time()
    results = bm25.retrieve(
        query_tokens, corpus=ids, k=max(K_VALUES), show_progress=True,
    )
    log(f"  all queries done in {time.time()-t0:.0f}s")

    docs_matrix = results.documents  # shape: (n_queries, k), each cell is a doc id string
    out_records: list[dict] = []
    csv_rows: list[dict] = []
    for qi, q in enumerate(questions):
        ranked_ids = [str(d) for d in docs_matrix[qi]]

        qid = q.get("question_id", "")
        exp_ids = q.get("expected_doc_ids", [])
        rec = {
            "question_id": qid,
            "question_type": q.get("question_type", ""),
            "source_types": q.get("source_types", []),
            "expected_doc_ids": exp_ids,
        }
        csv_row = {
            "question_id": qid,
            "question_type": q.get("question_type", ""),
            "source_types": "|".join(q.get("source_types", [])),
            "expected_doc_ids": "|".join(exp_ids),
        }
        for k in K_VALUES:
            top_k = ranked_ids[:k]
            rec[f"top_{k}_ids"] = top_k
            rank = best_match_rank(top_k, exp_ids)
            rec[f"hit_at_{k}"] = 1 if rank is not None else 0
            rec[f"rank_at_{k}"] = rank
            csv_row[f"hit@{k}"] = 1 if rank is not None else 0
            csv_row[f"rank@{k}"] = rank if rank is not None else ""
        out_records.append(rec)
        csv_rows.append(csv_row)

    # 6. Write JSONL
    log(f"writing {args.out}")
    with open(args.out, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
    log(f"  wrote {len(out_records)} records  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    # 7. Write CSV
    log(f"writing {args.csv}")
    fieldnames = ["question_id", "question_type", "source_types", "expected_doc_ids"]
    for k in K_VALUES:
        fieldnames.extend([f"hit@{k}", f"rank@{k}"])
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    log(f"  wrote {len(csv_rows)} rows")

    # 8. Summary
    print()
    print("=" * 70)
    print("BM25 RETRIEVAL — SUMMARY")
    print("=" * 70)
    print(f"{'k':>6}  {'hits':>6}  {'acc':>7}  {'missed':>6}")
    for k in K_VALUES:
        hits = sum(rec[f"hit_at_{k}"] for rec in out_records)
        acc = 100 * hits / len(out_records)
        print(f"{k:>6}  {hits:>6}  {acc:>6.1f}%  {len(out_records)-hits:>6}")

    # Per-source accuracy at hit_at_1000
    print()
    print("Per-source accuracy at hit_at_1000:")
    by_source: dict[str, list[int]] = {}
    for rec in out_records:
        if len(rec["source_types"]) == 1:
            src = rec["source_types"][0]
            by_source.setdefault(src, []).append(rec["hit_at_1000"])
    for src, hits in sorted(by_source.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        h = sum(hits)
        n = len(hits)
        print(f"  {src:<14}  {h:>4}/{n:<4}  {100*h/n:>6.1f}%")

    # Best rank distribution among hits at 5000
    hits_5k = [rec for rec in out_records if rec["hit_at_5000"] == 1]
    if hits_5k:
        print()
        print("Best rank distribution (among hits at top-5000):")
        buckets = [(1, 1, "Rank 1"), (2, 10, "Rank 2-10"), (11, 100, "Rank 11-100"),
                   (101, 500, "Rank 101-500"), (501, 1000, "Rank 501-1000"),
                   (1001, 2000, "Rank 1001-2000"), (2001, 5000, "Rank 2001-5000")]
        for lo, hi, label in buckets:
            n = sum(1 for rec in hits_5k if lo <= rec["rank_at_5000"] <= hi)
            print(f"  {label:<22}  {n:>4}  ({100*n/len(hits_5k):>5.1f}%)")

    log("done.")


if __name__ == "__main__":
    main()
