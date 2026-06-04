#!/usr/bin/env python3
"""
ColBERT MaxSim rerank on the jv500+bm2000 union, evaluated at multiple K values.

Pipeline:
  Stage 1 (cached): take jina-v3 top-500 + BM25 top-2000 → dedupe-union
                    → ~2359 candidates per question (mean).
  Stage 2: run jina-colbert-v2 MaxSim rerank on the union (top_k=None to
           get the full ranking).
  Evaluate hit@K for K ∈ {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150,
                            180, 200, 500, 750, 1000}.

Per-question CSV:
  question_id, source, expected_doc_ids,
  union_size, rerank_seconds,
  hit@K + rank@K for each of the 17 K values.

Summary CSV:
  K, hit_rate, mean_rank, p50_rank, p95_rank, total_seconds, mean_seconds_per_q.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/data/projects/rag/backend")

JINA_JSONL = "/data/projects/rag/data/jina_v3_topk_docids.jsonl"
BM25_JSONL = "/data/projects/rag/data/bm25_topk_docids.jsonl"
QUESTIONS  = "/data/projects/rag/data/questions.jsonl"
OUT_PER_Q  = "/data/projects/rag/data/colbert_rerank_jv500_bm2000_per_question.csv"
OUT_SUMMARY = "/data/projects/rag/data/colbert_rerank_jv500_bm2000_summary.csv"

K_VALUES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 500, 750, 1000)
JV_K, BM_K = 500, 2000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_match(ranked_ids: list[str], expected_ids: list[str]) -> int | None:
    """Return 1-based rank of first expected id found in ranked_ids, or None."""
    for eid in expected_ids:
        for ri, d in enumerate(ranked_ids):
            if eid in d:
                return ri + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--chunk-size", type=int, default=100,
                    help="ColBERT MaxSim chunk size (# docs per einsum).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of candidates per query (debug).")
    ap.add_argument("--out", default=OUT_PER_Q)
    ap.add_argument("--summary", default=OUT_SUMMARY)
    args = ap.parse_args()

    log(f"args: {vars(args)}")

    # Load jv + bm25 top-K doc IDs
    log("loading jv + BM25 top-K doc IDs …")
    with open(JINA_JSONL) as f:
        jv = {json.loads(l)["question_id"]: json.loads(l) for l in f}
    with open(BM25_JSONL) as f:
        bm = {json.loads(l)["question_id"]: json.loads(l) for l in f}

    # Load questions
    questions = []
    with open(args.questions) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            if d.get("question_id") in jv:
                questions.append(d)
                if len(questions) >= args.num:
                    break
    log(f"  loaded {len(questions)} questions")

    # Pre-compute unions
    log(f"pre-computing jv{JV_K}+bm{BM_K} unions …")
    unions: dict[str, list[str]] = {}
    for q in questions:
        qid = q["question_id"]
        union = list(dict.fromkeys(jv[qid][f"top_{JV_K}_ids"] + bm[qid][f"top_{BM_K}_ids"]))
        if args.limit:
            union = union[:args.limit]
        unions[qid] = union
    mean_sz = sum(len(u) for u in unions.values()) / len(unions)
    log(f"  union sizes: mean={mean_sz:.0f}  min={min(len(u) for u in unions.values())}  "
        f"max={max(len(u) for u in unions.values())}")

    # Load ColBERT model
    log("loading jina-colbert-v2 reranker …")
    from app.ml.colbert_reranker import colbert_rerank
    log("  model loaded")

    # Per-question rerank
    log(f"reranking {len(questions)} questions × ~{mean_sz:.0f} cands each …")
    t_total = time.time()
    per_q_rows: list[dict] = []
    for qi, q in enumerate(questions):
        qid = q["question_id"]
        query = q["question"]
        cands = unions[qid]
        exp = q.get("expected_doc_ids", [])

        t0 = time.time()
        ranked = colbert_rerank(query, cands, top_k=None, chunk_size=args.chunk_size)
        dt = time.time() - t0

        ranked_ids = [d for d, _ in ranked]
        row = {
            "question_id": qid,
            "question_type": q.get("question_type", ""),
            "source_types": "|".join(q.get("source_types", [])),
            "expected_doc_ids": "|".join(exp),
            "union_size": len(cands),
            "rerank_returned": len(ranked_ids),
            "rerank_seconds": round(dt, 3),
        }
        for k in K_VALUES:
            top_k = ranked_ids[:k]
            rank = best_match(top_k, exp)
            row[f"hit@{k}"] = 1 if rank is not None else 0
            row[f"rank@{k}"] = rank if rank is not None else ""
        per_q_rows.append(row)

        if (qi + 1) % 10 == 0:
            elapsed = time.time() - t_total
            rate = (qi + 1) / elapsed
            eta = (len(questions) - qi - 1) / rate
            log(f"  {qi+1}/{len(questions)}  ({rate:.2f} q/s, ETA {eta/60:.1f} min)")

    total_seconds = time.time() - t_total
    log(f"  all reranks done in {total_seconds/60:.1f} min "
        f"(mean {total_seconds/len(questions):.2f} s/q)")

    # Write per-question CSV
    log(f"writing {args.out}")
    base_cols = ["question_id", "question_type", "source_types",
                 "expected_doc_ids", "union_size", "rerank_returned",
                 "rerank_seconds"]
    k_cols = []
    for k in K_VALUES:
        k_cols.extend([f"hit@{k}", f"rank@{k}"])
    fieldnames = base_cols + k_cols
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_q_rows)
    log(f"  wrote {len(per_q_rows)} rows  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    # Summary
    n = len(per_q_rows)
    summary = []
    for k in K_VALUES:
        hits = sum(int(r[f"hit@{k}"]) for r in per_q_rows)
        ranks = [int(r[f"rank@{k}"]) for r in per_q_rows if r[f"rank@{k}"] != ""]
        if ranks:
            ranks_sorted = sorted(ranks)
            p50 = ranks_sorted[len(ranks_sorted) // 2]
            p95 = ranks_sorted[min(int(0.95 * len(ranks_sorted)), len(ranks_sorted) - 1)]
            mean_rank = sum(ranks) / len(ranks)
        else:
            p50 = p95 = mean_rank = None
        summary.append({
            "K": k,
            "hits": hits,
            "n": n,
            "hit_rate_pct": round(100 * hits / n, 2),
            "mean_rank": round(mean_rank, 2) if mean_rank is not None else "",
            "p50_rank": p50 if p50 is not None else "",
            "p95_rank": p95 if p95 is not None else "",
            "n_hits_with_rank": len(ranks),
        })

    log(f"writing {args.summary}")
    with open(args.summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    log(f"  wrote {len(summary)} summary rows")

    # Pretty print
    print()
    print("=" * 80)
    print(f"COLBERT MaxSim RERANK on jv{JV_K}+bm{BM_K} union "
          f"(mean {mean_sz:.0f} cands/query)")
    print("=" * 80)
    print(f"  {'K':>4}  {'hits':>5}  {'acc':>7}  {'mean_rank':>10}  {'p50':>5}  {'p95':>5}")
    print("  " + "-" * 60)
    for s in summary:
        print(f"  {s['K']:>4}  {s['hits']:>5}  {s['hit_rate_pct']:>6.1f}%  "
              f"{str(s['mean_rank']):>10}  {str(s['p50_rank']):>5}  {str(s['p95_rank']):>5}")

    # Latency
    times = [float(r["rerank_seconds"]) for r in per_q_rows]
    times.sort()
    print()
    print(f"Latency per query (s):")
    print(f"  total: {total_seconds/60:.1f} min")
    print(f"  mean:  {statistics.mean(times):.2f} s")
    print(f"  p50:   {times[len(times)//2]:.2f} s")
    print(f"  p95:   {times[min(int(0.95*len(times)), len(times)-1)]:.2f} s")
    print(f"  max:   {max(times):.2f} s")

    log("done.")


if __name__ == "__main__":
    main()
