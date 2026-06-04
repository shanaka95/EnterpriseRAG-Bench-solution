#!/usr/bin/env python3
"""
ColBERT MaxSim rerank on the RRF (jina-v3 top-N, BM25 top-N) top-1000.

This is the killer combination:
  Stage 1: RRF fusion of jv5000 + bm5000 → 9029 unique candidates
  Stage 2: Take top-1000 by RRF score → 1000 candidate pool (much better
           than the 2359 flat bag union)
  Stage 3: ColBERT MaxSim rerank on those 1000 → top-K final ranking

Compare to:
  - jv500+bm2000 flat union → ColBERT rerank (90.8% at K=1000, current best)
  - RRF alone (89.4% at K=1000)
  - ColBERT alone (90.8%)

Hypothesis: the RRF top-1000 is a tighter, better-ranked candidate pool than
the flat union, so ColBERT has an easier job and should improve.
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
COLBERT_LANCE = "/data/projects/rag/data/colbert_index/db"
OUT_PER_Q  = "/data/projects/rag/data/colbert_rerank_rrf_per_question.csv"
OUT_SUMMARY = "/data/projects/rag/data/colbert_rerank_rrf_summary.csv"

K_VALUES_OUT = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 500, 750, 1000)
N_INPUT = 2000
K0 = 60
RRF_POOL = 1000  # take top-RRF_POOL of RRF ranking as ColBERT candidate pool


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_match(ranked_ids: list[str], expected_ids: list[str]) -> int | None:
    for eid in expected_ids:
        for ri, d in enumerate(ranked_ids):
            if eid in d:
                return ri + 1
    return None


def rrf_score(jv_top_N, bm_top_N, k0):
    scores: dict[str, float] = defaultdict(float)
    for rank, d in enumerate(jv_top_N, 1):
        scores[d] += 1.0 / (k0 + rank)
    for rank, d in enumerate(bm_top_N, 1):
        scores[d] += 1.0 / (k0 + rank)
    jv_rank = {d: i for i, d in enumerate(jv_top_N)}
    bm_rank = {d: i for i, d in enumerate(bm_top_N)}
    return sorted(scores.keys(), key=lambda d: (-scores[d], jv_rank.get(d, 1e9), bm_rank.get(d, 1e9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--n-input", type=int, default=N_INPUT,
                    help="Top-N from each of jv and bm to feed RRF")
    ap.add_argument("--k0", type=int, default=K0, help="RRF constant")
    ap.add_argument("--rrf-pool", type=int, default=RRF_POOL,
                    help="Top-K of RRF to feed ColBERT rerank")
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--out", default=OUT_PER_Q)
    ap.add_argument("--summary", default=OUT_SUMMARY)
    args = ap.parse_args()

    log(f"args: {vars(args)}")

    log("loading jv + BM25 top-K doc IDs …")
    with open(JINA_JSONL) as f:
        jv = {json.loads(l)["question_id"]: json.loads(l) for l in f}
    with open(BM25_JSONL) as f:
        bm = {json.loads(l)["question_id"]: json.loads(l) for l in f}

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

    log("loading jina-colbert-v2 reranker …")
    from app.ml.colbert_reranker import colbert_rerank
    log("  model loaded")

    # Per-question: RRF top-K → ColBERT rerank
    log(f"reranking {len(questions)} questions through RRF (N={args.n_input}, k0={args.k0}) "
        f"top-{args.rrf_pool} → ColBERT MaxSim …")
    t_total = time.time()
    per_q_rows: list[dict] = []
    for qi, q in enumerate(questions):
        qid = q["question_id"]
        query = q["question"]
        exp = q.get("expected_doc_ids", [])

        # RRF top-K
        jv_top = jv[qid][f"top_{args.n_input}_ids"]
        bm_top = bm[qid][f"top_{args.n_input}_ids"]
        rrf_ranked = rrf_score(jv_top, bm_top, args.k0)
        cands = rrf_ranked[:args.rrf_pool]

        # ColBERT rerank on cands
        t0 = time.time()
        ranked = colbert_rerank(query, cands, top_k=None, chunk_size=args.chunk_size)
        dt = time.time() - t0
        ranked_ids = [d for d, _ in ranked]

        row = {
            "question_id": qid,
            "question_type": q.get("question_type", ""),
            "source_types": "|".join(q.get("source_types", [])),
            "expected_doc_ids": "|".join(exp),
            "rrf_pool_size": len(cands),
            "rerank_returned": len(ranked_ids),
            "rerank_seconds": round(dt, 3),
        }
        for k in K_VALUES_OUT:
            top_k = ranked_ids[:k]
            rank = best_match(top_k, exp)
            row[f"hit@{k}"] = 1 if rank is not None else 0
            row[f"rank@{k}"] = rank if rank is not None else ""
        per_q_rows.append(row)

        if (qi + 1) % 25 == 0:
            elapsed = time.time() - t_total
            rate = (qi + 1) / elapsed
            eta = (len(questions) - qi - 1) / rate
            log(f"  {qi+1}/{len(questions)}  ({rate:.2f} q/s, ETA {eta/60:.1f} min)")

    total_seconds = time.time() - t_total
    log(f"  all reranks done in {total_seconds/60:.1f} min "
        f"(mean {total_seconds/len(questions):.2f} s/q)")

    log(f"writing {args.out}")
    base_cols = ["question_id", "question_type", "source_types",
                 "expected_doc_ids", "rrf_pool_size", "rerank_returned", "rerank_seconds"]
    k_cols = []
    for k in K_VALUES_OUT:
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
    for k in K_VALUES_OUT:
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
            "K": k, "hits": hits, "n": n,
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

    # Pretty print
    print()
    print("=" * 80)
    print(f"COLBERT MaxSim on RRF top-{args.rrf_pool} "
          f"(RRF: jv{args.n_input} + bm{args.n_input}, k0={args.k0})")
    print("=" * 80)
    print(f"  {'K':>4}  {'hits':>5}  {'acc':>7}  {'mean_rank':>10}  {'p50':>5}  {'p95':>5}")
    print("  " + "-" * 60)
    for s in summary:
        print(f"  {s['K']:>4}  {s['hits']:>5}  {s['hit_rate_pct']:>6.1f}%  "
              f"{str(s['mean_rank']):>10}  {str(s['p50_rank']):>5}  {str(s['p95_rank']):>5}")

    times = [float(r["rerank_seconds"]) for r in per_q_rows]
    times.sort()
    print()
    print(f"Latency per query (s):")
    print(f"  total: {total_seconds/60:.1f} min")
    print(f"  mean:  {statistics.mean(times):.2f} s")
    print(f"  p50:   {times[len(times)//2]:.2f} s")
    print(f"  p95:   {times[min(int(0.95*len(times)), len(times)-1)]:.2f} s")

    log("done.")


if __name__ == "__main__":
    main()
