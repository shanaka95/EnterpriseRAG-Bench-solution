#!/usr/bin/env python3
"""
Cross-encoder rerank on the jv500+bm2000 union, evaluated at multiple K values.

This is the same experiment structure as the ColBERT MaxSim rerank, but using
a transformer cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 or
BAAI/bge-reranker-base) for scoring.

Per (query, doc) pair the cross-encoder outputs a single relevance logit,
which is then used to rank the candidates.

Pipeline:
  Stage 1 (cached): jv500 + bm2000 union (mean 2359 cands/query)
  Stage 2: cross-encoder scores (query, doc) for every candidate
  Stage 3: take top-K at the specified K values, evaluate hit@K

K values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 500, 750, 1000
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

JINA_JSONL = "/data/projects/rag/data/jina_v3_topk_docids.jsonl"
BM25_JSONL = "/data/projects/rag/data/bm25_topk_docids.jsonl"
QUESTIONS  = "/data/projects/rag/data/questions.jsonl"
CORPUS_DIR = "/data/projects/rag/data/all_documents"
OUT_PER_Q  = "/data/projects/rag/data/cross_encoder_rerank_per_question.csv"
OUT_SUMMARY = "/data/projects/rag/data/cross_encoder_rerank_summary.csv"

K_VALUES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 500, 750, 1000)
JV_K, BM_K = 500, 2000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_match(ranked_ids: list[str], expected_ids: list[str]) -> int | None:
    for eid in expected_ids:
        for ri, d in enumerate(ranked_ids):
            if eid in d:
                return ri + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of candidates per query (debug).")
    ap.add_argument("--out", default=OUT_PER_Q)
    ap.add_argument("--summary", default=OUT_SUMMARY)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU
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

    # Pre-compute unions + load doc text
    log(f"pre-computing jv{JV_K}+bm{BM_K} unions and reading doc text …")
    unions: dict[str, list[str]] = {}
    doc_text_cache: dict[str, str] = {}
    for q in questions:
        qid = q["question_id"]
        union = list(dict.fromkeys(jv[qid][f"top_{JV_K}_ids"] + bm[qid][f"top_{BM_K}_ids"]))
        if args.limit:
            union = union[:args.limit]
        unions[qid] = union
        # Read the doc text for each unique doc in the union
        for did in union:
            if did not in doc_text_cache:
                fp = os.path.join(CORPUS_DIR, did)
                try:
                    with open(fp, encoding="utf-8", errors="replace") as fh:
                        doc_text_cache[did] = fh.read()
                except FileNotFoundError:
                    doc_text_cache[did] = ""
    mean_sz = sum(len(u) for u in unions.values()) / len(unions)
    log(f"  union sizes: mean={mean_sz:.0f}  unique docs read: {len(doc_text_cache)}")

    # Load cross-encoder
    log(f"loading cross-encoder: {args.model} …")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(args.model, max_length=args.max_length, device="cpu")
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

        # Build (query, doc_text) pairs
        pairs = [[query, doc_text_cache.get(d, "")] for d in cands]

        t0 = time.time()
        scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
        dt = time.time() - t0

        # Rank by score desc
        order = sorted(range(len(cands)), key=lambda i: -float(scores[i]))
        ranked_ids = [cands[i] for i in order]

        row = {
            "question_id": qid,
            "question_type": q.get("question_type", ""),
            "source_types": "|".join(q.get("source_types", [])),
            "expected_doc_ids": "|".join(exp),
            "union_size": len(cands),
            "rerank_seconds": round(dt, 3),
        }
        for k in K_VALUES:
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

    # Write per-question CSV
    log(f"writing {args.out}")
    base_cols = ["question_id", "question_type", "source_types",
                 "expected_doc_ids", "union_size", "rerank_seconds"]
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
    log(f"  wrote {len(summary)} summary rows")

    # Pretty print
    print()
    print("=" * 80)
    print(f"CROSS-ENCODER ({args.model}) RERANK on jv{JV_K}+bm{BM_K} union "
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
