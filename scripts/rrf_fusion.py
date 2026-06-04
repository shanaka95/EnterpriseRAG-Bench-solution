#!/usr/bin/env python3
"""
Reciprocal Rank Fusion (RRF) over jina-v3 top-N and BM25 top-N.

For each question:
  For each doc d in (jv_top_N ∪ bm_top_N):
    rrf_score(d) = Σ 1/(k0 + rank_r(d))    over retrievers r that include d

Then sort by rrf_score desc → take top-K → evaluate hit@K.

Reference: Cormack, Clarke, Buettcher, "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods", SIGIR 2009.
  https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

Sweeps k0 ∈ {10, 30, 60, 100} and N ∈ {500, 1000, 2000, 5000}.
Output: per-question CSV (per config) + per-config summary + per-K hit rate.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

JINA_JSONL = "/data/projects/rag/data/jina_v3_topk_docids.jsonl"
BM25_JSONL = "/data/projects/rag/data/bm25_topk_docids.jsonl"
QUESTIONS  = "/data/projects/rag/data/questions.jsonl"
OUT_DIR    = "/data/projects/rag/data"

K_VALUES_OUT = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 500, 750, 1000)
N_VALUES = (500, 1000, 2000, 5000)
K0_VALUES = (10, 30, 60, 100)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_match(ranked_ids: list[str], expected_ids: list[str]) -> int | None:
    for eid in expected_ids:
        for ri, d in enumerate(ranked_ids):
            if eid in d:
                return ri + 1
    return None


def rrf_score(jv_top_N: list[str], bm_top_N: list[str], k0: int) -> list[str]:
    """Return RRF-ranked list of doc IDs, desc by RRF score."""
    scores: dict[str, float] = defaultdict(float)
    for rank, d in enumerate(jv_top_N, start=1):
        scores[d] += 1.0 / (k0 + rank)
    for rank, d in enumerate(bm_top_N, start=1):
        scores[d] += 1.0 / (k0 + rank)
    # Sort by score desc, tie-break by jv rank then bm rank
    jv_rank = {d: i for i, d in enumerate(jv_top_N)}
    bm_rank = {d: i for i, d in enumerate(bm_top_N)}
    return sorted(scores.keys(), key=lambda d: (-scores[d], jv_rank.get(d, 1e9), bm_rank.get(d, 1e9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--out-prefix", default="rrf_fusion")
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

    # Pre-extract top-N for each N
    log("pre-extracting top-N lists per N value …")
    jv_top = {N: {q["question_id"]: jv[q["question_id"]][f"top_{N}_ids"] for q in questions} for N in N_VALUES}
    bm_top = {N: {q["question_id"]: bm[q["question_id"]][f"top_{N}_ids"] for q in questions} for N in N_VALUES}

    # For each (N, k0) config, compute per-question RRF ranking + evaluate
    per_config_results: dict[tuple[int, int], list[dict]] = {}

    for N in N_VALUES:
        for k0 in K0_VALUES:
            log(f"computing RRF for N={N}, k0={k0} …")
            t0 = time.time()
            per_q_rows = []
            for q in questions:
                qid = q["question_id"]
                exp = q.get("expected_doc_ids", [])
                ranked = rrf_score(jv_top[N][qid], bm_top[N][qid], k0)
                row = {
                    "question_id": qid,
                    "question_type": q.get("question_type", ""),
                    "source_types": "|".join(q.get("source_types", [])),
                    "expected_doc_ids": "|".join(exp),
                    "n_unique_cands": len(set(jv_top[N][qid] + bm_top[N][qid])),
                    "rrf_pool_size": len(ranked),
                }
                for k in K_VALUES_OUT:
                    top_k = ranked[:k]
                    rank = best_match(top_k, exp)
                    row[f"hit@{k}"] = 1 if rank is not None else 0
                    row[f"rank@{k}"] = rank if rank is not None else ""
                per_q_rows.append(row)
            per_config_results[(N, k0)] = per_q_rows
            max_pool = max(int(r["rrf_pool_size"]) for r in per_q_rows)
            log(f"  done in {time.time()-t0:.1f}s  (max pool = {max_pool})")

    # Save per-question CSVs (one per config)
    log("writing per-question CSVs …")
    for (N, k0), rows in per_config_results.items():
        path = os.path.join(OUT_DIR, f"{args.out_prefix}_N{N}_k0{k0}_per_question.csv")
        cols = ["question_id", "question_type", "source_types", "expected_doc_ids",
                "n_unique_cands", "rrf_pool_size"]
        for k in K_VALUES_OUT:
            cols.extend([f"hit@{k}", f"rank@{k}"])
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        log(f"  wrote {path}")

    # Summary
    log("writing summary …")
    summary_rows = []
    for N in N_VALUES:
        for k0 in K0_VALUES:
            rows = per_config_results[(N, k0)]
            n = len(rows)
            row = {"N": N, "k0": k0, "n": n}
            for k in K_VALUES_OUT:
                hits = sum(int(r[f"hit@{k}"]) for r in rows)
                row[f"hit@{k}"] = round(100 * hits / n, 2)
            row["mean_pool_size"] = round(sum(int(r["rrf_pool_size"]) for r in rows) / n, 1)
            summary_rows.append(row)

    summary_path = os.path.join(OUT_DIR, f"{args.out_prefix}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    log(f"  wrote {summary_path}")

    # Pretty print: hit@K matrix for each (N, k0) config
    print()
    print("=" * 95)
    print("RRF FUSION — hit_rate at selected K values for each (N, k0) config")
    print("=" * 95)
    K_show = (20, 50, 100, 200, 500, 1000)
    for N in N_VALUES:
        print()
        print(f"--- Input depth N={N} (top-{N} from each of jina-v3 and BM25) ---")
        print(f'{"k0":>4}  ' + "  ".join(f"K={k}".rjust(7) for k in K_show) + "  pool")
        for k0 in K0_VALUES:
            row = next(r for r in summary_rows if r["N"] == N and r["k0"] == k0)
            cells = "  ".join(f"{row[f'hit@{k}']:>6.1f}%".rjust(7) for k in K_show)
            pool = row["mean_pool_size"]
            print(f"{k0:>4}  {cells}  {pool:.0f}")

    # Best per-K config
    print()
    print("=" * 95)
    print("Best (N, k0) config at each K — by hit_rate")
    print("=" * 95)
    for k in K_show:
        kk = f"hit@{k}"
        best = max(summary_rows, key=lambda r: r[kk])
        print(f"  K={k:>4}  best={best[kk]:.2f}%  (N={best['N']}, k0={best['k0']})")

    # Best per-k0 (averaged over N)
    print()
    print("Average hit_rate per k0 (averaged over N=500,1000,2000,5000):")
    for k0 in K0_VALUES:
        rows = [r for r in summary_rows if r["k0"] == k0]
        cells = "  ".join(
            f"K={kk}={sum(r[f'hit@{kk}'] for r in rows)/len(rows):>5.2f}%".rjust(15)
            for kk in K_show
        )
        print(f"  k0={k0:>3}  {cells}")

    # Best per-N (averaged over k0)
    print()
    print("Average hit_rate per N (averaged over k0=10,30,60,100):")
    for N in N_VALUES:
        rows = [r for r in summary_rows if r["N"] == N]
        cells = "  ".join(
            f"K={kk}={sum(r[f'hit@{kk}'] for r in rows)/len(rows):>5.2f}%".rjust(15)
            for kk in K_show
        )
        print(f"  N={N:>4}  {cells}")

    log("done.")


if __name__ == "__main__":
    main()
