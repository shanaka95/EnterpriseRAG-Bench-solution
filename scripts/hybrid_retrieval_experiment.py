#!/usr/bin/env python3
"""
Hybrid retrieval experiment — 5×5 = 25 combinations of (jina-v3 K, BM25 K).

For each of the 500 questions:
  For each (jv_k, bm_k) in {100,500,1000,2000,5000} × {100,500,1000,2000,5000}:
    union_ids = dedupe(set(jv_top_k) | set(bm_top_k))   # preserves jina-v3 order, then BM25
    hit = 1 if any expected_doc_id appears as a substring of any union id, else 0
    union_size = |union_ids|

Then aggregate over questions to produce:
  - Per-question CSV: question_id, source, expected_doc_ids,
    and 25 columns of "hit_jv{K}_bm{K}" + 25 columns of "union_jv{K}_bm{K}"
  - Summary CSV: 5×5 hit-rate matrix + 5×5 mean-union-size matrix

Outputs:
  - /data/projects/rag/data/hybrid_retrieval_per_question.csv
  - /data/projects/rag/data/hybrid_retrieval_summary.csv
  - Stdout: pretty 5×5 hit-rate matrix and union-size matrix
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

JINA_JSONL = "/data/projects/rag/data/jina_v3_topk_docids.jsonl"
BM25_JSONL = "/data/projects/rag/data/bm25_topk_docids.jsonl"
OUT_PER_Q  = "/data/projects/rag/data/hybrid_retrieval_per_question.csv"
OUT_SUMMARY = "/data/projects/rag/data/hybrid_retrieval_summary.csv"

K_VALUES = (100, 500, 1000, 2000, 5000)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_match(ranked_ids: list[str], expected_ids: list[str]) -> int | None:
    """Return 1-based rank of the first expected_id found in ranked_ids, or None."""
    for eid in expected_ids:
        for ri, doc_path in enumerate(ranked_ids):
            if eid in doc_path:
                return ri + 1
    return None


def main():
    log("loading jina-v3 + BM25 top-K doc IDs …")
    with open(JINA_JSONL) as f:
        jv_rows = [json.loads(l) for l in f]
    with open(BM25_JSONL) as f:
        bm_rows = [json.loads(l) for l in f]
    if len(jv_rows) != len(bm_rows):
        sys.exit(f"row count mismatch: jina={len(jv_rows)}, bm25={len(bm_rows)}")
    log(f"  loaded {len(jv_rows)} rows from each")
    jv_by_id = {r["question_id"]: r for r in jv_rows}
    bm_by_id = {r["question_id"]: r for r in bm_rows}
    question_ids = [r["question_id"] for r in jv_rows]

    # Compute hybrid unions and hits per question
    log("computing 5×5 hybrid unions + hits …")
    t0 = time.time()

    # Pre-extract the jv and bm top-K lists per question for fast access
    jv_topk = {qid: {k: jv_by_id[qid][f"top_{k}_ids"] for k in K_VALUES} for qid in question_ids}
    bm_topk = {qid: {k: bm_by_id[qid][f"top_{k}_ids"] for k in K_VALUES} for qid in question_ids}

    # Per-question results
    per_q_rows: list[dict] = []
    # Aggregators
    hit_sums: dict[tuple[int, int], int] = {}      # (jv_k, bm_k) → # of hits
    union_sums: dict[tuple[int, int], int] = {}    # (jv_k, bm_k) → sum of union sizes

    for qid in question_ids:
        q = jv_by_id[qid]
        exp_ids = q.get("expected_doc_ids", [])
        row = {
            "question_id": qid,
            "question_type": q.get("question_type", ""),
            "source_types": "|".join(q.get("source_types", [])),
            "expected_doc_ids": "|".join(exp_ids),
        }
        for jv_k in K_VALUES:
            for bm_k in K_VALUES:
                # Dedupe-union: jv first (preserves jina-v3 order for ties), then bm-only items appended
                jv_list = jv_topk[qid][jv_k]
                bm_list = bm_topk[qid][bm_k]
                seen: set[str] = set()
                union: list[str] = []
                for d in jv_list:
                    if d not in seen:
                        seen.add(d); union.append(d)
                for d in bm_list:
                    if d not in seen:
                        seen.add(d); union.append(d)
                union_size = len(union)
                hit = 1 if best_match(union, exp_ids) is not None else 0
                col_hit = f"hit_jv{jv_k}_bm{bm_k}"
                col_sz  = f"size_jv{jv_k}_bm{bm_k}"
                row[col_hit] = hit
                row[col_sz]  = union_size
                hit_sums[(jv_k, bm_k)]   = hit_sums.get((jv_k, bm_k), 0) + hit
                union_sums[(jv_k, bm_k)] = union_sums.get((jv_k, bm_k), 0) + union_size
        per_q_rows.append(row)

    log(f"  done in {time.time()-t0:.0f}s")

    # Write per-question CSV
    log(f"writing {OUT_PER_Q}")
    base_cols = ["question_id", "question_type", "source_types", "expected_doc_ids"]
    combo_cols = []
    for jv_k in K_VALUES:
        for bm_k in K_VALUES:
            combo_cols.append(f"hit_jv{jv_k}_bm{bm_k}")
            combo_cols.append(f"size_jv{jv_k}_bm{bm_k}")
    fieldnames = base_cols + combo_cols
    with open(OUT_PER_Q, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_q_rows)
    log(f"  wrote {len(per_q_rows)} rows  ({os.path.getsize(OUT_PER_Q)/1e6:.1f} MB)")

    # Build and write summary CSV
    n = len(question_ids)
    summary_rows: list[dict] = []
    # Hit-rate matrix
    summary_rows.append({"metric": "hit_rate_pct", "axis": "BM25_K→"})
    for jv_k in K_VALUES:
        row = {"metric": "hit_rate_pct", "axis": f"jina-v3 K={jv_k}"}
        for bm_k in K_VALUES:
            rate = 100 * hit_sums[(jv_k, bm_k)] / n
            row[f"bm{bm_k}"] = round(rate, 2)
        summary_rows.append(row)
    # Union-size matrix
    summary_rows.append({"metric": "mean_union_size", "axis": "BM25_K→"})
    for jv_k in K_VALUES:
        row = {"metric": "mean_union_size", "axis": f"jina-v3 K={jv_k}"}
        for bm_k in K_VALUES:
            mean_sz = union_sums[(jv_k, bm_k)] / n
            row[f"bm{bm_k}"] = round(mean_sz, 1)
        summary_rows.append(row)

    log(f"writing {OUT_SUMMARY}")
    with open(OUT_SUMMARY, "w", newline="") as f:
        cols = ["metric", "axis"] + [f"bm{bm_k}" for bm_k in K_VALUES]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)
    log(f"  wrote {len(summary_rows)} summary rows")

    # Print pretty tables
    print()
    print("=" * 90)
    print("HYBRID RETRIEVAL EXPERIMENT — 5×5 (jina-v3 K rows × BM25 K columns)")
    print("=" * 90)

    print()
    print("Hit-rate matrix (% of 500 questions where expected doc appears in deduped union):")
    header = f"{'jina-v3 K \\\\ BM25 K':<18}" + "".join(f"{f'bm{k}':>10}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for jv_k in K_VALUES:
        cells = "".join(f"{100*hit_sums[(jv_k, bm_k)]/n:>9.1f}%" for bm_k in K_VALUES)
        print(f"{f'jv{jv_k}':<18}{cells}")

    print()
    print("Mean union size (deduped):")
    header = f"{'jina-v3 K \\\\ BM25 K':<18}" + "".join(f"{f'bm{k}':>10}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for jv_k in K_VALUES:
        cells = "".join(f"{union_sums[(jv_k, bm_k)]/n:>10.1f}" for bm_k in K_VALUES)
        print(f"{f'jv{jv_k}':<18}{cells}")

    # Reference rows: standalone jina-v3 and BM25 (K = jv_k only, K = bm_k only)
    print()
    print("Reference (single-source):")
    print(f"  {'jina-v3 alone:':<22} " + " ".join(
        f"@jv{k}={100*hit_sums[(k, 0)] if (k, 0) in hit_sums else '-':>6}"
        for k in K_VALUES
    ))
    # jina-v3 alone = jv_k + bm_k=0 (no BM25) →  union is just jv_k; use jv_rows
    jv_alone = {k: 0 for k in K_VALUES}
    bm_alone = {k: 0 for k in K_VALUES}
    for r in jv_rows:
        for k in K_VALUES:
            if r[f"hit_at_{k}"] == 1:
                jv_alone[k] += 1
    for r in bm_rows:
        for k in K_VALUES:
            if r[f"hit_at_{k}"] == 1:
                bm_alone[k] += 1
    print(f"  {'jina-v3 alone hits:':<22} " + " ".join(
        f"@jv{k}={100*jv_alone[k]/n:>5.1f}%" for k in K_VALUES
    ))
    print(f"  {'BM25 alone hits:':<22} " + " ".join(
        f"@bm{k}={100*bm_alone[k]/n:>5.1f}%" for k in K_VALUES
    ))

    # Highlight the best combination
    best = max(hit_sums.items(), key=lambda kv: kv[1])
    print()
    print(f"Best combination: jv={best[0][0]} + bm={best[0][1]} → "
          f"{best[1]}/500 = {100*best[1]/n:.2f}% hit rate "
          f"(mean union size = {union_sums[best[0]]/n:.1f})")

    log("done.")


if __name__ == "__main__":
    main()
