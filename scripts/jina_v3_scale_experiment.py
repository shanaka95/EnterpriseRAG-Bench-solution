#!/usr/bin/env python3
"""
Jina-v3 scale experiment — first N questions, hit@k for k ∈ {100,500,1000,2000,5000}.

Uses jina-embeddings-v3 with task="retrieval.query", flat L2 search over the
2.0 GB LanceDB. Expected doc IDs (bare dsids) are matched as substrings of
full doc paths (e.g. "dsid_xxx" in "github/dsid_xxx__filename.txt").

Output: CSV with per-question results + summary.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/data/projects/rag/backend")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import lancedb

# ---------------------------------------------------------------------------

QUESTIONS_PATH = "/home/shanaka/Desktop/projects/rag/data/questions.jsonl"
DENSE_LANCE   = "/data/projects/rag/data/dense_index/db"
RESULTS_CSV   = "/data/projects/rag/data/jina_v3_scale_experiment.csv"

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out", default=RESULTS_CSV)
    args = ap.parse_args()

    log(f"args: {vars(args)}")
    questions = load_questions(args.questions, args.num)
    log(f"loaded {len(questions)} questions")

    # Load model
    from sentence_transformers import SentenceTransformer
    log("loading jina-embeddings-v3 …")
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, device="cpu")

    # Encode all queries in one batch
    queries = [q["question"] for q in questions]
    log(f"encoding {len(queries)} queries …")
    t0 = time.time()
    qvecs = model.encode(
        queries, task="retrieval.query", batch_size=32,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )
    log(f"  encoded in {time.time()-t0:.1f}s")

    # Open LanceDB
    table = lancedb.connect(DENSE_LANCE).open_table("documents")
    n_docs = table.count_rows()
    log(f"LanceDB: {n_docs:,} docs")

    # Search per query
    results_per_q: list[dict] = []   # [{100: [ids], 500: [ids], ...}, ...]
    log(f"searching {len(queries)} queries × {n_docs:,} docs …")
    for qi, vec in enumerate(qvecs):
        hits = table.search(vec.tolist()).limit(max(K_VALUES)).to_list()
        doc_ids = [h["id"] for h in hits]
        results_per_q.append({k: doc_ids[:k] for k in K_VALUES})
        if (qi + 1) % 10 == 0:
            log(f"  {qi+1}/{len(queries)} done")

    # Evaluate
    log("evaluating …")
    all_rows = []
    for i, q in enumerate(questions):
        exp_ids = q.get("expected_doc_ids", [])
        row = {
            "question_id": q.get("question_id", ""),
            "question_type": q.get("question_type", ""),
            "source_types": "|".join(q.get("source_types", [])),
            "question": q.get("question", ""),
            "expected_doc_ids": "|".join(exp_ids),
            "gold_answer": q.get("gold_answer", ""),
            "answer_facts": " || ".join(q.get("answer_facts", [])),
        }
        for k in K_VALUES:
            ranked = results_per_q[i][k]
            hit = False
            best_rank = None
            for eid in exp_ids:
                for ri, doc_path in enumerate(ranked):
                    if eid in doc_path:
                        hit = True
                        if best_rank is None or ri + 1 < best_rank:
                            best_rank = ri + 1
                        break
            row[f"hit@{k}"] = "1" if hit else "0"
            row[f"rank@{k}"] = str(best_rank) if best_rank else ""
            row[f"top{k}_docs"] = ";".join(ranked[:5])  # first 5 for quick inspection
        all_rows.append(row)

    # Write CSV
    log(f"writing {args.out}")
    with open(args.out, "w", newline="") as f:
        fieldnames = [
            "question_id", "question_type", "source_types",
            "question", "expected_doc_ids", "gold_answer", "answer_facts",
        ]
        for k in K_VALUES:
            fieldnames.extend([f"hit@{k}", f"rank@{k}", f"top{k}_docs"])
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    # Summary
    print()
    print("=" * 70)
    print("JINA-V3 SCALE EXPERIMENT SUMMARY")
    print("=" * 70)
    for k in K_VALUES:
        hits = sum(1 for row in all_rows if row[f"hit@{k}"] == "1")
        print(f"  hit@{k:5d}:  {hits}/{len(all_rows)}  ({100*hits/len(all_rows):.1f}%)")

    # Per-question detail (which questions hit at which k)
    print()
    print("Per-question hit progression:")
    print(f"{'qid':<12} {'q_type':<8} {'src':<15} {'@100':<5} {'@500':<5} {'@1000':<6} {'@2000':<6} {'@5000':<6} {'best_rank':<10}")
    print("-" * 75)
    for row in all_rows:
        best_rank = None
        for k in K_VALUES:
            r = row[f"rank@{k}"]
            if r:
                rnum = int(r)
                if best_rank is None or rnum < best_rank:
                    best_rank = rnum
        print(f"{row['question_id']:<12} {row['question_type']:<8} {row['source_types']:<15} "
              f"{row['hit@100']:<5} {row['hit@500']:<5} {row['hit@1000']:<6} "
              f"{row['hit@2000']:<6} {row['hit@5000']:<6} "
              f"{str(best_rank) if best_rank else '-':<10}")

    # Is any k at 100%?
    print()
    for k in K_VALUES:
        hits = sum(1 for row in all_rows if row[f"hit@{k}"] == "1")
        if hits == len(all_rows):
            print(f"✓ hit@{k} = 100% — ALL questions hit!")
        else:
            print(f"✗ hit@{k} = {100*hits/len(all_rows):.1f}% — {len(all_rows)-hits} questions missed")

    log(f"done — {args.out}")


if __name__ == "__main__":
    main()
