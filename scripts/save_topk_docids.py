#!/usr/bin/env python3
"""
Regenerate and save the full top-K doc ID lists for all 500 questions using
jina-embeddings-v3 (task="retrieval.query"), flat L2 over LanceDB.

The previous experiment (jina_v3_scale_experiment.py) only saved the first 5
doc IDs per top-K column for quick inspection. This script saves the full
top-100, 500, 1000, 2000, and 5000 doc ID lists so they can be reused in
subsequent experiments (e.g. reranking, hybrid fusion) without re-running
the expensive first-stage retrieval.

Output:
  - /data/projects/rag/data/jina_v3_topk_docids.jsonl  (one row per question)
        {"question_id": "qst_0001",
         "question_type": "basic",
         "source_types": ["github"],
         "expected_doc_ids": ["dsid_..."],
         "top_100_ids":   [...],   # length 100
         "top_500_ids":   [...],   # length 500
         "top_1000_ids":  [...],   # length 1000
         "top_2000_ids":  [...],   # length 2000
         "top_5000_ids":  [...],   # length 5000
         "hit_at_100": 0|1, "rank_at_100": int|null, ...
         "hit_at_5000": 0|1, "rank_at_5000": int|null}

  - A summary printed to stdout matching the original CSV's hit@K table,
    so the regenerated results can be sanity-checked against the existing
    jina_v3_scale_experiment.csv.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/data/projects/rag/backend")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import lancedb

# ---------------------------------------------------------------------------

QUESTIONS_PATH = "/data/projects/rag/data/questions.jsonl"
DENSE_LANCE   = "/data/projects/rag/data/dense_index/db"
OUT_JSONL     = "/data/projects/rag/data/jina_v3_topk_docids.jsonl"

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


def best_match_rank(ranked_ids: list[str], expected_ids: list[str]) -> int | None:
    """Return the 1-based rank of the first expected_id found in ranked_ids, or None."""
    for ei, eid in enumerate(expected_ids):
        for ri, doc_path in enumerate(ranked_ids):
            if eid in doc_path:
                return ri + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--out", default=OUT_JSONL)
    ap.add_argument("--model-device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    log(f"args: {vars(args)}")
    questions = load_questions(args.questions, args.num)
    log(f"loaded {len(questions)} questions")

    # Load model (jina-v3, retrieval.query task)
    from sentence_transformers import SentenceTransformer
    log("loading jina-embeddings-v3 …")
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        device=args.model_device,
    )

    # Encode all queries in one batch
    queries = [q["question"] for q in questions]
    log(f"encoding {len(queries)} queries on {args.model_device} …")
    t0 = time.time()
    qvecs = model.encode(
        queries,
        task="retrieval.query",
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    log(f"  encoded in {time.time()-t0:.1f}s  (shape={qvecs.shape})")

    # Open LanceDB
    table = lancedb.connect(DENSE_LANCE).open_table("documents")
    n_docs = table.count_rows()
    log(f"LanceDB: {n_docs:,} docs")

    # Search per query, store full top-5000 for each
    log(f"searching {len(queries)} queries × top-{max(K_VALUES)} …")
    t_search = time.time()
    per_q_top_ids: list[list[str]] = []
    for qi, vec in enumerate(qvecs):
        hits = table.search(vec.tolist()).limit(max(K_VALUES)).to_list()
        ids = [h["id"] for h in hits]
        per_q_top_ids.append(ids)
        if (qi + 1) % 25 == 0:
            log(f"  {qi+1}/{len(queries)} done  (elapsed {time.time()-t_search:.0f}s)")
    log(f"  all searches done in {time.time()-t_search:.0f}s")

    # Build per-question records with full top-K slices + hit/rank
    log("building output records …")
    out_records: list[dict] = []
    for i, q in enumerate(questions):
        qid = q.get("question_id", "")
        exp_ids = q.get("expected_doc_ids", [])
        full_ranked = per_q_top_ids[i]
        rec = {
            "question_id": qid,
            "question_type": q.get("question_type", ""),
            "source_types": q.get("source_types", []),
            "expected_doc_ids": exp_ids,
        }
        for k in K_VALUES:
            rec[f"top_{k}_ids"] = full_ranked[:k]
            rank = best_match_rank(rec[f"top_{k}_ids"], exp_ids)
            rec[f"hit_at_{k}"] = 1 if rank is not None else 0
            rec[f"rank_at_{k}"] = rank
        out_records.append(rec)

    # Write JSONL
    log(f"writing {args.out}")
    with open(args.out, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    log(f"  wrote {len(out_records)} records  ({size_mb:.1f} MB)")

    # Summary — match the original CSV's hit@K table for sanity check
    print()
    print("=" * 70)
    print("REGENERATED JINA-V3 TOP-K DOC IDS — SUMMARY (sanity check)")
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
        # single-source questions only
        if len(rec["source_types"]) == 1:
            src = rec["source_types"][0]
            by_source.setdefault(src, []).append(rec["hit_at_1000"])
    for src, hits in sorted(by_source.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        h = sum(hits)
        n = len(hits)
        print(f"  {src:<14}  {h:>4}/{n:<4}  {100*h/n:>6.1f}%")

    log("done.")


if __name__ == "__main__":
    main()
