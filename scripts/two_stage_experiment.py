#!/usr/bin/env python3
"""
Two-Stage Retrieval Experiment.

Stage 1: fast single-vector retrieval (jina-v3 OR gte-large)
          → top-1,000 candidate doc_ids.

Stage 2: exact ColBERT MaxSim rerank on those 1,000 candidates
          → top-100 final ranking.

Two independent pipelines are run:
  A. jina-v3 (Stage 1) → ColBERT MaxSim (Stage 2)
  B. gte-large (Stage 1) → ColBERT MaxSim (Stage 2)

Output CSV:
  All original question fields +
  jina_v3_stage1_top1000  (semicolon-separated doc IDs from Stage 1)
  gte_stage1_top1000
  jina_v3_then_colbert_top100  (Stage 2 rerank on jina-v3 pool)
  gte_then_colbert_top100      (Stage 2 rerank on gte pool)
  Plus per-k accuracy columns for k ∈ {1, 5, 10, 20, 50, 100}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import lancedb

sys.path.insert(0, "/data/projects/rag/backend")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# ---------------------------------------------------------------------------
# Constants

QUESTIONS_PATH = "/home/shanaka/Desktop/projects/rag/data/questions.jsonl"
COLBERT_LANCE = "/data/projects/rag/data/colbert_index/db"
DENSE_LANCE   = "/data/projects/rag/data/dense_index/db"
GTE_LANCE     = "/data/projects/rag/lancedb_data"
RESULTS_CSV   = "/data/projects/rag/data/two_stage_experiment.csv"

EMB_DIM_COLBERT = 128
STAGE1_K = 1000
STAGE2_K = 100

# ---------------------------------------------------------------------------
# Helpers

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_questions(path: str, n: int) -> list[dict]:
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# Model loaders (lazy singletons)

_state = {"models": {}, "tables": {}}

def get_colbert_model():
    if "colbert" not in _state["models"]:
        from pylate import models as pylate_models
        _state["models"]["colbert"] = pylate_models.ColBERT(
            model_name_or_path="jinaai/jina-colbert-v2",
            document_length=8192,
            query_prefix="[QueryMarker]",
            document_prefix="[DocumentMarker]",
            attend_to_expansion_tokens=True,
            trust_remote_code=True,
            device="cpu",
        )
    return _state["models"]["colbert"]


def get_jina_v3_model():
    if "jina_v3" not in _state["models"]:
        from sentence_transformers import SentenceTransformer
        _state["models"]["jina_v3"] = SentenceTransformer(
            "jinaai/jina-embeddings-v3", trust_remote_code=True, device="cpu"
        )
    return _state["models"]["jina_v3"]


def get_gte_model():
    if "gte" not in _state["models"]:
        from sentence_transformers import SentenceTransformer
        _state["models"]["gte"] = SentenceTransformer(
            "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, device="cpu"
        )
    return _state["models"]["gte"]


def get_colbert_table():
    if "colbert" not in _state["tables"]:
        _state["tables"]["colbert"] = lancedb.connect(COLBERT_LANCE).open_table("documents")
    return _state["tables"]["colbert"]


def get_dense_table():
    if "dense" not in _state["tables"]:
        _state["tables"]["dense"] = lancedb.connect(DENSE_LANCE).open_table("documents")
    return _state["tables"]["dense"]


def get_gte_table():
    if "gte" not in _state["tables"]:
        _state["tables"]["gte"] = lancedb.connect(GTE_LANCE).open_table("documents")
    return _state["tables"]["gte"]


# ---------------------------------------------------------------------------
# Stage 1: single-vector retrieval

def jina_v3_stage1(queries: list[str], k: int) -> list[list[str]]:
    model = get_jina_v3_model()
    log("  encoding jina-v3 queries (task=retrieval.query) …")
    q = model.encode(
        queries, task="retrieval.query", batch_size=len(queries),
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )
    q = np.asarray(q, dtype=np.float32)
    t = get_dense_table()
    out: list[list[str]] = []
    for vec in q:
        hits = t.search(vec.tolist()).limit(k).to_list()
        out.append([h["id"] for h in hits])
    return out


def gte_stage1(queries: list[str], k: int) -> list[list[str]]:
    model = get_gte_model()
    log("  encoding gte queries …")
    q = model.encode(
        queries, batch_size=len(queries),
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )
    q = np.asarray(q, dtype=np.float32)
    t = get_gte_table()
    out: list[list[str]] = []
    for vec in q:
        hits = t.search(vec.tolist()).limit(k).to_list()
        out.append([h["id"] for h in hits])
    return out


# ---------------------------------------------------------------------------
# Stage 2: exact ColBERT MaxSim on candidate pool

def _batched_maxsim(query_vecs: np.ndarray, doc_vecs_list: list[np.ndarray]) -> np.ndarray:
    if not doc_vecs_list:
        return np.empty(0, dtype=np.float32)
    n_docs = len(doc_vecs_list)
    Q = query_vecs.shape[0]
    max_k = max(d.shape[0] for d in doc_vecs_list)
    D = np.zeros((n_docs, max_k, EMB_DIM_COLBERT), dtype=np.float32)
    mask = np.zeros((n_docs, max_k), dtype=bool)
    for i, d in enumerate(doc_vecs_list):
        k = d.shape[0]
        D[i, :k] = d; mask[i, :k] = True
    sim = np.einsum("qd,nkd->nqk", query_vecs, D, optimize=True)
    sim = np.where(mask[:, None, :], sim, -np.inf)
    max_per_q = sim.max(axis=2)
    max_per_q = np.where(np.isfinite(max_per_q), max_per_q, 0.0)
    return max_per_q.sum(axis=1).astype(np.float32)


def colbert_stage2(queries: list[str], candidate_lists: list[list[str]], k: int) -> list[list[str]]:
    """
    For each query, run exact MaxSim over the provided candidate doc_ids.
    Returns top-k doc_ids per query.
    """
    model = get_colbert_model()
    t = get_colbert_table()
    log("  encoding colbert queries for Stage 2 …")
    out: list[list[str]] = []
    for qi, (q, cands) in enumerate(zip(queries, candidate_lists)):
        t0 = time.time()
        q_emb = model.encode(
            sentences=[q], is_query=True, batch_size=1,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_vecs = np.asarray(q_emb[0], dtype=np.float32)

        # Batch-fetch candidate embeddings from LanceDB
        safe_ids = ["'" + d.replace("'", "''") + "'" for d in cands]
        in_clause = ", ".join(safe_ids)
        arrow = t.to_lance().to_table(
            columns=["id", "n_tokens", "scale", "embeddings"],
            filter=f"id IN ({in_clause})",
        )
        rows = {arrow.column("id").to_pylist()[i]: i
                for i in range(arrow.num_rows)}

        D_list, order = [], []
        for did in cands:
            i = rows.get(did)
            if i is None:
                continue
            n_t = int(arrow.column("n_tokens").to_pylist()[i])
            s = float(arrow.column("scale").to_pylist()[i])
            blob = arrow.column("embeddings").to_pylist()[i]
            if n_t == 0 or not blob:
                continue
            v = np.frombuffer(blob, dtype=np.int8).reshape(n_t, EMB_DIM_COLBERT).astype(np.float32) * s
            D_list.append(v); order.append(did)

        scores = _batched_maxsim(q_vecs, D_list)
        order_np = np.array(order)
        order_sorted = order_np[np.argsort(-scores)]
        topk = order_sorted[:k].tolist()
        out.append(topk)
        log(f"    q{qi:02d}  {len(D_list)} cands → MaxSim in {time.time()-t0:.2f}s")
    return out


# ---------------------------------------------------------------------------
# Accuracy metrics

def compute_accuracy(expected_ids: list[str], ranked_docs: list[str], k_values=(1, 5, 10, 20, 50, 100)) -> dict:
    """For a single question, compute hit@k for each k."""
    results = {}
    for k in k_values:
        hit = False
        best_rank = None
        for eid in expected_ids:
            for i, doc_path in enumerate(ranked_docs[:k]):
                if eid in doc_path:
                    hit = True
                    if best_rank is None or i+1 < best_rank:
                        best_rank = i+1
                    break
        results[f"hit@{k}"] = hit
        results[f"rank@{k}"] = best_rank if hit else None
    return results


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--stage1-k", type=int, default=STAGE1_K)
    ap.add_argument("--stage2-k", type=int, default=STAGE2_K)
    ap.add_argument("--out", default=RESULTS_CSV)
    args = ap.parse_args()

    log(f"args: {vars(args)}")
    questions = load_questions(args.questions, args.num)
    log(f"loaded {len(questions)} questions")
    queries = [q["question"] for q in questions]

    # Stage 1A: jina-v3 → top-1000
    log("=== Stage 1A: jina-embeddings-v3 ===")
    jina_pool = jina_v3_stage1(queries, args.stage1_k)

    # Stage 1B: gte → top-1000
    log("=== Stage 1B: gte-large-en-v1.5 ===")
    gte_pool = gte_stage1(queries, args.stage1_k)

    # Stage 2A: ColBERT MaxSim on jina-v3 pool
    log("=== Stage 2A: ColBERT rerank on jina-v3 pool ===")
    jina_then_colbert = colbert_stage2(queries, jina_pool, args.stage2_k)

    # Stage 2B: ColBERT MaxSim on gte pool
    log("=== Stage 2B: ColBERT rerank on gte pool ===")
    gte_then_colbert = colbert_stage2(queries, gte_pool, args.stage2_k)

    # Write CSV
    log(f"writing {args.out}")
    k_vals = (1, 5, 10, 20, 50, 100)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "question_id", "question_type", "source_types",
            "question", "expected_doc_ids", "gold_answer", "answer_facts",
            "jina_v3_stage1_top1000",
            "gte_stage1_top1000",
            "jina_v3_then_colbert_top100",
            "gte_then_colbert_top100",
        ]
        for prefix in ["jina_v3", "gte", "jina_then_colbert", "gte_then_colbert"]:
            for k in k_vals:
                header.append(f"{prefix}_hit@{k}")
        w.writerow(header)

        for i, q in enumerate(questions):
            exp_ids = q.get("expected_doc_ids", [])
            row = [
                q.get("question_id", ""),
                q.get("question_type", ""),
                "|".join(q.get("source_types", [])),
                q.get("question", ""),
                "|".join(exp_ids),
                q.get("gold_answer", ""),
                " || ".join(q.get("answer_facts", [])),
                ";".join(jina_pool[i]),
                ";".join(gte_pool[i]),
                ";".join(jina_then_colbert[i]),
                ";".join(gte_then_colbert[i]),
            ]
            # accuracy for each of the 4 pipelines
            for ranked in [jina_pool[i], gte_pool[i], jina_then_colbert[i], gte_then_colbert[i]]:
                acc = compute_accuracy(exp_ids, ranked, k_vals)
                for k in k_vals:
                    row.append("1" if acc[f"hit@{k}"] else "0")
            w.writerow(row)

    log(f"done — wrote {len(questions)} rows to {args.out}")

    # Print summary
    print()
    print("=" * 70)
    print("ACCURACY SUMMARY")
    print("=" * 70)
    pipelines = [
        ("jina-v3 Stage1 (alone)",    [jina_pool[i] for i in range(len(questions))]),
        ("gte Stage1 (alone)",        [gte_pool[i] for i in range(len(questions))]),
        ("jina-v3 → ColBERT",         [jina_then_colbert[i] for i in range(len(questions))]),
        ("gte → ColBERT",             [gte_then_colbert[i] for i in range(len(questions))]),
    ]
    for name, all_ranked in pipelines:
        print(f"\n{name}:")
        for k in k_vals:
            hits = sum(1 for i, q in enumerate(questions)
                       if compute_accuracy(q.get("expected_doc_ids", []), all_ranked[i], [k])[f"hit@{k}"])
            print(f"  hit@{k:3d}:  {hits}/{len(questions)}  ({100*hits/len(questions):.1f}%)")


if __name__ == "__main__":
    main()
