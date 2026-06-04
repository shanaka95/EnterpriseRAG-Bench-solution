#!/usr/bin/env python3
"""
Retrieval experiment — top-100 docs from 3 indexes for N questions.

For each question (configurable N, default 10) from
/home/shanaka/Desktop/projects/rag/data/questions.jsonl, this script:

  1. Dense jina-embeddings-v3:
       - Encode query with `task="retrieval.query"` (MUST match retrieval.passage
         used at build time), mean-pool, L2-normalize → 1024-dim.
       - Flat L2 over the 2.0 GB LanceDB → top-100.
       - L2 on unit vectors == cosine similarity.

  2. Dense gte-large-en-v1.5 (legacy, also at /data/projects/rag/lancedb_data):
       - Encode query with sentence-transformers (CLS pool, normalize).
       - Flat L2 search on the 2.9 GB LanceDB → top-100.
       - Same cosine-via-L2 trick.

  3. Multi-vector jina-colbert-v2:
       - Encode query → (Q, 128) token vectors (PyLate, query prefix).
       - Build per-doc centroid (mean of doc token vectors) once → 128-dim
         per doc, cached to .npy on disk. We also include the per-token
         MaxSim-ready embeddings (int8 + scale, from LanceDB) so we can
         rerank candidates with EXACT MaxSim.
       - For each query, dot-product query centroid vs all doc centroids
         → top-1000 candidate doc_ids.
       - Exact MaxSim over the 1000 candidates → top-100.

Output:
  CSV at /data/projects/rag/data/retrieval_experiment.csv with all original
  question fields (question_id, question_type, source_types, question,
  expected_doc_ids, gold_answer, answer_facts) plus three columns of
  semicolon-separated doc_ids:
    colbert_top100, jina_v3_top100, gte_large_top100

The doc_ids in the CSV are full relative paths (as stored in the indexes),
e.g.  'gmail/ben_carter/dsid_xxxx__2026-08-21-onprem-infra-validation.txt'.
The expected_doc_ids in the source JSONL are bare dsids
(e.g.  'dsid_ae068ee4aa9640159427cd941bef0238'); downstream comparison
should substring-match.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import lancedb

# venv
sys.path.insert(0, "/data/projects/rag/backend")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# ---------------------------------------------------------------------------
# Constants

QUESTIONS_PATH = "/home/shanaka/Desktop/projects/rag/data/questions.jsonl"
COLBERT_LANCE = "/data/projects/rag/data/colbert_index/db"
DENSE_LANCE   = "/data/projects/rag/data/dense_index/db"
GTE_LANCE     = "/data/projects/rag/lancedb_data"
CENTROID_NPY  = "/data/projects/rag/data/dense_index/_build/colbert_doc_maxpool.npy"
DOCIDS_NPY    = "/data/projects/rag/data/dense_index/_build/colbert_doc_ids.npy"
RESULTS_CSV   = "/data/projects/rag/data/retrieval_experiment.csv"

EMB_DIM_COLBERT = 128
EMB_DIM_DENSE   = 1024
TOP_K           = 100
COLBERT_CANDIDATES = 1000   # centroid ANN pool, then exact MaxSim rerank

# ---------------------------------------------------------------------------
# Helpers

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_questions(path: str, n: int) -> list[dict]:
    """Read first n non-empty questions from the JSONL."""
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
# ColBERT doc centroids

def build_colbert_centroids(force: bool = False) -> tuple[np.ndarray, list[str]]:
    """
    Compute (n_docs, 128) **max-pooled** doc representation and aligned doc_id list.
    Stream from LanceDB in row-group chunks. Dequantize int8 → float32 with
    per-doc scale, then element-wise max over tokens. Cached to disk.

    Why max-pool not mean: a doc that strongly contains one query-relevant
    token (and many irrelevant ones) has a strong max-pool element in that
    direction. The mean drowns it out. Max-pool preserves the "best match"
    signal that late-interaction retrieval fundamentally relies on.
    """
    if (not force
            and os.path.exists(CENTROID_NPY)
            and os.path.exists(DOCIDS_NPY)):
        cents = np.load(CENTROID_NPY)
        ids   = np.load(DOCIDS_NPY, allow_pickle=True).tolist()
        log(f"  loaded cached max-pool doc reps: {cents.shape}  ({len(ids):,} ids)")
        return cents, ids

    log("  computing per-doc max-pool from ColBERT LanceDB (one-time, streamed) …")
    t0 = time.time()
    table = lancedb.connect(COLBERT_LANCE).open_table("documents")
    n = table.count_rows()
    cents = np.zeros((n, EMB_DIM_COLBERT), dtype=np.float32)
    ids: list[str] = []
    CHUNK = 1000   # rows per streamed scan
    processed = 0
    for offset in range(0, n, CHUNK):
        arrow = table.to_lance().to_table(
            columns=["id", "n_tokens", "scale", "embeddings"],
            offset=offset, limit=CHUNK,
        )
        n_rows = arrow.num_rows
        if n_rows == 0:
            break
        col_id    = arrow.column("id").to_pylist()
        col_tok   = arrow.column("n_tokens").to_pylist()
        col_scale = arrow.column("scale").to_pylist()
        col_emb   = arrow.column("embeddings").to_pylist()
        for i in range(n_rows):
            idx = offset + i
            ids.append(col_id[i])
            n_t = int(col_tok[i])
            if n_t == 0 or not col_emb[i]:
                continue
            q = np.frombuffer(col_emb[i], dtype=np.int8).reshape(n_t, EMB_DIM_COLBERT)
            v = q.astype(np.float32) * float(col_scale[i])
            # element-wise max over tokens — preserves best-match signal
            cents[idx] = v.max(axis=0)
        processed += n_rows
        if (offset // CHUNK) % 20 == 0:
            log(f"    max-pool progress: {processed:,}/{n:,}  ({100*processed/n:.1f}%)")

    # L2-normalize so dot product = cosine
    norms = np.linalg.norm(cents, axis=1, keepdims=True).clip(min=1e-9)
    cents = cents / norms

    np.save(CENTROID_NPY, cents)
    np.save(DOCIDS_NPY, np.array(ids, dtype=object), allow_pickle=True)
    log(f"  max-pool doc reps built in {time.time()-t0:.1f}s  shape={cents.shape}")
    return cents, ids


# ---------------------------------------------------------------------------
# Model loaders (lazy, cached after first call)

_state = {"models": {}, "tables": {}, "centroids": None, "centroid_ids": None}

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
# Per-algorithm retrieval

def jina_v3_topk(queries: list[str], k: int) -> list[list[str]]:
    """jina-embeddings-v3 single-vector → top-k via L2 over normalized 1024-dim vectors."""
    model = get_jina_v3_model()
    log("  encoding jina-v3 queries (task=retrieval.query) …")
    q = model.encode(
        queries, task="retrieval.query", batch_size=len(queries),
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )
    # convert to list-of-lists for lance
    q = np.asarray(q, dtype=np.float32)
    t = get_dense_table()
    log(f"  jina-v3 search ({len(queries)} q × {t.count_rows():,} d) …")
    out: list[list[str]] = []
    for i, vec in enumerate(q):
        hits = t.search(vec.tolist()).limit(k).to_list()
        out.append([h["id"] for h in hits])
    return out


def gte_topk(queries: list[str], k: int) -> list[list[str]]:
    """gte-large-en-v1.5 single-vector → top-k via L2 over normalized 1024-dim vectors."""
    model = get_gte_model()
    log("  encoding gte queries (CLS pool, default normalize) …")
    q = model.encode(
        queries, batch_size=len(queries),
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )
    q = np.asarray(q, dtype=np.float32)
    t = get_gte_table()
    log(f"  gte search ({len(queries)} q × {t.count_rows():,} d) …")
    out: list[list[str]] = []
    for i, vec in enumerate(q):
        hits = t.search(vec.tolist()).limit(k).to_list()
        out.append([h["id"] for h in hits])
    return out


def colbert_topk(queries: list[str], k: int, candidates: int = COLBERT_CANDIDATES) -> list[list[str]]:
    """
    jina-colbert-v2 multi-vector → 2-stage:
      Stage 1 — multi-token ANN against max-pool doc reps:
        doc_max[d] = max over doc tokens of d_t ∈ R^128           (n_docs, 128)
        For each query token q_i:  sims_i[d] = q_i · doc_max[d]   (n_docs,)
        For each doc:  aggsim[d] = max_i sims_i[d]               (n_docs,)
        top-candidates = argpartition(aggsim, candidates)
      Stage 2 — exact MaxSim over the candidates → top-k.
    """
    model = get_colbert_model()
    cents, doc_ids = build_colbert_centroids()      # these are max-pool reps
    t = get_colbert_table()

    log("  encoding colbert queries (with [QueryMarker] prefix) …")
    out: list[list[str]] = []
    for qi, q in enumerate(queries):
        t0 = time.time()
        q_emb = model.encode(
            sentences=[q], is_query=True, batch_size=1,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_vecs = np.asarray(q_emb[0], dtype=np.float32)            # (Q, 128)

        # Stage 1: multi-token ANN — for each query token, sim vs all doc max reps
        # sims shape (Q, n_docs); aggsim shape (n_docs,) = max over Q
        sims = q_vecs @ cents.T                                    # (Q, n_docs)
        aggsim = sims.max(axis=0)                                  # (n_docs,)
        cand_idx = np.argpartition(-aggsim, candidates)[:candidates]
        cand_idx = cand_idx[np.argsort(-aggsim[cand_idx])]        # top-C sorted
        cand_doc_ids = [doc_ids[i] for i in cand_idx]

        # Stage 2: exact MaxSim rerank over the candidates
        safe_ids = ["'" + d.replace("'", "''") + "'" for d in cand_doc_ids]
        in_clause = ", ".join(safe_ids)
        arrow = (
            t.to_lance().to_table(
                columns=["id", "n_tokens", "scale", "embeddings"],
                filter=f"id IN ({in_clause})",
            )
        )
        D_list = []
        order = []
        rows = {arrow.column("id").to_pylist()[i]: i
                for i in range(arrow.num_rows)}
        for did in cand_doc_ids:
            i = rows.get(did)
            if i is None:
                continue
            n_t = int(arrow.column("n_tokens").to_pylist()[i])
            s   = float(arrow.column("scale").to_pylist()[i])
            blob = arrow.column("embeddings").to_pylist()[i]
            if n_t == 0 or not blob:
                continue
            v = np.frombuffer(blob, dtype=np.int8).reshape(n_t, EMB_DIM_COLBERT).astype(np.float32) * s
            D_list.append(v); order.append(did)

        scores = _batched_maxsim(q_vecs, D_list)                   # (N,)
        order_np = np.array(order)
        order_sorted = order_np[np.argsort(-scores)]
        topk = order_sorted[:k].tolist()
        out.append(topk)
        log(f"    q{qi:02d}  MaxSim over {len(D_list)} cands in {time.time()-t0:.2f}s")
    return out


def _batched_maxsim(query_vecs: np.ndarray, doc_vecs_list: list[np.ndarray]) -> np.ndarray:
    """
    For one query, batched exact MaxSim over N docs.
    query_vecs: (Q, 128) float32
    doc_vecs_list: list of (n_tokens_i, 128) float32
    Returns: (N,) float32 — sum over Q of max over doc tokens.
    Pads to max doc length in the batch, masks pad with -inf.
    """
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
    max_per_q = sim.max(axis=2)                                    # (N, Q)
    max_per_q = np.where(np.isfinite(max_per_q), max_per_q, 0.0)
    return max_per_q.sum(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    ap.add_argument("--num", type=int, default=10,
                    help="How many questions to read from the head of the JSONL.")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--colbert-candidates", type=int, default=COLBERT_CANDIDATES)
    ap.add_argument("--out", default=RESULTS_CSV)
    ap.add_argument("--skip-colbert", action="store_true")
    ap.add_argument("--skip-jina-v3", action="store_true")
    ap.add_argument("--skip-gte",     action="store_true")
    ap.add_argument("--rebuild-centroids", action="store_true")
    args = ap.parse_args()

    log(f"args: {vars(args)}")

    questions = load_questions(args.questions, args.num)
    log(f"loaded {len(questions)} questions")

    queries = [q["question"] for q in questions]
    results_colbert: list[list[str]] = [[] for _ in questions]
    results_jina:    list[list[str]] = [[] for _ in questions]
    results_gte:     list[list[str]] = [[] for _ in questions]

    if not args.skip_jina_v3:
        log("=== jina-embeddings-v3 ===")
        results_jina = jina_v3_topk(queries, args.top_k)

    if not args.skip_gte:
        log("=== gte-large-en-v1.5 ===")
        results_gte = gte_topk(queries, args.top_k)

    if not args.skip_colbert:
        log("=== jina-colbert-v2 (centroid ANN + exact MaxSim) ===")
        if args.rebuild_centroids:
            build_colbert_centroids(force=True)
        results_colbert = colbert_topk(queries, args.top_k, args.colbert_candidates)

    # Write CSV
    log(f"writing {args.out}")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "question_type", "source_types",
            "question", "expected_doc_ids", "gold_answer", "answer_facts",
            "colbert_top100", "jina_v3_top100", "gte_large_top100",
            "colbert_n_unique", "jina_v3_n_unique", "gte_n_unique",
        ])
        for i, q in enumerate(questions):
            cb = results_colbert[i] if results_colbert else []
            jn = results_jina[i]    if results_jina    else []
            gt = results_gte[i]     if results_gte     else []
            w.writerow([
                q.get("question_id", ""),
                q.get("question_type", ""),
                "|".join(q.get("source_types", [])),
                q.get("question", ""),
                "|".join(q.get("expected_doc_ids", [])),
                q.get("gold_answer", ""),
                " || ".join(q.get("answer_facts", [])),
                ";".join(cb),
                ";".join(jn),
                ";".join(gt),
                len(set(cb)),
                len(set(jn)),
                len(set(gt)),
            ])
    log(f"done — wrote {len(questions)} rows to {args.out}")


if __name__ == "__main__":
    main()
