"""
Jina-ColBERT-v2 reranker module — replaces the cross-encoder in Phase 5 of
routes.py when COLBERT_RERANK_ENABLED=true.

Loads pre-computed multi-vector embeddings from a local LanceDB instance
(int8-quantized, 128-dim per token), encodes the query at request time on
CPU with PyLate, then computes MaxSim scores against candidate doc embeddings.

Architecture:
  - Lazy singleton model + LanceDB connection (thread-safe, same pattern as
    embedding.py).
  - Query encoding via PyLate ColBERT on CPU with attn_implementation='eager'.
  - Doc-embeddings cache: an in-process LRU is unnecessary because the
    rerank candidates change per query; we batch-fetch each call.
  - MaxSim computed with batched np.einsum on length-padded chunks of 100.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Singletons (thread-safe lazy-init, mirrors embedding.py)
_model = None
_model_lock = threading.Lock()
_lance_table = None
_lance_lock = threading.Lock()

# Constants shared with the server-side embed pipeline
EMB_DIM = 128


# ---------------------------------------------------------------- model

def get_model():
    """Lazy-load the PyLate ColBERT model on CPU.

    Jina-ColBERT-v2 uses its own custom XLMRoberta implementation
    (trust_remote_code) that auto-activates flash-attn when available.
    On the local CPU box flash_attn is not installed; the model falls
    back to its internal eager path automatically. We do NOT pass
    attn_implementation — HF's base XLMRobertaModel rejects it for
    this architecture.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from pylate import models as pylate_models
                from app.core.config import settings
                logger.info(
                    f"Loading ColBERT reranker: {settings.COLBERT_MODEL_NAME} "
                    f"(CPU)"
                )
                t0 = time.time()
                _model = pylate_models.ColBERT(
                    model_name_or_path=settings.COLBERT_MODEL_NAME,
                    document_length=8192,
                    query_prefix=settings.COLBERT_QUERY_PREFIX,
                    document_prefix=settings.COLBERT_DOC_PREFIX,
                    attend_to_expansion_tokens=True,
                    trust_remote_code=True,
                    device="cpu",
                )
                logger.info(f"ColBERT reranker loaded in {time.time()-t0:.1f}s")
    return _model


def get_table():
    """Lazy-open the LanceDB documents table."""
    global _lance_table
    if _lance_table is None:
        with _lance_lock:
            if _lance_table is None:
                import lancedb
                from app.core.config import settings
                logger.info(f"Opening ColBERT LanceDB at {settings.COLBERT_INDEX_PATH}")
                db = lancedb.connect(settings.COLBERT_INDEX_PATH)
                _lance_table = db.open_table("documents")
                n = _lance_table.count_rows()
                logger.info(f"ColBERT index loaded: {n} rows")
    return _lance_table


# ---------------------------------------------------------------- fetch

def _fetch_doc_embeddings(doc_ids: list[str]) -> dict[str, np.ndarray]:
    """Fetch and dequantize doc embeddings for the given IDs.

    Returns a dict mapping doc_id -> (n_tokens, 128) float32 array.
    Missing IDs are simply absent from the result.
    """
    if not doc_ids:
        return {}
    table = get_table()
    # LanceDB SQL filter: id IN (...). We escape single quotes by replacing
    # with two single quotes; doc_ids are file paths so quotes are rare.
    safe_ids = ["'" + d.replace("'", "''") + "'" for d in doc_ids]
    in_clause = ", ".join(safe_ids)
    # Use to_lance().to_table() with column projection to avoid loading text
    arrow_tbl = (
        table.to_lance()
             .to_table(columns=["id", "n_tokens", "scale", "embeddings"],
                       filter=f"id IN ({in_clause})")
    )
    out: dict[str, np.ndarray] = {}
    ids = arrow_tbl.column("id").to_pylist()
    n_tokens = arrow_tbl.column("n_tokens").to_pylist()
    scales = arrow_tbl.column("scale").to_pylist()
    embs = arrow_tbl.column("embeddings").to_pylist()
    for i, did in enumerate(ids):
        n = int(n_tokens[i])
        scale = float(scales[i])
        if n == 0 or not embs[i]:
            continue
        q = np.frombuffer(embs[i], dtype=np.int8).reshape(n, EMB_DIM)
        out[did] = q.astype(np.float32) * scale
    return out


# ---------------------------------------------------------------- maxsim

def _maxsim_padded(query_vecs: np.ndarray,
                   doc_vecs_list: list[np.ndarray]) -> np.ndarray:
    """Compute MaxSim scores for one query against a list of docs.

    query_vecs: (Q, 128) float32, L2-norm per token.
    doc_vecs_list: list of (n_tokens_i, 128) float32, L2-norm per token.

    Returns: (N,) float32 array of MaxSim scores.
    Uses padded einsum: pads docs to max length in the chunk, masks pad
    tokens with -inf so they never win the max.
    """
    if not doc_vecs_list:
        return np.empty(0, dtype=np.float32)
    n_docs = len(doc_vecs_list)
    Q = query_vecs.shape[0]
    max_k = max(d.shape[0] for d in doc_vecs_list)

    # Stack with right-pad
    D = np.zeros((n_docs, max_k, EMB_DIM), dtype=np.float32)
    mask = np.zeros((n_docs, max_k), dtype=bool)
    for i, d in enumerate(doc_vecs_list):
        k = d.shape[0]
        D[i, :k] = d
        mask[i, :k] = True

    # sim: (N, Q, K)
    sim = np.einsum("qd,nkd->nqk", query_vecs, D, optimize=True)
    # mask out pad positions: set them to -inf so they lose the max
    sim = np.where(mask[:, None, :], sim, -np.inf)
    # MaxSim: for each (n, q), max over k, then sum over q
    max_per_query_tok = sim.max(axis=2)            # (N, Q)
    # If a row is all -inf (empty doc), skip — sum yields -inf → coerce to 0
    max_per_query_tok = np.where(np.isfinite(max_per_query_tok),
                                  max_per_query_tok, 0.0)
    scores = max_per_query_tok.sum(axis=1)         # (N,)
    return scores.astype(np.float32)


# ---------------------------------------------------------------- public API

def colbert_rerank(query: str,
                   doc_ids: list[str],
                   top_k: int | None = None,
                   chunk_size: int = 100) -> list[tuple[str, float]]:
    """Rerank doc_ids by ColBERT MaxSim against the query.

    Args:
        query: user query string.
        doc_ids: candidate doc IDs from first-stage retrieval.
        top_k: return only top_k highest-scoring; if None, return all.
        chunk_size: # docs scored together in one einsum call. Smaller →
            lower peak memory; larger → more BLAS efficiency. 100 is a
            sweet spot for ~1160-token avg docs on a 56-core CPU.

    Returns:
        List of (doc_id, score) tuples sorted desc by score. May be
        shorter than len(doc_ids) if some IDs were missing from the
        index.
    """
    if not doc_ids:
        return []
    t_start = time.time()
    model = get_model()

    # Encode query → (Q, 128) float32, L2-norm per token
    t0 = time.time()
    q_out = model.encode(
        sentences=[query],
        is_query=True,
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if isinstance(q_out, list):
        q_vecs = q_out[0].astype(np.float32, copy=False)
    else:
        q_vecs = np.asarray(q_out, dtype=np.float32).reshape(-1, EMB_DIM)
    t_query = time.time() - t0

    # Fetch doc embeddings in one go
    t0 = time.time()
    docs = _fetch_doc_embeddings(doc_ids)
    t_fetch = time.time() - t0

    # MaxSim in chunks
    t0 = time.time()
    results: list[tuple[str, float]] = []
    ordered_ids = [d for d in doc_ids if d in docs]
    for i in range(0, len(ordered_ids), chunk_size):
        chunk_ids = ordered_ids[i:i + chunk_size]
        chunk_vecs = [docs[d] for d in chunk_ids]
        scores = _maxsim_padded(q_vecs, chunk_vecs)
        for d, s in zip(chunk_ids, scores):
            results.append((d, float(s)))
    t_score = time.time() - t0

    results.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        results = results[:top_k]

    logger.info(
        f"colbert_rerank: {len(doc_ids)} in, {len(results)} scored, "
        f"q_encode={t_query*1000:.0f}ms, fetch={t_fetch*1000:.0f}ms, "
        f"maxsim={t_score*1000:.0f}ms, total={(time.time()-t_start)*1000:.0f}ms"
    )
    return results
