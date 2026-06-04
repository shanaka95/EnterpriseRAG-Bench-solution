"""Retrieval nodes for the LangGraph RAG agent.

Three sequential nodes mirror the production pipeline:
  bm25_node      -> top-N BM25 doc IDs           (sparse)
  jina_node      -> top-N jina-v3 doc IDs        (dense)
  rrf_node       -> RRF-fused top-K doc IDs      (final list passed to the agent)

The RRF ranking uses k0=60 (Cormack SIGIR 2009 default) which we found
to be the best config in the experiments.
"""
from __future__ import annotations
import time
from collections import defaultdict
from pathlib import Path
import sys

# Ensure backend root is on sys.path so we can use the same libs as
# scripts/retrieve_100.py
_BACKEND = "/data/projects/rag/backend"
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from .state import AgentState
from .config import load_config


# Module-level caches (per-process) — building BM25 + loading jina takes
# ~30-60 s the first time; we want every question after the first to be fast.
_BM25 = None
_BM25_IDS: list[str] | None = None
_JINA_MODEL = None
_LANCE_TABLE = None


def _get_bm25(cfg):
    """Lazily build the BM25 index over the 511k corpus."""
    global _BM25, _BM25_IDS
    if _BM25 is not None:
        return _BM25, _BM25_IDS
    import bm25s
    t0 = time.time()
    texts: list[str] = []
    ids: list[str] = []
    for fp in sorted(Path(cfg.corpus_dir).rglob("*.txt")):
        ids.append(fp.relative_to(cfg.corpus_dir).as_posix())
        texts.append(fp.read_text(encoding="utf-8", errors="replace"))
    tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    bm25 = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    bm25.index(tokens, show_progress=False)
    print(f"[retrieval] BM25 index built: {len(ids):,} docs in {time.time()-t0:.1f}s",
          flush=True)
    _BM25, _BM25_IDS = bm25, ids
    return bm25, ids


def _get_jina_and_lance(cfg):
    """Lazily load jina-v3 (CPU) and open the LanceDB table."""
    global _JINA_MODEL, _LANCE_TABLE
    if _JINA_MODEL is not None:
        return _JINA_MODEL, _LANCE_TABLE
    from sentence_transformers import SentenceTransformer
    import lancedb
    t0 = time.time()
    model = SentenceTransformer(cfg.jina_model_id,
                                 trust_remote_code=True, device="cpu")
    table = lancedb.connect(cfg.dense_index_dir).open_table("documents")
    print(f"[retrieval] jina-v3 + LanceDB ready in {time.time()-t0:.1f}s",
          flush=True)
    _JINA_MODEL, _LANCE_TABLE = model, table
    return model, table


# ---------- nodes ----------

def bm25_retrieve(state: AgentState) -> dict:
    """BM25 over the full corpus → top-N doc IDs."""
    cfg = load_config()
    import bm25s
    bm25, ids = _get_bm25(cfg)
    t0 = time.time()
    qtok = bm25s.tokenize([state["question"]], stopwords="en", show_progress=False)
    res = bm25.retrieve(qtok, corpus=ids, k=cfg.bm_top_n, show_progress=False)
    bm_ranked = [str(d) for d in res.documents[0]]
    elapsed = time.time() - t0
    return {
        "bm_ranked": bm_ranked,
        "node_trace": [{
            "node": "bm25_retrieve", "elapsed_s": round(elapsed, 3),
            "n_returned": len(bm_ranked),
        }],
    }


def jina_dense_retrieve(state: AgentState) -> dict:
    """jina-embeddings-v3 dense search over the 511k LanceDB table → top-N doc IDs."""
    cfg = load_config()
    model, table = _get_jina_and_lance(cfg)
    t0 = time.time()
    qvec = model.encode(
        [state["question"]], task="retrieval.query", batch_size=1,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )[0]
    hits = table.search(qvec.tolist()).limit(cfg.jv_top_n).to_list()
    jv_ranked = [h["id"] for h in hits]
    elapsed = time.time() - t0
    return {
        "jv_ranked": jv_ranked,
        "node_trace": [{
            "node": "jina_dense_retrieve", "elapsed_s": round(elapsed, 3),
            "n_returned": len(jv_ranked),
        }],
    }


def make_rrf_fuse(state_holder: dict):
    """Create the rrf_fuse node bound to ``state_holder``.

    The state_holder is the same dict the tool closure uses, so writing
    refined_doc_ids into it makes them visible to the agent's tool.
    """
    cfg = load_config()

    def rrf_fuse(state: AgentState) -> dict:
        jv = state.get("jv_ranked") or []
        bm = state.get("bm_ranked") or []
        if not jv or not bm:
            raise RuntimeError("rrf_fuse called before both retrievers ran")

        t0 = time.time()
        scores: dict[str, float] = defaultdict(float)
        for rank, d in enumerate(jv, 1):
            scores[d] += 1.0 / (cfg.rrf_k0 + rank)
        for rank, d in enumerate(bm, 1):
            scores[d] += 1.0 / (cfg.rrf_k0 + rank)
        jvr = {d: i for i, d in enumerate(jv, 1)}
        bmr = {d: i for i, d in enumerate(bm, 1)}
        ranked = sorted(scores.keys(),
                        key=lambda d: (-scores[d], jvr.get(d, 10**9), bmr.get(d, 10**9)))
        refined = ranked[:cfg.rrf_top_k]
        elapsed = time.time() - t0

        # Sync into the tool's state_holder so the agent's tool closure
        # can see the refined list and the cursor.
        state_holder["refined_doc_ids"] = refined
        state_holder["current_idx"] = 0

        return {
            "rrf_ranked": ranked,
            "rrf_scores": [scores[d] for d in ranked],
            "refined_doc_ids": refined,
            "current_idx": 0,
            "node_trace": [{
                "node": "rrf_fuse", "elapsed_s": round(elapsed, 3),
                "n_unique": len(ranked), "n_returned": len(refined),
            }],
        }

    return rrf_fuse
