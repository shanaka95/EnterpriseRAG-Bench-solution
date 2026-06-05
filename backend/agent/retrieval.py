"""Retrieval nodes for the LangGraph RAG agent.

Three sequential nodes mirror the production pipeline:
  bm25_node      -> top-N BM25 doc IDs           (sparse)
  jina_node      -> top-N jina-v3 doc IDs        (dense)
  rrf_node       -> RRF-fused top-K doc IDs      (final list passed to the agent)

The RRF ranking uses k0=60 (Cormack SIGIR 2009 default) which we found
to be the best config in the experiments.

BM25 on-disk cache
------------------
The BM25 index build is the dominant cold-start cost (~300 s for 511k docs).
Because the index is a function of the **corpus** (not the query), it can be
persisted to disk and reused across script invocations. The cache lives at
``cfg.bm25_index_dir/<corpus_basename>__k1{K}_b{B>/`` and contains:
  * the bm25s CSC arrays (mmap'd on load → no 600 MB RSS spike)
  * ``doc_ids.json`` — 511,962 strings, the bm25s positions → doc_ids mapping
  * ``corpus_fingerprint.txt`` — sha256 of sorted rel-paths + dir mtime, used
    to invalidate the cache when the corpus changes
"""
from __future__ import annotations
import hashlib
import json
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


# Files that bm25.save() writes (so we can detect a complete cache dir).
_BM25_CACHE_FILES = (
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "vocab.index.json",
    "params.index.json",
    "doc_ids.json",
    "corpus_fingerprint.txt",
)


def _bm25_cache_dir(cfg) -> Path:
    """Return the on-disk cache directory for the current config.

    Keyed by corpus basename + k1 + b. Changing any of these creates a fresh
    subdirectory, so parameter sweeps don't collide.
    """
    return (Path(cfg.bm25_index_dir)
            / f"{Path(cfg.corpus_dir).name}__k1={cfg.bm25_k1}_b={cfg.bm25_b}")


def _corpus_fingerprint(corpus_dir: str) -> str:
    """Cheap fingerprint of the corpus: hash of sorted rel-paths + dir mtime.

    Detects add/remove/rename and any change under corpus_dir. Does NOT hash
    file contents (overkill at 511k docs / 3.3 GB).
    """
    root = Path(corpus_dir)
    rels = sorted(str(p.relative_to(root).as_posix())
                  for p in root.rglob("*.txt"))
    try:
        mtime = root.stat().st_mtime
    except OSError:
        mtime = 0
    h = hashlib.sha256()
    h.update(f"n={len(rels)}\nmtime={mtime:.3f}\n".encode())
    for r in rels:
        h.update(r.encode())
        h.update(b"\n")
    return h.hexdigest()


def _try_load_bm25(cfg):
    """Return (bm25, ids) from disk cache, or (None, None) on miss."""
    cache_dir = _bm25_cache_dir(cfg)
    if not cache_dir.is_dir():
        return None, None
    for fname in _BM25_CACHE_FILES:
        if not (cache_dir / fname).is_file():
            return None, None
    fp_path = cache_dir / "corpus_fingerprint.txt"
    if fp_path.read_text().strip() != _corpus_fingerprint(cfg.corpus_dir):
        return None, None
    import bm25s
    t0 = time.time()
    bm25 = bm25s.BM25.load(str(cache_dir), mmap=True, load_corpus=False)
    ids = json.loads((cache_dir / "doc_ids.json").read_text())
    print(f"[retrieval] BM25 index loaded from cache in {time.time()-t0:.1f}s "
          f"({len(ids):,} docs, k1={cfg.bm25_k1}, b={cfg.bm25_b})",
          flush=True)
    return bm25, ids


def _save_bm25(bm25, ids, cfg):
    """Persist the BM25 index to disk. Best-effort — failures are logged
    but never break the in-memory index."""
    cache_dir = _bm25_cache_dir(cfg)
    try:
        t0 = time.time()
        cache_dir.mkdir(parents=True, exist_ok=True)
        # corpus=None skips the wasteful corpus.jsonl dump — we already have
        # the corpus on disk and don't need the text in the cache.
        bm25.save(str(cache_dir), corpus=None)
        (cache_dir / "doc_ids.json").write_text(json.dumps(ids))
        (cache_dir / "corpus_fingerprint.txt").write_text(
            _corpus_fingerprint(cfg.corpus_dir)
        )
        print(f"[retrieval] BM25 index cached at {cache_dir} "
              f"in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[retrieval] WARNING: BM25 cache save failed: {e!r}",
              flush=True)


def _get_bm25(cfg):
    """Lazily build (or load) the BM25 index over the corpus.

    Order of operations:
      1. In-process warm cache (module-level ``_BM25``) — fastest, no I/O.
      2. On-disk cache — mmap'd load, ~5-10 s for 511k docs.
      3. Cold build from corpus — ~300 s; saves the result for next time.
    """
    global _BM25, _BM25_IDS
    if _BM25 is not None:
        return _BM25, _BM25_IDS

    # Try on-disk cache first.
    cached_bm, cached_ids = _try_load_bm25(cfg)
    if cached_bm is not None:
        _BM25, _BM25_IDS = cached_bm, cached_ids
        return _BM25, _BM25_IDS

    # Cold build.
    import bm25s
    t0 = time.time()
    texts: list[str] = []
    ids: list[str] = []
    for fp in sorted(Path(cfg.corpus_dir).rglob("*.txt")):
        ids.append(fp.relative_to(cfg.corpus_dir).as_posix())
        texts.append(fp.read_text(encoding="utf-8", errors="replace"))
    tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    bm25 = bm25s.BM25(method="lucene", k1=cfg.bm25_k1, b=cfg.bm25_b)
    bm25.index(tokens, show_progress=False)
    print(f"[retrieval] BM25 index built: {len(ids):,} docs in {time.time()-t0:.1f}s",
          flush=True)
    _BM25, _BM25_IDS = bm25, ids
    _save_bm25(bm25, ids, cfg)
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
