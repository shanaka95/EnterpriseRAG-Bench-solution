"""
FAISS-based ANN index for fast Bi-Encoder retrieval on large datasets.
Replaces the full-scan cosine similarity approach for O(log N) retrieval.
"""
import os
import json
import logging
import threading
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

import faiss

logger = logging.getLogger(__name__)

# Singleton index
_index = None
_id_map = None  # Maps FAISS internal ID -> doc_id
_index_lock = threading.Lock()
INDEX_DIR = "/app/backend/faiss_index"

# Safety limits
MAX_TOP_K = 25000  # Prevent excessive memory use / slow queries


def _get_index_path():
    return os.path.join(INDEX_DIR, "index.faiss")


def _get_id_map_path():
    return os.path.join(INDEX_DIR, "id_map.json")


def build_index(embeddings: np.ndarray, doc_ids: List[str]) -> None:
    """
    Build a FAISS IVF index from document embeddings.

    Args:
        embeddings: numpy array of shape (N, dim)
        doc_ids: list of document IDs corresponding to each embedding
    """
    global _index, _id_map

    n, dim = embeddings.shape
    logger.info(f"Building FAISS index for {n} documents, dim={dim}")

    # Normalize embeddings for cosine similarity (inner product = cosine sim)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_normed = (embeddings / norms).astype(np.float32)

    # For 512K documents @ 1024 dims, IndexFlatIP uses ~2GB RAM and gives
    # EXACT search (100% index recall). This is critical for hitting 90%+
    # benchmark recall. Approximate indices (IVF, PQ) sacrifice recall.
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim after normalization
    index.add(embeddings_normed)

    # Store mapping
    id_map = {i: doc_id for i, doc_id in enumerate(doc_ids)}

    with _index_lock:
        _index = index
        _id_map = id_map

    logger.info(f"FAISS index built: {n} vectors, type={type(index).__name__}")


def save_index() -> None:
    """Save the FAISS index and ID map to disk."""
    global _index, _id_map

    if _index is None:
        logger.warning("No FAISS index to save")
        return

    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(_index, _get_index_path())
    with open(_get_id_map_path(), "w") as f:
        json.dump(_id_map, f)

    logger.info(f"FAISS index saved to {INDEX_DIR}")


def load_index() -> bool:
    """
    Load the FAISS index and ID map from disk.
    Returns True if successful, False otherwise.
    """
    global _index, _id_map

    if not os.path.exists(_get_index_path()):
        return False

    try:
        index = faiss.read_index(_get_index_path())
        with open(_get_id_map_path(), "r") as f:
            id_map = json.load(f)
        # Convert string keys back to int
        id_map = {int(k): v for k, v in id_map.items()}
        with _index_lock:
            _index = index
            _id_map = id_map
        logger.info(f"FAISS index loaded: {_index.ntotal} vectors")
        return True
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return False


def search(query_embedding: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
    """
    Search for the most similar documents using FAISS exact search.

    Args:
        query_embedding: Query embedding vector (1D or 2D with shape (1, dim))
        top_k: Number of results to return (clamped to [1, MAX_TOP_K])

    Returns:
        List of (doc_id, similarity_score) tuples, sorted by score descending
    """
    global _index, _id_map

    if _index is None:
        raise RuntimeError("FAISS index not built or loaded. Call build_index() or load_index() first.")

    # Validate top_k to prevent abuse / OOM
    top_k = max(1, min(int(top_k), MAX_TOP_K))

    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Normalize query for cosine similarity
    query_embedding = query_embedding.astype(np.float32)
    norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    query_embedding = query_embedding / norms

    # Search (IndexFlatIP is exact, no nprobe needed)
    with _index_lock:
        actual_k = min(top_k, _index.ntotal)
        scores, indices = _index.search(query_embedding, actual_k)

    results = []
    for i in range(actual_k):
        idx = int(indices[0][i])
        if idx < 0:
            continue  # FAISS returns -1 for missing entries
        doc_id = _id_map.get(idx)
        if doc_id is not None:
            results.append((doc_id, float(scores[0][i])))

    return results


def is_ready() -> bool:
    """Check if the FAISS index is loaded and ready for search."""
    return _index is not None and _id_map is not None


def get_index_stats() -> dict:
    """Get statistics about the FAISS index."""
    if _index is None:
        return {"status": "not_loaded"}
    return {
        "status": "ready",
        "ntotal": _index.ntotal,
        "dim": _index.d,
        "type": type(_index).__name__,
    }
