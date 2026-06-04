"""
Bi-Encoder (BAAI/bge-m3) and Cross-Encoder manager.
Handles model loading, batching, and inference with GPU support.
Includes OOM recovery: auto-reduces batch size on CUDA OOM.
"""
import gc
import logging
import threading
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Singleton model caches (thread-safe)
_bi_encoder = None
_cross_encoder = None
_bi_encoder_lock = threading.Lock()
_cross_encoder_lock = threading.Lock()


def _clear_gpu():
    """Force-clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_bi_encoder():
    """Lazy-load and return the Bi-Encoder model (BAAI/bge-m3)."""
    global _bi_encoder
    if _bi_encoder is None:
        with _bi_encoder_lock:
            if _bi_encoder is None:
                from sentence_transformers import SentenceTransformer
                from app.core.config import settings
                logger.info(f"Loading Bi-Encoder: {settings.BI_ENCODER_MODEL}")
                _bi_encoder = SentenceTransformer(
                    settings.BI_ENCODER_MODEL,
                    device=settings.GPU_DEVICE if settings.USE_GPU else "cpu",
                )
                # Reduce memory: move to half precision if on GPU
                if settings.USE_GPU:
                    _bi_encoder.half()
                    logger.info("Bi-Encoder using FP16 for memory efficiency.")
                logger.info("Bi-Encoder loaded successfully.")
    return _bi_encoder


def get_cross_encoder():
    """Lazy-load and return the Cross-Encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                from sentence_transformers import CrossEncoder
                from app.core.config import settings
                logger.info(f"Loading Cross-Encoder: {settings.CROSS_ENCODER_MODEL}")
                _cross_encoder = CrossEncoder(
                    settings.CROSS_ENCODER_MODEL,
                    device=settings.GPU_DEVICE if settings.USE_GPU else "cpu",
                )
                logger.info("Cross-Encoder loaded successfully.")
    return _cross_encoder


def encode_documents(texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
    """
    Encode a list of document texts into embeddings using the Bi-Encoder.
    Includes OOM recovery: auto-reduces batch size on CUDA OOM.

    Args:
        texts: List of document texts to encode.
        batch_size: Override the default batch size.

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    from app.core.config import settings

    model = get_bi_encoder()
    bs = batch_size or settings.EMBEDDING_BATCH_SIZE

    # Truncate long texts to save memory
    truncated_texts = [t[:2000] for t in texts]

    logger.info(f"Encoding {len(truncated_texts)} documents with batch_size={bs}")

    # Try encoding with OOM recovery
    max_retries = 3
    for attempt in range(max_retries):
        try:
            embeddings = model.encode(
                truncated_texts,
                batch_size=bs,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.array(embeddings, dtype=np.float32)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"CUDA OOM on attempt {attempt+1}/{max_retries}, reducing batch size from {bs} to {bs//2}")
            _clear_gpu()
            bs = max(1, bs // 2)
            if bs == 1:
                logger.error("Batch size reduced to 1, still OOM. Falling back to CPU.")
                model = get_bi_encoder()
                model.to("cpu")
                embeddings = model.encode(
                    truncated_texts,
                    batch_size=8,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                # Move back to GPU for future use
                from app.core.config import settings
                if settings.USE_GPU:
                    model.to(settings.GPU_DEVICE)
                return np.array(embeddings, dtype=np.float32)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"CUDA OOM (RuntimeError) on attempt {attempt+1}, reducing batch size")
                _clear_gpu()
                bs = max(1, bs // 2)
            else:
                raise

    raise RuntimeError("Failed to encode documents after OOM recovery attempts")


def cross_encoder_score(pairs: List[List[str]]) -> np.ndarray:
    """
    Score query-document pairs using the Cross-Encoder.

    Args:
        pairs: List of [query, document_text] pairs.

    Returns:
        numpy array of scores.
    """
    model = get_cross_encoder()

    # Truncate long texts in pairs
    truncated_pairs = [[p[0], p[1][:512]] for p in pairs]

    try:
        scores = model.predict(truncated_pairs, show_progress_bar=False, batch_size=32)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM in Cross-Encoder, retrying with smaller batch")
        _clear_gpu()
        scores = model.predict(truncated_pairs, show_progress_bar=False, batch_size=8)

    return np.array(scores, dtype=np.float32)
