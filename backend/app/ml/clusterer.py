"""
GPU-accelerated clustering module using RAPIDS cuML (with CPU fallback).
Implements UMAP dimensionality reduction and HDBSCAN soft-assignment clustering.
"""
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Track whether we can use GPU-accelerated cuML
_CUML_AVAILABLE = False

def _check_cuml():
    """Check if cuML is available."""
    global _CUML_AVAILABLE
    try:
        import cuml
        _CUML_AVAILABLE = True
        logger.info("cuML (RAPIDS) is available - using GPU acceleration.")
    except ImportError:
        _CUML_AVAILABLE = False
        logger.info("cuML not available - falling back to CPU (umap-learn, hdbscan).")


_check_cuml()


def reduce_dimensions(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 32,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Reduce embedding dimensionality using UMAP (GPU if available, else CPU).

    Args:
        embeddings: numpy array of shape (n_docs, embedding_dim)
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        n_components: Target number of dimensions.
        metric: Distance metric for UMAP.

    Returns:
        Reduced embeddings of shape (n_docs, n_components)
    """
    if len(embeddings) <= n_neighbors:
        logger.warning(
            f"Only {len(embeddings)} samples, too few for UMAP (n_neighbors={n_neighbors}). "
            "Returning original embeddings truncated/padded."
        )
        if embeddings.shape[1] >= n_components:
            return embeddings[:, :n_components]
        else:
            pad = np.zeros((len(embeddings), n_components - embeddings.shape[1]), dtype=np.float32)
            return np.hstack([embeddings, pad])

    if _CUML_AVAILABLE:
        from cuml.manifold import UMAP
        logger.info(f"Running cuML UMAP: {embeddings.shape} -> ({len(embeddings)}, {n_components})")
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)
        return np.array(reduced, dtype=np.float32)
    else:
        import umap
        logger.info(f"Running CPU UMAP: {embeddings.shape} -> ({len(embeddings)}, {n_components})")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)
        return np.array(reduced, dtype=np.float32)


def cluster_with_soft_assignment(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
    soft_threshold: float = 0.2,
) -> Tuple[Dict[int, List[int]], np.ndarray, np.ndarray]:
    """
    Cluster embeddings using HDBSCAN with soft assignment.

    Args:
        reduced_embeddings: numpy array of shape (n_docs, n_components)
        min_cluster_size: HDBSCAN min_cluster_size.
        min_samples: HDBSCAN min_samples.
        metric: Distance metric.
        soft_threshold: Minimum probability for soft assignment.

    Returns:
        Tuple of:
        - cluster_to_docs: mapping of cluster_label -> list of doc indices
        - probabilities: soft assignment probability matrix (n_docs, n_clusters)
        - labels: hard cluster labels (-1 for noise)
    """
    n_docs = len(reduced_embeddings)

    if n_docs < min_cluster_size * 2:
        logger.warning(
            f"Too few documents ({n_docs}) for meaningful clustering. "
            "Assigning all to single cluster."
        )
        return {0: list(range(n_docs))}, np.ones((n_docs, 1), dtype=np.float32), np.zeros(n_docs, dtype=int)

    if _CUML_AVAILABLE:
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        logger.info(f"Running cuML HDBSCAN on {n_docs} documents")
        clusterer = cuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(reduced_embeddings)
        labels = np.array(labels, dtype=int)

        # Try to get soft assignment probabilities
        try:
            # cuML HDBSCAN probabilities
            probabilities = clusterer.probabilities_
            probabilities = np.array(probabilities, dtype=np.float32)

            # For soft assignment, we need all_points_membership_vectors
            # cuML may not support this directly, so we fall back to computing
            # soft assignments from the clusterer internals
            soft_probs = _compute_soft_assignments_cuml(clusterer, reduced_embeddings, labels)
        except Exception as e:
            logger.warning(f"cuML soft assignment failed: {e}. Using hard assignment.")
            soft_probs = _hard_to_soft(labels)
    else:
        import hdbscan
        logger.info(f"Running CPU HDBSCAN on {n_docs} documents")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(reduced_embeddings)
        labels = np.array(labels, dtype=int)

        # Get soft assignment probabilities using all_points_membership_vectors
        try:
            soft_probs = hdbscan.all_points_membership_vectors(clusterer)
            soft_probs = np.array(soft_probs, dtype=np.float32)
        except Exception as e:
            logger.warning(f"HDBSCAN soft assignment failed: {e}. Using hard assignment.")
            soft_probs = _hard_to_soft(labels)

    # Build cluster_to_docs mapping with conservative soft assignment.
    # Strategy: each doc goes to its PRIMARY cluster (highest probability).
    # Multi-assignment only if a secondary cluster has probability > threshold
    # AND is within 20% of the primary cluster's probability.
    # This prevents fanout while preserving meaningful multi-membership.
    unique_labels = sorted(set(labels[labels >= 0]))
    n_clusters = len(unique_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    cluster_to_docs = {cl: [] for cl in unique_labels}

    # First pass: hard assignments
    for i, label in enumerate(labels):
        if label >= 0 and label in label_map:
            cluster_to_docs[label].append(i)

    # Second pass: selective soft multi-assignment
    if soft_probs.shape[1] >= n_clusters and n_clusters > 1:
        # Remap soft_probs columns to our label_map indices if needed
        for doc_idx in range(len(labels)):
            # Get this doc's probabilities across all clusters
            doc_probs = soft_probs[doc_idx]
            # Map to our unique labels
            cluster_probs = []
            for cl in unique_labels:
                col_idx = label_map[cl]
                if col_idx < doc_probs.shape[0]:
                    cluster_probs.append((cl, doc_probs[col_idx]))
                else:
                    cluster_probs.append((cl, 0.0))

            # Sort by probability descending
            cluster_probs.sort(key=lambda x: x[1], reverse=True)

            if len(cluster_probs) < 2:
                continue

            primary_cluster, primary_prob = cluster_probs[0]
            secondary_cluster, secondary_prob = cluster_probs[1]

            # Multi-assign only if:
            # 1. Secondary prob > threshold
            # 2. Secondary prob >= 80% of primary prob (near-equal relevance)
            # 3. Doc is not already in the secondary cluster
            multi_assign_margin = 0.8  # Secondary must be >= 80% of primary
            if (secondary_prob > soft_threshold and
                secondary_prob >= primary_prob * multi_assign_margin and
                doc_idx not in cluster_to_docs[secondary_cluster]):
                cluster_to_docs[secondary_cluster].append(doc_idx)

    return cluster_to_docs, soft_probs, labels


def _hard_to_soft(labels: np.ndarray) -> np.ndarray:
    """Convert hard labels to soft probability matrix."""
    n_docs = len(labels)
    unique_labels = sorted(set(labels[labels >= 0]))
    n_clusters = len(unique_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}

    soft = np.zeros((n_docs, n_clusters), dtype=np.float32)
    for i, l in enumerate(labels):
        if l >= 0 and l in label_map:
            soft[i, label_map[l]] = 1.0

    return soft


def _compute_soft_assignments_cuml(clusterer, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute soft assignments for cuML HDBSCAN.
    Uses distance-based membership scoring as a fallback since
    cuML HDBSCAN doesn't directly expose all_points_membership_vectors.
    """
    unique_labels = sorted(set(labels[labels >= 0]))
    n_clusters = len(unique_labels)
    n_docs = len(embeddings)
    label_map = {l: i for i, l in enumerate(unique_labels)}

    soft = np.zeros((n_docs, n_clusters), dtype=np.float32)

    # Compute cluster centroids
    for cluster_label in unique_labels:
        mask = labels == cluster_label
        centroid = embeddings[mask].mean(axis=0)
        # Compute distance-based membership
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        max_dist = distances.max() + 1e-8
        membership = 1.0 - (distances / max_dist)
        soft[:, label_map[cluster_label]] = membership.astype(np.float32)

    # Normalize rows to sum to 1
    row_sums = soft.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    soft = soft / row_sums

    return soft
