"""
c-TF-IDF (Class-based TF-IDF) implementation for keyword extraction.
Treats each cluster as a single "class" and computes TF-IDF across clusters.
"""
import logging
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def compute_ctfidf(
    cluster_texts: Dict[str, str],
    top_k: int = 10,
) -> Dict[str, List[str]]:
    """
    Compute c-TF-IDF keywords for each cluster.

    Args:
        cluster_texts: Mapping of cluster_id -> concatenated document text for that cluster.
        top_k: Number of top keywords to extract per cluster.

    Returns:
        Mapping of cluster_id -> list of top-k keywords.
    """
    if not cluster_texts:
        return {}

    cluster_ids = list(cluster_texts.keys())
    documents = [cluster_texts[cid] for cid in cluster_ids]

    # Step 1: Create count matrix (clusters × terms)
    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
        )
        count_matrix = vectorizer.fit_transform(documents)  # shape: (n_clusters, n_terms)
    except ValueError as e:
        logger.warning(f"CountVectorizer failed: {e}. Returning empty keywords.")
        return {cid: [] for cid in cluster_ids}

    feature_names = vectorizer.get_feature_names_out()

    # Step 2: Compute TF (within each cluster)
    # TF = count of term in cluster / total terms in cluster
    cluster_sizes = np.array(count_matrix.sum(axis=1)).flatten()
    cluster_sizes[cluster_sizes == 0] = 1  # avoid division by zero
    tf = count_matrix.toarray().astype(np.float32) / cluster_sizes[:, np.newaxis]

    # Step 3: Compute IDF (across clusters)
    # IDF = log(1 + (total_clusters / (1 + n_clusters_containing_term)))
    n_clusters = len(cluster_ids)
    n_clusters_with_term = np.array(
        (count_matrix > 0).sum(axis=0)
    ).flatten().astype(np.float32)
    idf = np.log(1 + (n_clusters / (1 + n_clusters_with_term)))

    # Step 4: c-TF-IDF = TF * IDF
    ctfidf = tf * idf[np.newaxis, :]

    # Step 5: Extract top-k keywords per cluster
    result = {}
    for i, cid in enumerate(cluster_ids):
        row = ctfidf[i]
        top_indices = row.argsort()[::-1][:top_k]
        keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        result[cid] = keywords[:top_k]

    return result
