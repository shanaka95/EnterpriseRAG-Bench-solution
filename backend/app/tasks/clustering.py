"""
Celery task: Recursive Cluster Tree Builder.

This is the heart of the system. It:
1. Embeds documents using Bi-Encoder
2. Reduces dimensions with UMAP
3. Clusters with HDBSCAN (soft assignment)
4. Extracts c-TF-IDF keywords
5. Refines borderline docs with Cross-Encoder
6. Recursively processes sub-clusters > MIN_DOCS_FOR_SPLIT
7. Broadcasts tree updates via Redis pub/sub
"""
import logging
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
from celery import shared_task, current_task
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.schemas import ClusterNode, Document, IngestionJob
from app.ml.embedding import encode_documents, cross_encoder_score
from app.ml.clusterer import reduce_dimensions, cluster_with_soft_assignment
from app.ml.ctfidf import compute_ctfidf

logger = logging.getLogger(__name__)


_redis_pool = None


def _get_redis():
    """Lazy-init a shared Redis connection for broadcasts."""
    global _redis_pool
    if _redis_pool is None:
        import redis as redis_lib
        _redis_pool = redis_lib.Redis.from_url(settings.REDIS_URL)
    return _redis_pool


def _broadcast_event(event_type: str, data: dict):
    """Publish a tree update event to Redis for SSE broadcasting."""
    try:
        r = _get_redis()
        payload = json.dumps({
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        })
        r.publish("cluster_tree_updates", payload)
    except Exception as e:
        logger.warning(f"Failed to broadcast event: {e}")


# PostgreSQL parameter limit safety
_PG_MAX_PARAMS = 50000


def _chunked_doc_query(db: Session, doc_ids: List[str]):
    """Fetch documents in chunks to stay under PostgreSQL's parameter limit."""
    results = []
    for i in range(0, len(doc_ids), _PG_MAX_PARAMS):
        chunk = doc_ids[i : i + _PG_MAX_PARAMS]
        results.extend(db.query(Document).filter(Document.id.in_(chunk)).all())
    return results


def _get_doc_texts_and_embeddings(doc_ids: List[str], db: Session):
    """
    Retrieve document texts and pre-computed embeddings from DB.
    If embeddings are missing, compute them on-the-fly.
    """
    # Chunked query: 512K doc_ids would exceed PostgreSQL's 65K parameter limit
    documents = _chunked_doc_query(db, doc_ids)
    doc_map = {d.id: d for d in documents}

    texts = []
    embeddings = []
    found_doc_ids = []

    for doc_id in doc_ids:
        doc = doc_map.get(doc_id)
        if doc is None:
            logger.warning(f"Document {doc_id} not found in DB, skipping.")
            continue
        found_doc_ids.append(doc_id)
        texts.append(doc.text)
        if doc.embedding is not None and len(doc.embedding) > 0:
            embeddings.append(np.array(doc.embedding, dtype=np.float32))
        else:
            embeddings.append(None)

    # If any embeddings missing, compute them all at once for consistency
    missing_count = sum(1 for e in embeddings if e is None)
    if missing_count > 0:
        logger.info(f"Computing embeddings for {len(texts)} documents ({missing_count} missing)...")
        all_embeddings = encode_documents(texts)
        # Save embeddings back to DB
        for i, doc_id in enumerate(found_doc_ids):
            doc = doc_map.get(doc_id)
            if doc and (doc.embedding is None or len(doc.embedding) == 0):
                doc.embedding = all_embeddings[i].tolist()
        db.commit()
        return texts, all_embeddings

    return texts, np.array(embeddings, dtype=np.float32)


def _compute_medoid(embeddings: np.ndarray, indices: List[int]) -> int:
    """Find the medoid (most central point) of a cluster."""
    cluster_embs = embeddings[indices]
    centroid = cluster_embs.mean(axis=0)
    distances = np.linalg.norm(cluster_embs - centroid, axis=1)
    medoid_local = int(np.argmin(distances))
    return indices[medoid_local]


@shared_task(bind=True, name="build_cluster_tree", max_retries=2, time_limit=14400)
def build_cluster_tree(self, node_id: str, doc_ids: List[str], depth: int = 0):
    """
    Recursively build the cluster tree for a given set of documents.

    Args:
        node_id: UUID of the parent ClusterNode (or 'root' for initial call).
        doc_ids: List of document IDs to cluster.
        depth: Current depth in the tree.
    """
    logger.info(
        f"[Depth {depth}] Building cluster tree for node={node_id} with {len(doc_ids)} docs"
    )

    db = SessionLocal()
    try:
        # ── Step 1: Get or create the parent node ──
        if node_id == "root":
            # Create root node
            root = ClusterNode(
                id=uuid.uuid4(),
                parent_id=None,
                doc_count=len(doc_ids),
                is_leaf=False,
                doc_ids=None,
                keywords=[],
                depth=0,
                medoid_doc_id=None,
            )
            db.add(root)
            db.commit()
            db.refresh(root)
            parent_id = root.id
            node_id = str(parent_id)
            _broadcast_event("node_created", root.to_dict())
        else:
            parent_id = uuid.UUID(node_id)

        parent_node = db.query(ClusterNode).filter(ClusterNode.id == parent_id).first()
        if not parent_node:
            logger.error(f"Parent node {node_id} not found!")
            return

        # ── Step 2: Check if we should split ──
        max_depth = getattr(settings, 'MAX_TREE_DEPTH', 8)
        if len(doc_ids) <= settings.MIN_DOCS_FOR_SPLIT or depth >= max_depth:
            parent_node.is_leaf = True
            parent_node.doc_ids = doc_ids
            parent_node.doc_count = len(doc_ids)
            db.commit()
            db.refresh(parent_node)
            _broadcast_event("node_updated", parent_node.to_dict())
            logger.info(f"[Depth {depth}] Node {node_id} is leaf with {len(doc_ids)} docs (depth_limit={depth >= max_depth}).")
            return

        # ── Step 3: Get texts and embeddings ──
        texts, embeddings = _get_doc_texts_and_embeddings(doc_ids, db)

        if len(texts) < settings.MIN_DOCS_FOR_SPLIT:
            parent_node.is_leaf = True
            parent_node.doc_ids = doc_ids
            parent_node.doc_count = len(doc_ids)
            db.commit()
            _broadcast_event("node_updated", parent_node.to_dict())
            return

        # ── Step 4: Dimensionality Reduction (UMAP) ──
        logger.info(f"[Depth {depth}] Running UMAP on {len(embeddings)} embeddings...")
        reduced = reduce_dimensions(
            embeddings,
            n_neighbors=settings.UMAP_N_NEIGHBORS,
            min_dist=settings.UMAP_MIN_DIST,
            n_components=settings.UMAP_N_COMPONENTS,
            metric=settings.UMAP_METRIC,
        )

        # ── Step 5: HDBSCAN Clustering with Soft Assignment ──
        logger.info(f"[Depth {depth}] Running HDBSCAN clustering...")
        cluster_to_docs, soft_probs, labels = cluster_with_soft_assignment(
            reduced,
            min_cluster_size=settings.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=settings.HDBSCAN_MIN_SAMPLES,
            metric=settings.HDBSCAN_METRIC,
            soft_threshold=settings.SOFT_ASSIGNMENT_THRESHOLD,
        )

        # Check if HDBSCAN found any clusters (not just noise)
        if len(cluster_to_docs) == 0:
            logger.warning(f"[Depth {depth}] HDBSCAN found no clusters. Making leaf node.")
            parent_node.is_leaf = True
            parent_node.doc_ids = doc_ids
            parent_node.doc_count = len(doc_ids)
            db.commit()
            _broadcast_event("node_updated", parent_node.to_dict())
            return

        # ── Step 6: Handle noise points (-1 label) ──
        # Assign noise points to nearest cluster
        noise_indices = np.where(labels == -1)[0]
        if len(noise_indices) > 0 and len(cluster_to_docs) > 0:
            # Find nearest cluster for each noise point
            unique_labels = sorted(cluster_to_docs.keys())
            centroids = {}
            for cl in unique_labels:
                cl_set = set(cluster_to_docs[cl])
                mask = np.array([i in cl_set for i in range(len(reduced))])
                centroids[cl] = reduced[mask].mean(axis=0)

            for idx in noise_indices:
                min_dist = float('inf')
                best_cluster = list(unique_labels)[0]
                for cl, centroid in centroids.items():
                    dist = np.linalg.norm(reduced[idx] - centroid)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = cl
                cluster_to_docs[best_cluster].append(int(idx))

        # ── Step 7: c-TF-IDF Keyword Extraction ──
        logger.info(f"[Depth {depth}] Computing c-TF-IDF keywords...")
        cluster_texts = {}
        for cluster_label, doc_indices in cluster_to_docs.items():
            cluster_text = " ".join(texts[i] for i in doc_indices if i < len(texts))
            cluster_texts[str(cluster_label)] = cluster_text

        keywords_by_cluster = compute_ctfidf(
            cluster_texts, top_k=settings.CTFIDF_TOP_K
        )

        # ── Step 8: Create child nodes and apply Cross-Encoder refinement ──
        child_nodes = []
        for cluster_label, doc_indices in cluster_to_docs.items():
            # Get actual doc IDs for these indices
            actual_doc_ids = [doc_ids[i] for i in doc_indices if i < len(doc_ids)]

            if not actual_doc_ids:
                continue

            # Compute medoid
            medoid_idx = _compute_medoid(embeddings, doc_indices)
            medoid_doc_id = doc_ids[medoid_idx] if medoid_idx < len(doc_ids) else actual_doc_ids[0]

            # Get medoid text
            medoid_text = texts[medoid_idx] if medoid_idx < len(texts) else ""

            # Cross-Encoder refinement for borderline documents
            borderline_indices = []
            threshold = settings.SOFT_ASSIGNMENT_THRESHOLD
            margin = settings.BORDERLINE_MARGIN

            if soft_probs.shape[0] > 0 and soft_probs.shape[1] > cluster_label:
                for idx in doc_indices:
                    if idx < soft_probs.shape[0]:
                        prob = soft_probs[idx, cluster_label]
                        if threshold - margin <= prob <= threshold + margin:
                            borderline_indices.append(idx)

            if borderline_indices and len(cluster_to_docs) > 1:
                logger.info(
                    f"[Depth {depth}] Cross-Encoder refining {len(borderline_indices)} "
                    f"borderline docs in cluster {cluster_label}"
                )
                # Get medoids of all clusters for comparison
                medoid_texts = []
                medoid_labels = []
                for cl_label, cl_indices in cluster_to_docs.items():
                    cl_medoid_idx = _compute_medoid(embeddings, cl_indices)
                    medoid_texts.append(texts[cl_medoid_idx] if cl_medoid_idx < len(texts) else "")
                    medoid_labels.append(cl_label)

                # Score borderline docs against all medoids (batch for efficiency)
                borderline_set = set(borderline_indices)
                to_move = {}  # target_cluster -> set of indices to move
                for idx in borderline_indices:
                    if idx >= len(texts):
                        borderline_set.discard(idx)
                        continue
                    pairs = [[texts[idx], mt] for mt in medoid_texts]
                    scores = cross_encoder_score(pairs)
                    best_cluster_idx = int(np.argmax(scores))
                    best_cluster = medoid_labels[best_cluster_idx]

                    if best_cluster != cluster_label:
                        to_move.setdefault(best_cluster, set()).add(idx)
                        borderline_set.discard(idx)

                # Apply moves in O(1) per doc using set difference
                if to_move:
                    current_set = set(doc_indices)
                    current_set -= set(borderline_indices)  # remove all borderline first
                    current_set |= borderline_set           # add back those that stayed
                    for target_cluster, moved_indices in to_move.items():
                        current_set -= moved_indices
                        cluster_to_docs[target_cluster] = list(
                            set(cluster_to_docs[target_cluster]) | moved_indices
                        )
                    doc_indices = list(current_set)
                    cluster_to_docs[cluster_label] = doc_indices
                    actual_doc_ids = [doc_ids[i] for i in doc_indices if i < len(doc_ids)]

            # Create child ClusterNode
            child_node = ClusterNode(
                id=uuid.uuid4(),
                parent_id=parent_id,
                medoid_doc_id=medoid_doc_id,
                doc_count=len(actual_doc_ids),
                keywords=keywords_by_cluster.get(str(cluster_label), []),
                is_leaf=len(actual_doc_ids) <= settings.MIN_DOCS_FOR_SPLIT,
                doc_ids=actual_doc_ids if len(actual_doc_ids) <= settings.MIN_DOCS_FOR_SPLIT else None,
                depth=depth + 1,
            )
            db.add(child_node)
            child_nodes.append((child_node, actual_doc_ids))
            _broadcast_event("node_created", child_node.to_dict())

        db.commit()

        # Update parent node
        parent_node.is_leaf = False
        parent_node.doc_ids = None
        db.commit()
        _broadcast_event("node_updated", parent_node.to_dict())

        # ── Step 9: Recursively process non-leaf children ──
        for child_node, child_doc_ids in child_nodes:
            if not child_node.is_leaf and len(child_doc_ids) > settings.MIN_DOCS_FOR_SPLIT:
                logger.info(
                    f"[Depth {depth+1}] Spawning recursive task for node {child_node.id} "
                    f"with {len(child_doc_ids)} docs"
                )
                # Chain the next task
                build_cluster_tree.delay(
                    node_id=str(child_node.id),
                    doc_ids=child_doc_ids,
                    depth=depth + 1,
                )
            else:
                _broadcast_event("node_completed", child_node.to_dict())

        logger.info(f"[Depth {depth}] Cluster tree building complete for node={node_id}")

    except Exception as e:
        logger.error(f"[Depth {depth}] Error building cluster tree: {e}", exc_info=True)
        # Update parent node with error
        try:
            parent = db.query(ClusterNode).filter(ClusterNode.id == uuid.UUID(node_id)).first()
            if parent:
                parent.is_leaf = True
                db.commit()
        except Exception:
            pass
        raise
    finally:
        db.close()


@shared_task(name="ingest_and_build")
def ingest_and_build(job_id: str, doc_ids: List[str]):
    """
    Start the ingestion pipeline: embed documents, then build the cluster tree.
    """
    db = SessionLocal()
    try:
        job = db.query(IngestionJob).filter(IngestionJob.id == uuid.UUID(job_id)).first()
        if not job:
            logger.error(f"Ingestion job {job_id} not found!")
            return

        job.status = "processing"
        db.commit()

        # Build the cluster tree starting from root
        build_cluster_tree.delay(
            node_id="root",
            doc_ids=doc_ids,
            depth=0,
        )

        job.status = "tree_building"
        job.processed_docs = len(doc_ids)
        db.commit()
        _broadcast_event("ingestion_status", job.to_dict())

    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        try:
            job = db.query(IngestionJob).filter(IngestionJob.id == uuid.UUID(job_id)).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
