#!/usr/bin/env python3
"""
Build FAISS index from all document embeddings in the database.
This is the most critical step for efficient retrieval on 512K documents.
Also builds the cluster tree for leaf-based routing.
"""
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, "/app/backend")

from app.core.database import SessionLocal
from app.models.schemas import Document, ClusterNode, IngestionJob


def build_faiss_index():
    """Build FAISS ANN index from all document embeddings."""
    import numpy as np
    from app.ml.faiss_index import build_index, save_index, get_index_stats

    db = SessionLocal()
    try:
        logger.info("Fetching document embeddings from database...")
        start = time.time()

        # Fetch in batches to avoid OOM
        batch_size = 50000
        all_embeddings = []
        all_doc_ids = []

        offset = 0
        while True:
            docs = db.query(Document).filter(
                Document.embedding.isnot(None)
            ).offset(offset).limit(batch_size).all()

            if not docs:
                break

            for doc in docs:
                all_doc_ids.append(doc.id)
                all_embeddings.append(doc.embedding)

            offset += len(docs)
            logger.info(f"  Fetched {len(all_doc_ids)} embeddings so far...")

            if len(docs) < batch_size:
                break

        if not all_embeddings:
            logger.error("No document embeddings found!")
            return False

        embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Loaded {len(all_doc_ids)} embeddings in {time.time()-start:.1f}s, shape={embeddings.shape}")

        # Build FAISS index
        logger.info("Building FAISS index...")
        build_start = time.time()
        build_index(embeddings, all_doc_ids)
        save_index()
        logger.info(f"FAISS index built in {time.time()-build_start:.1f}s")

        stats = get_index_stats()
        logger.info(f"FAISS index stats: {stats}")

        return True

    finally:
        db.close()


def verify_faiss_search():
    """Verify FAISS search works with a sample query."""
    from app.ml.embedding import get_bi_encoder
    from app.ml.faiss_index import search

    bi_encoder = get_bi_encoder()
    query_emb = bi_encoder.encode(
        ["What are the default size limits for file uploads?"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    results = search(query_emb, top_k=10)
    logger.info(f"Sample search results (top-10):")
    for doc_id, score in results[:5]:
        logger.info(f"  {doc_id[:40]}... score={score:.4f}")

    return len(results) > 0


def build_cluster_tree():
    """Build the cluster tree from all documents."""
    db = SessionLocal()
    try:
        # Clear existing tree
        db.query(ClusterNode).delete()
        db.query(IngestionJob).delete()
        db.commit()

        all_docs = db.query(Document).all()
        doc_ids = [d.id for d in all_docs]
        logger.info(f"Building cluster tree for {len(doc_ids)} documents...")

        # Queue the tree build
        from app.tasks.celery_app import celery_app
        celery_app.send_task(
            "build_cluster_tree",
            args=["root", doc_ids, 0],
            queue="gpu",
        )

        logger.info("Tree build task queued. Start celery worker to process.")
        return True

    finally:
        db.close()


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "faiss"

    if action == "faiss":
        success = build_faiss_index()
        if success:
            verify_faiss_search()
    elif action == "tree":
        build_cluster_tree()
    elif action == "verify":
        verify_faiss_search()
    elif action == "all":
        build_faiss_index()
        verify_faiss_search()
        build_cluster_tree()
    else:
        print(f"Usage: {sys.argv[0]} [faiss|tree|verify|all]")
