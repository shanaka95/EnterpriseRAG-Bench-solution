#!/usr/bin/env python3
"""
Bulk embed remaining documents using GPU 1 (cuda:1) to avoid thermal throttling.
"""
import os
import sys
import time
import logging

os.environ["GPU_DEVICE"] = "cuda:1"
os.environ["USE_GPU"] = "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/app/backend")

from app.core.database import SessionLocal
from app.models.schemas import Document, ClusterNode, IngestionJob
from app.ml.embedding import encode_documents
from sqlalchemy import func


def bulk_embed():
    db = SessionLocal()
    try:
        total = db.query(func.count(Document.id)).scalar()
        with_emb = db.query(func.count(Document.id)).filter(Document.embedding.isnot(None)).scalar()
        without = total - with_emb
        logger.info(f"Total docs: {total}, With embeddings: {with_emb}, Without: {without}")

        if without == 0:
            logger.info("All documents already have embeddings.")
            return True

        logger.info("Fetching doc IDs without embeddings...")
        doc_ids_no_emb = [row[0] for row in db.query(Document.id).filter(
            Document.embedding.is_(None)
        ).all()]
        logger.info(f"Fetched {len(doc_ids_no_emb)} doc IDs.")

        chunk_size = 500
        processed = 0
        start_time = time.time()

        for i in range(0, len(doc_ids_no_emb), chunk_size):
            chunk_ids = doc_ids_no_emb[i:i + chunk_size]

            docs = db.query(Document).filter(Document.id.in_(chunk_ids)).all()
            doc_map = {d.id: d for d in docs}
            texts = []
            ordered_docs = []
            for did in chunk_ids:
                doc = doc_map.get(did)
                if doc:
                    texts.append(doc.text)
                    ordered_docs.append(doc)

            if not texts:
                continue

            embeddings = encode_documents(texts, batch_size=64)

            for j, doc in enumerate(ordered_docs):
                doc.embedding = embeddings[j].tolist()

            db.commit()
            processed += len(ordered_docs)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = len(doc_ids_no_emb) - processed
            eta = remaining / rate if rate > 0 else 0
            logger.info(
                f"Embedded {processed}/{len(doc_ids_no_emb)} docs "
                f"({rate:.1f} docs/sec, ETA: {eta/60:.1f}min)"
            )

        logger.info(f"Bulk embedding complete. Total processed: {processed}")
        return True
    except Exception as e:
        logger.error(f"Bulk embedding failed: {e}", exc_info=True)
        return False
    finally:
        db.close()


def build_faiss():
    from app.ml.faiss_index import build_index, save_index, get_index_stats
    import numpy as np

    db = SessionLocal()
    try:
        logger.info("Fetching document embeddings for FAISS index...")
        start = time.time()

        batch_size = 50000
        all_embeddings = []
        all_doc_ids = []
        offset = 0

        while True:
            docs = db.query(Document.id, Document.embedding).filter(
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
            logger.error("No embeddings found!")
            return False

        embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Loaded {len(all_doc_ids)} embeddings in {time.time()-start:.1f}s, shape={embeddings.shape}")

        logger.info("Building FAISS index...")
        build_start = time.time()
        build_index(embeddings, all_doc_ids)
        save_index()
        logger.info(f"FAISS index built in {time.time()-build_start:.1f}s")
        logger.info(f"FAISS stats: {get_index_stats()}")
        return True
    finally:
        db.close()


def build_tree():
    db = SessionLocal()
    try:
        logger.info("Clearing old cluster tree...")
        db.query(ClusterNode).delete()
        db.query(IngestionJob).delete()
        db.commit()

        all_doc_ids = [row[0] for row in db.query(Document.id).all()]
        logger.info(f"Building cluster tree for {len(all_doc_ids)} documents...")

        from app.tasks.celery_app import celery_app
        celery_app.send_task(
            "build_cluster_tree",
            args=["root", all_doc_ids, 0],
            queue="gpu",
        )
        logger.info("Tree build task queued.")
        return True
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BULK EMBED ON GPU 1 + FAISS + TREE BUILD")
    logger.info("=" * 60)

    if not bulk_embed():
        sys.exit(1)

    if not build_faiss():
        sys.exit(1)

    if not build_tree():
        sys.exit(1)

    logger.info("All steps complete.")
