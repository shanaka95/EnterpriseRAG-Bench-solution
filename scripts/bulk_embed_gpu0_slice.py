#!/usr/bin/env python3
"""
Bulk embed a slice of remaining documents using GPU 0 (cuda:0).
Run in parallel with bulk_embed_gpu1.py to double throughput.
"""
import os
import sys
import time
import logging

os.environ["GPU_DEVICE"] = "cuda:0"
os.environ["USE_GPU"] = "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/app/backend")

from app.core.database import SessionLocal
from app.models.schemas import Document
from app.ml.embedding import encode_documents
from sqlalchemy import func


def bulk_embed_slice(slice_idx=0, num_slices=2):
    db = SessionLocal()
    try:
        total = db.query(func.count(Document.id)).scalar()
        with_emb = db.query(func.count(Document.id)).filter(Document.embedding.isnot(None)).scalar()
        without = total - with_emb
        logger.info(f"[Slice {slice_idx}/{num_slices}] Total docs: {total}, With embeddings: {with_emb}, Without: {without}")

        if without == 0:
            logger.info("All documents already have embeddings.")
            return True

        logger.info("Fetching doc IDs without embeddings...")
        doc_ids_no_emb = [row[0] for row in db.query(Document.id).filter(
            Document.embedding.is_(None)
        ).all()]
        logger.info(f"Fetched {len(doc_ids_no_emb)} doc IDs.")

        # Take every num_slices-th element starting at slice_idx
        my_ids = doc_ids_no_emb[slice_idx::num_slices]
        logger.info(f"[Slice {slice_idx}] Processing {len(my_ids)} docs")

        chunk_size = 500
        processed = 0
        start_time = time.time()

        for i in range(0, len(my_ids), chunk_size):
            chunk_ids = my_ids[i:i + chunk_size]

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
            remaining = len(my_ids) - processed
            eta = remaining / rate if rate > 0 else 0
            logger.info(
                f"[Slice {slice_idx}] Embedded {processed}/{len(my_ids)} docs "
                f"({rate:.1f} docs/sec, ETA: {eta/60:.1f}min)"
            )

        logger.info(f"[Slice {slice_idx}] Complete. Total processed: {processed}")
        return True
    except Exception as e:
        logger.error(f"[Slice {slice_idx}] Failed: {e}", exc_info=True)
        return False
    finally:
        db.close()


if __name__ == "__main__":
    slice_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_slices = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if not bulk_embed_slice(slice_idx, num_slices):
        sys.exit(1)
    logger.info("Slice complete.")
