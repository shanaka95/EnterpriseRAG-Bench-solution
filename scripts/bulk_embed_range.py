#!/usr/bin/env python3
"""
Bulk embed a range of remaining documents on a specific GPU.
Run multiple instances in parallel on different GPUs.
"""
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/app/backend")

from app.core.database import SessionLocal
from app.models.schemas import Document
from app.ml.embedding import encode_documents
from sqlalchemy import func


def bulk_embed_range(gpu_id: str, start_idx: int, end_idx: int):
    os.environ["GPU_DEVICE"] = f"cuda:{gpu_id}"
    os.environ["USE_GPU"] = "true"

    db = SessionLocal()
    try:
        total = db.query(func.count(Document.id)).scalar()
        with_emb = db.query(func.count(Document.id)).filter(Document.embedding.isnot(None)).scalar()
        without = total - with_emb
        logger.info(f"[GPU {gpu_id}] Total docs: {total}, With embeddings: {with_emb}, Without: {without}")

        if without == 0:
            logger.info("All documents already have embeddings.")
            return True

        logger.info("Fetching doc IDs without embeddings...")
        doc_ids_no_emb = [row[0] for row in db.query(Document.id).filter(
            Document.embedding.is_(None)
        ).all()]
        logger.info(f"Fetched {len(doc_ids_no_emb)} doc IDs.")

        my_ids = doc_ids_no_emb[start_idx:end_idx]
        logger.info(f"[GPU {gpu_id}] Processing indices {start_idx}-{end_idx} ({len(my_ids)} docs)")

        chunk_size = 1000
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
                f"[GPU {gpu_id}] Embedded {processed}/{len(my_ids)} docs "
                f"({rate:.1f} docs/sec, ETA: {eta/60:.1f}min)"
            )

        logger.info(f"[GPU {gpu_id}] Complete. Total processed: {processed}")
        return True
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed: {e}", exc_info=True)
        return False
    finally:
        db.close()


if __name__ == "__main__":
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end_idx = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    if not bulk_embed_range(gpu_id, start_idx, end_idx):
        sys.exit(1)
