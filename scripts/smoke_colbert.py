"""
Smoke test for the local ColBERT reranker pipeline.

Runs end-to-end:
  1. Open the local LanceDB at settings.COLBERT_INDEX_PATH.
  2. Pick 20 random doc_ids.
  3. Encode a hand-written query.
  4. Run colbert_rerank, print scored ranking.

Run from /data/projects/rag:
  ./backend/venv/bin/python scripts/smoke_colbert.py
"""
import os
import random
import sys

# Make 'app' importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.ml.colbert_reranker import colbert_rerank, get_table  # noqa: E402


def main():
    table = get_table()
    n = table.count_rows()
    print(f"LanceDB rows: {n}")

    # Sample 20 random ids
    sample = table.to_lance().to_table(
        columns=["id"], limit=2000
    )
    all_ids = sample.column("id").to_pylist()
    random.seed(42)
    picked = random.sample(all_ids, min(20, len(all_ids)))

    query = "KMS rotation MFA fallout SSO auth failures"
    print(f"\nQuery: {query!r}")
    print(f"Candidates: {len(picked)} random docs")

    ranked = colbert_rerank(query, picked, top_k=10)
    print(f"\nTop-10 results:")
    for i, (did, score) in enumerate(ranked, 1):
        print(f"  {i:2d}. {score:8.3f}  {did}")

    print("\n=== smoke OK ===")


if __name__ == "__main__":
    main()
