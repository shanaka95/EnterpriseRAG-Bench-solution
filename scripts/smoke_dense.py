#!/usr/bin/env python3
"""Smoke-test the local dense LanceDB after rsync from the server."""
import sys
import lancedb
import numpy as np
import random

LANCE = "/data/projects/rag/data/dense_index/db"
TABLE = "documents"

def main():
    db = lancedb.connect(LANCE)
    if TABLE not in db.table_names():
        sys.exit(f"FAIL: table {TABLE} not in {LANCE}")
    t = db.open_table(TABLE)
    n = t.count_rows()
    print(f"rows: {n:,}")
    if n < 500_000:
        sys.exit(f"FAIL: too few rows ({n})")

    print("\nSchema:")
    print(t.schema)

    print("\nSample 5 rows:")
    random.seed(42)
    arrow = t.to_lance().to_table(columns=["id","source","n_tokens","embedding"], limit=2000)
    for i in random.sample(range(arrow.num_rows), 5):
        r = arrow.slice(i, 1).to_pylist()[0]
        v = np.asarray(r["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(v))
        print(f"  ✓ {r['id'][:60]:60s}  src={r['source']:<14s}  n_tok={r['n_tokens']:<6d}  dim={v.shape[0]}  norm={norm:.4f}")
        assert 0.95 < norm < 1.05, f"bad norm {norm}"

    print("\nSource counts (sampled 50k):")
    sample = t.to_lance().to_table(columns=["source"], limit=50_000)
    from collections import Counter
    c = Counter(sample.column("source").to_pylist())
    for src, cnt in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {src:<14s} {cnt:>6,}")

    print("\nOK ✅")

if __name__ == "__main__":
    main()
