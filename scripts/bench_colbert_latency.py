"""
Latency benchmark for the ColBERT reranker.

Runs N queries against a pool of M random doc_ids and reports p50/p95/p99.
Budget: p95 < 1.2 s on local CPU.

Run from /data/projects/rag:
  ./backend/venv/bin/python scripts/bench_colbert_latency.py --pool 1000 --queries 10
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.ml.colbert_reranker import colbert_rerank, get_table  # noqa: E402

QUERIES = [
    "KMS rotation MFA fallout SSO auth failures",
    "asymmetric token ladder streaming grace credits",
    "Northbridge Payments benchmark walkthrough methodology",
    "data deletion TTL enforcement standards",
    "cohort-driven shadow validation playbook",
    "probe pruning adaptive sampling 2026",
    "incident escalation rollback support coverage",
    "Redwood Private upgrade guarantees",
    "sales call action items follow-up",
    "engineering team OKR planning Q3 2025",
    "test flakes retry markers CI logs",
    "feature flag rollout deprecation timeline",
    "customer churn predictor model drift",
    "regional latency SLO breach root cause",
    "GraphQL schema migration breaking changes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    table = get_table()
    n = table.count_rows()
    print(f"LanceDB rows: {n}, pool={args.pool}, queries={args.queries}")

    # Sample one large pool, reuse for all queries (worst-case fetch)
    sample = table.to_lance().to_table(columns=["id"], limit=max(args.pool * 5, 10_000))
    all_ids = sample.column("id").to_pylist()
    random.seed(args.seed)
    pool_ids = random.sample(all_ids, args.pool)

    qs = QUERIES[: args.queries]
    if len(qs) < args.queries:
        qs = (qs * ((args.queries // len(qs)) + 1))[: args.queries]

    # Warmup (avoids model-load time in measurements)
    for q in qs[: args.warmup]:
        colbert_rerank(q, pool_ids[:50], top_k=10)

    latencies = []
    for i, q in enumerate(qs):
        t0 = time.time()
        _ = colbert_rerank(q, pool_ids, top_k=10)
        dt = time.time() - t0
        latencies.append(dt)
        print(f"  q{i+1:2d}: {dt*1000:7.1f} ms  | {q[:60]}")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(0.95 * len(latencies))]
    p99 = latencies[int(0.99 * len(latencies))]
    print(f"\nLatency (n={len(latencies)}, pool={args.pool}):")
    print(f"  p50 = {p50*1000:7.1f} ms")
    print(f"  p95 = {p95*1000:7.1f} ms")
    print(f"  p99 = {p99*1000:7.1f} ms")
    print(f"  avg = {statistics.mean(latencies)*1000:7.1f} ms")
    print(f"  budget: p95 < 1200 ms → {'PASS' if p95 < 1.2 else 'FAIL'}")


if __name__ == "__main__":
    main()
