#!/usr/bin/env python3
import csv
from collections import defaultdict

with open("/app/eval_results/retrieval_results.csv") as f:
    rows = list(csv.DictReader(f))

failures = [r for r in rows if float(r["recall_all"]) < 0.5]
print("Total evaluated:", len(rows))
print("Failures (recall < 50%):", len(failures))

by_type = defaultdict(lambda: {"total": 0, "fail": 0})
for r in rows:
    t = r["question_type"]
    by_type[t]["total"] += 1
    if float(r["recall_all"]) < 0.5:
        by_type[t]["fail"] += 1

print("\nFailures by type:")
for t, v in sorted(by_type.items(), key=lambda x: x[1]["fail"]/max(x[1]["total"],1), reverse=True):
    pct = v["fail"]/v["total"]*100
    print(f"  {t}: {v['fail']}/{v['total']} = {pct:.1f}% failed")

print("\nSample semantic failures:")
semantic_failures = [r for r in failures if r["question_type"] == "semantic"]
for r in semantic_failures[:10]:
    q = r["question"][:100]
    print(f"  {q}... -> recall={r['recall_all']}")

print("\nSample basic failures:")
basic_failures = [r for r in failures if r["question_type"] == "basic"]
for r in basic_failures[:5]:
    q = r["question"][:100]
    print(f"  {q}... -> recall={r['recall_all']}")
