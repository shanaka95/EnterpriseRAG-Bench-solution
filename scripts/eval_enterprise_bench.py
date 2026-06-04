#!/usr/bin/env python3
"""
EnterpriseRAG-Bench evaluation script.

Evaluates document recall against 500 benchmark questions.
Measures:
- Document Recall: % of expected gold documents in retrieved set
- Top-10 Recall: % of gold docs in top-10 results
- Top-20 Recall: % of gold docs in top-20 results

Uses questions.jsonl from the EnterpriseRAG-Bench dataset.
"""
import json
import csv
import time
import sys
import os
import requests
from collections import defaultdict

API_BASE = os.environ.get("API_BASE", "http://localhost:8080")
QUESTIONS_FILE = os.environ.get("QUESTIONS_FILE", "/app/questions.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/eval_results")


def load_questions(filepath):
    """Load questions from JSONL file."""
    questions = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def retrieve(query, top_k=100):
    """Run retrieval query against the API."""
    try:
        resp = requests.post(
            f"{API_BASE}/api/v1/query",
            json={"query": query, "top_k": top_k},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  Query API error: {resp.status_code}")
            return None
    except Exception as e:
        print(f"  Query error: {e}")
        return None


def evaluate_questions(questions, top_k_values=[10, 20, 50]):
    """
    Evaluate document recall for all questions.

    For each question:
    - expected_doc_ids: the gold standard document IDs
    - We check if each gold doc appears in the top-K retrieved docs
    """
    results = []

    # Per-type metrics
    type_metrics = defaultdict(lambda: {"total": 0, "recall_all": 0, "recall_10": 0, "recall_20": 0, "recall_50": 0})

    total_recall_all = 0
    total_recall_10 = 0
    total_recall_20 = 0
    total_recall_50 = 0
    total_questions = 0
    total_gold_docs = 0
    total_found_all = 0
    total_found_10 = 0
    total_found_20 = 0
    total_found_50 = 0

    for i, q in enumerate(questions):
        question_id = q.get("question_id", f"q_{i}")
        question = q["question"]
        question_type = q.get("question_type", "unknown")
        expected_doc_ids = q.get("expected_doc_ids", [])
        gold_answer = q.get("gold_answer", "")

        # Skip questions with no expected docs (e.g., "info_not_found" type)
        if not expected_doc_ids:
            results.append({
                "question_id": question_id,
                "question_type": question_type,
                "question": question[:200],
                "num_expected": 0,
                "num_found_all": 0,
                "recall_all": 1.0,  # No docs expected = perfect recall
                "recall_10": 1.0,
                "recall_20": 1.0,
                "recall_50": 1.0,
                "error": None,
            })
            continue

        total_questions += 1
        num_expected = len(expected_doc_ids)
        total_gold_docs += num_expected

        # Retrieve with very large top_k to maximize recall
        result = retrieve(question, top_k=20000)
        if result is None:
            results.append({
                "question_id": question_id,
                "question_type": question_type,
                "question": question[:200],
                "num_expected": num_expected,
                "num_found_all": 0,
                "recall_all": 0.0,
                "recall_10": 0.0,
                "recall_20": 0.0,
                "recall_50": 0.0,
                "error": "retrieval_failed",
            })
            continue

        retrieved_all = result.get("doc_ids", [])

        # The ingestion pipeline may append __suffix to doc IDs (e.g. dsid_xxx__pr-123).
        # The benchmark gold IDs only have the base dsid_xxx prefix.
        # Strip suffixes for fair comparison.
        def _base_id(doc_id: str) -> str:
            return doc_id.split("__")[0] if "__" in doc_id else doc_id

        retrieved_base = [_base_id(d) for d in retrieved_all]

        # Check recall at different top-k values
        found_all = sum(1 for did in expected_doc_ids if did in retrieved_base)
        found_10 = sum(1 for did in expected_doc_ids if did in retrieved_base[:10])
        found_20 = sum(1 for did in expected_doc_ids if did in retrieved_base[:20])
        found_50 = sum(1 for did in expected_doc_ids if did in retrieved_base[:50])

        recall_all = found_all / num_expected if num_expected > 0 else 1.0
        recall_10 = found_10 / num_expected if num_expected > 0 else 1.0
        recall_20 = found_20 / num_expected if num_expected > 0 else 1.0
        recall_50 = found_50 / num_expected if num_expected > 0 else 1.0

        total_found_all += found_all
        total_found_10 += found_10
        total_found_20 += found_20
        total_found_50 += found_50

        # Track per-type metrics
        type_metrics[question_type]["total"] += 1
        type_metrics[question_type]["recall_all"] += recall_all
        type_metrics[question_type]["recall_10"] += recall_10
        type_metrics[question_type]["recall_20"] += recall_20
        type_metrics[question_type]["recall_50"] += recall_50

        results.append({
            "question_id": question_id,
            "question_type": question_type,
            "question": question[:200],
            "num_expected": num_expected,
            "num_retrieved": len(retrieved_all),
            "num_found_all": found_all,
            "num_found_10": found_10,
            "num_found_20": found_20,
            "num_found_50": found_50,
            "recall_all": round(recall_all, 4),
            "recall_10": round(recall_10, 4),
            "recall_20": round(recall_20, 4),
            "recall_50": round(recall_50, 4),
            "error": None,
        })

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(questions)}] "
                  f"Doc Recall (all): {total_found_all}/{total_gold_docs}={total_found_all/total_gold_docs*100:.1f}% "
                  f"Top-10: {total_found_10}/{total_gold_docs}={total_found_10/total_gold_docs*100:.1f}% "
                  f"Top-20: {total_found_20}/{total_gold_docs}={total_found_20/total_gold_docs*100:.1f}%")

        time.sleep(float(os.environ.get("EVAL_RATE_LIMIT_SEC", "0.05")))  # Configurable rate limit

    # ── Summary ──
    print("\n" + "=" * 70)
    print("ENTERPRISE RAG BENCH - EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions evaluated: {total_questions}")
    print(f"Total gold documents: {total_gold_docs}")
    print()
    print(f"Document Recall (all retrieved): {total_found_all}/{total_gold_docs} = {total_found_all/total_gold_docs*100:.1f}%")
    print(f"Document Recall (top-10):         {total_found_10}/{total_gold_docs} = {total_found_10/total_gold_docs*100:.1f}%")
    print(f"Document Recall (top-20):         {total_found_20}/{total_gold_docs} = {total_found_20/total_gold_docs*100:.1f}%")
    print(f"Document Recall (top-50):         {total_found_50}/{total_gold_docs} = {total_found_50/total_gold_docs*100:.1f}%")
    print()

    # Per-type breakdown
    print("PER-TYPE BREAKDOWN:")
    print(f"{'Type':<25} {'Count':>6} {'Recall(all)':>12} {'Recall@10':>11} {'Recall@20':>11}")
    print("-" * 70)
    for qtype, metrics in sorted(type_metrics.items()):
        if metrics["total"] == 0:
            continue
        n = metrics["total"]
        print(f"{qtype:<25} {n:>6} {metrics['recall_all']/n*100:>11.1f}% {metrics['recall_10']/n*100:>10.1f}% {metrics['recall_20']/n*100:>10.1f}%")

    # Save detailed results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV output
    csv_path = os.path.join(OUTPUT_DIR, "retrieval_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        writer.writeheader()
        writer.writerows(results)

    # Summary JSON
    summary = {
        "total_questions": total_questions,
        "total_gold_documents": total_gold_docs,
        "doc_recall_all": round(total_found_all / total_gold_docs * 100, 1) if total_gold_docs > 0 else 0,
        "doc_recall_top10": round(total_found_10 / total_gold_docs * 100, 1) if total_gold_docs > 0 else 0,
        "doc_recall_top20": round(total_found_20 / total_gold_docs * 100, 1) if total_gold_docs > 0 else 0,
        "doc_recall_top50": round(total_found_50 / total_gold_docs * 100, 1) if total_gold_docs > 0 else 0,
        "per_type": {k: {
            "count": v["total"],
            "recall_all": round(v["recall_all"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
            "recall_10": round(v["recall_10"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
            "recall_20": round(v["recall_20"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
        } for k, v in type_metrics.items()},
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    questions_file = sys.argv[1] if len(sys.argv) > 1 else QUESTIONS_FILE
    questions = load_questions(questions_file)
    print(f"Loaded {len(questions)} questions from {questions_file}")

    # Count by type
    type_counts = defaultdict(int)
    for q in questions:
        type_counts[q.get("question_type", "unknown")] += 1
    print("Question types:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    summary = evaluate_questions(questions)
