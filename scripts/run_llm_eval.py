#!/usr/bin/env python3
"""Run LLM-as-judge evaluation on existing RAG results.

Reads ``MINIMAX_API_KEY`` and ``MINIMAX_BASE_URL`` from the environment
(set via .env, scripts/run_ui.sh, or export). The input CSV defaults to
``data/agent_answers_first10.csv``; override with ``RAG_RESULTS_CSV``.
"""
import csv
import json
import os
import requests
import numpy as np

LLM_API_URL = (os.environ.get("MINIMAX_BASE_URL",
                              "https://api.minimax.io/anthropic")
               + "/v1/messages")
LLM_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
LLM_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")
RESULTS_CSV = os.environ.get("RAG_RESULTS_CSV",
                             "/data/projects/rag/data/agent_answers_first10.csv")

if not LLM_API_KEY:
    raise SystemExit(
        "MINIMAX_API_KEY env var is required. Export it or use scripts/run_ui.sh."
    )

def extract_text(response_json):
    """Extract text content from MiniMax response."""
    for block in response_json.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    if response_json.get("content"):
        return response_json["content"][-1].get("text", "")
    return ""

def llm_judge(question, generated_answer, gt_answer):
    """Use LLM to evaluate answer quality."""
    prompt = f"""You are evaluating a RAG system. Rate the generated answer vs the ground truth.

QUESTION: {question}
GROUND TRUTH: {gt_answer[:500]}
GENERATED: {generated_answer[:500]}

Rate 0.0-1.0 on each dimension. RESPOND WITH ONLY THIS JSON:
{{"faithfulness": 0.0, "answer_relevance": 0.0, "completeness": 0.0, "correctness": 0.0, "overall": 0.0}}"""

    try:
        resp = requests.post(
            LLM_API_URL,
            headers={
                "x-api-key": LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code == 200:
            content = extract_text(resp.json())
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
    except Exception as e:
        print(f"  LLM judge error: {e}")
    return None

# Read results
results = []
with open(RESULTS_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    results = list(reader)

print(f"Total results: {len(results)}")

# Evaluate first 20 with LLM judge
metrics = {"faithfulness": [], "answer_relevance": [], "completeness": [], "correctness": [], "overall": []}
evaluated = 0

for i, r in enumerate(results[:20]):
    q = r.get("question", "")
    ga = r.get("generated_answer", "")
    gta = r.get("gt_answer", "")

    if not ga or not gta:
        continue

    print(f"Evaluating {i+1}/20...")
    scores = llm_judge(q, ga, gta)

    if scores:
        for k in metrics:
            if k in scores:
                metrics[k].append(float(scores[k]))
        evaluated += 1

# Print summary
print("\n" + "="*60)
print("LLM-JUDGED EVALUATION RESULTS")
print("="*60)
print(f"Questions evaluated: {evaluated}")
if evaluated > 0:
    for k, v in metrics.items():
        if v:
            print(f"  {k}: {np.mean(v):.3f}")

# Also compute retrieval metrics
recalls = [float(r.get("retrieval_recall", 0)) for r in results if r.get("retrieval_recall")]
gt_in = sum(1 for r in results if r.get("gt_in_retrieved") == "True")
total = len(results)
docs_retrieved = [float(r.get("num_docs_retrieved", 0)) for r in results if r.get("num_docs_retrieved")]

print(f"\nRETRIEVAL METRICS (all {total} questions)")
print(f"  Mean Retrieval Recall: {np.mean(recalls):.3f}" if recalls else "  No recall data")
print(f"  Ground Truth Found: {gt_in}/{total} ({gt_in/total*100:.1f}%)")
print(f"  Mean Docs Retrieved: {np.mean(docs_retrieved):.1f}" if docs_retrieved else "  No docs data")
print("="*60)
