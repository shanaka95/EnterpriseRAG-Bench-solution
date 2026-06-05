#!/usr/bin/env python3
"""Run the full agent workflow on the first N questions and write the
answers to a CSV (default) or JSONL file.

CSV columns (default):
    question_id, question, answer, document_ids, llm_turns,
    docs_read, elapsed_s, hit

JSONL (--fmt jsonl):
    {"question_id": "qst_0001", "answer": "...", "document_ids": ["dsid_abc", ...]}

Usage:
    ./backend/venv/bin/python scripts/run_first_10_questions.py [--n 10] [--out PATH] [--fmt csv|jsonl]

The BM25 and jina-v3 indexes are loaded once and cached in process memory,
so the first question takes ~4 min (BM25 build) and subsequent questions
take only the LLM time (5-15s each).
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Ensure backend is on the path
BACKEND = "/data/projects/rag/backend"
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# Load API key from env (scripts/run_ui.sh sets it; or set directly)
if not os.environ.get("MINIMAX_API_KEY"):
    print("ERROR: MINIMAX_API_KEY not set. Export it or use scripts/run_ui.sh.",
          file=sys.stderr)
    sys.exit(1)

from agent import run_agent  # noqa: E402

QUESTIONS_PATH = "/data/projects/rag/data/questions.jsonl"

CSV_COLUMNS = [
    "question_id", "question", "answer", "document_ids",
    "llm_turns", "docs_read", "elapsed_s", "hit",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="how many of the first questions to run (default 10)")
    ap.add_argument("--out", default="/data/projects/rag/data/agent_answers_first10.csv",
                    help="output file path (extension determines default format)")
    ap.add_argument("--fmt", choices=["csv", "jsonl"], default=None,
                    help="output format (default: inferred from --out extension)")
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    args = ap.parse_args()

    if args.fmt is None:
        args.fmt = "csv" if args.out.endswith(".csv") else "jsonl"

    # Load all questions; take the first N
    with open(args.questions, encoding="utf-8") as f:
        questions = [json.loads(ln) for ln in f if ln.strip()]
    questions = questions[: args.n]
    print(f"[run] {len(questions)} questions selected  format={args.fmt}", flush=True)

    # Truncate output file (start fresh)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")  # truncate

    # Open the writer once so we stream results in real time
    csv_file = open(out_path, "a", encoding="utf-8", newline="") \
        if args.fmt == "csv" else None
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS) \
        if csv_file else None
    if csv_writer:
        csv_writer.writeheader()
        csv_file.flush()

    t_total = time.time()
    n_done = 0
    n_hits = 0
    for i, q in enumerate(questions, 1):
        qid = q["question_id"]
        qtext = q["question"]
        print(f"\n[{i:>2}/{len(questions)}] {qid}: {qtext[:80]}", flush=True)
        t0 = time.time()
        try:
            final = run_agent(
                qtext,
                question_id=qid,
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue
        elapsed = time.time() - t0

        answer = final.get("final_answer") or ""
        supporting = final.get("supporting_doc_ids") or []
        n_ai = sum(1 for m in final.get("messages", []) if m.__class__.__name__ == "AIMessage")
        n_docs = final.get("current_idx", 0)
        expected = q.get("expected_doc_ids", []) or []
        hit = any(any(e in d for d in supporting) for e in expected)
        if hit:
            n_hits += 1
        n_done += 1

        if args.fmt == "csv":
            csv_writer.writerow({
                "question_id": qid,
                "question": qtext,
                "answer": answer,
                "document_ids": "|".join(supporting),  # pipe-joined; CSV-safe
                "llm_turns": n_ai,
                "docs_read": n_docs,
                "elapsed_s": round(elapsed, 2),
                "hit": int(bool(hit)),
            })
            csv_file.flush()
        else:
            rec = {
                "question_id": qid,
                "answer": answer,
                "document_ids": supporting,
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  answer: {answer[:120]!r}", flush=True)
        print(f"  doc_ids: {supporting}", flush=True)
        print(f"  llm_turns={n_ai}  docs_read={n_docs}  "
              f"elapsed={elapsed:.1f}s  hit={'✅' if hit else '❌'}",
              flush=True)

    if csv_file:
        csv_file.close()

    total = time.time() - t_total
    print(f"\n[done] wrote {n_done} records to {out_path}", flush=True)
    print(f"[done] hits: {n_hits}/{n_done}  total wall clock: {total:.1f}s "
          f"({total/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
