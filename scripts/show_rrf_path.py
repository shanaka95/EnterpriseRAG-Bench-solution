#!/usr/bin/env python3
"""
Show the full RRF (N=2000, k0=60) ranked path for any question.

Usage:
  ./venv/bin/python scripts/show_rrf_path.py --qid qst_0001            # specific question
  ./venv/bin/python scripts/show_rrf_path.py --top 5                    # first 5 questions
  ./venv/bin/python scripts/show_rrf_path.py --qid qst_0001 --k 20      # top-20 only
  ./venv/bin/python scripts/show_rrf_path.py --search "multipart"       # search by query snippet
"""
import argparse
import json
import sys

PATH = "/data/projects/rag/data/rrf_full_ranking_N2000_k060.jsonl"


def best_match(ranked, expected):
    for eid in expected:
        for ri, d in enumerate(ranked, 1):
            if eid in d:
                return ri
    return None


def show(rec, k):
    print(f'Q: {rec["question"]}')
    print(f'  qid:        {rec["question_id"]}')
    print(f'  source:     {rec["source_types"]}')
    print(f'  expected:   {rec["expected_doc_ids"]}')
    pool = rec['ranked_doc_ids']
    rank = best_match(pool, rec['expected_doc_ids'])
    if rank is None:
        rank_str = "NOT IN RANKING (impossible)"
    elif rank <= k:
        rank_str = f"RANK {rank} (in top-{k})"
    else:
        rank_str = f"RANK {rank} (NOT in top-{k})"
    print(f'  → gold doc is at: {rank_str}')
    print(f'  pool size:  {len(pool)}')
    print()
    print(f'  Top-{k} RRF path:')
    for j, (doc_id, score) in enumerate(zip(pool[:k], rec['rrf_scores'][:k]), 1):
        is_expected = any(e in doc_id for e in rec['expected_doc_ids'])
        marker = "  ★ EXPECTED" if is_expected else ""
        print(f'    {j:>4}. RRF={score:.5f}  {doc_id}{marker}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", help="specific question id (e.g. qst_0001)")
    ap.add_argument("--top", type=int, help="show the first N questions")
    ap.add_argument("--k", type=int, default=20, help="top-K to display (default 20)")
    ap.add_argument("--search", help="substring match on question text")
    args = ap.parse_args()

    shown = 0
    with open(PATH) as f:
        for ln in f:
            rec = json.loads(ln)
            if args.qid and rec['question_id'] != args.qid:
                continue
            if args.search and args.search.lower() not in rec['question'].lower():
                continue
            show(rec, args.k)
            print('-' * 100)
            shown += 1
            if args.top and shown >= args.top:
                break
            if args.qid:
                break

    if shown == 0:
        print(f'No matches. Try --qid qst_0001, --top 5, or --search "snippet"')
        sys.exit(1)


if __name__ == "__main__":
    main()
