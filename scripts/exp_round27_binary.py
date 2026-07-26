#!/usr/bin/env python3
"""
Round 27: Batched binary yes/no rerank via MiniMax.

For each question, take top-100 by RRF, split into 5 chunks of 20, ask
the LLM "is each document relevant?" as a batched yes/no JSON array.
Score = 1.0 for yes, 0.0 for no. Then sort by score desc, ties broken
by RRF order.

This is closer to the Qwen3 yes/no logprob approach but uses MiniMax.
The LLM is much better at yes/no than at ranking 100 items.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
A_PLUS = PROJECT_ROOT / "data/exp_mq_rrf_500/results/A+_rankings.jsonl"
BM25 = PROJECT_ROOT / "data/bm25_topk_docids.jsonl"
QUESTIONS = PROJECT_ROOT / "data/questions.jsonl"
EXPECTED = PROJECT_ROOT / "data/expected_doc_ids.json"
CORPUS = PROJECT_ROOT / "data/all_documents"
INDEX = PROJECT_ROOT / "data/exp_full_cluster_v5/index"
QEMB = PROJECT_ROOT / "data/jina_v5_query_emb.npy"
OUT_DIR = PROJECT_ROOT / "data/exp_cluster_filter_rerank"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DSID_RE = re.compile(r"dsid_[a-f0-9]{32}")
K_VALUES = (5, 10, 15, 20, 50, 100)

_CENTS = _CLUSTER_SIZES = _CLUSTER_DOC_LISTS = _BM25 = _DOC_CLUSTER = _DOC_PATH = None
_BASE_URL = _API_KEY = _MODEL = None


def get_dsid(p):
    if not p: return ""
    m = DSID_RE.search(str(p))
    return m.group(0) if m else ""


def load_eval_qs():
    qs = [json.loads(l) for l in open(QUESTIONS)]
    expected = set(json.load(open(EXPECTED)))
    return [{"qid": q["question_id"], "row_idx": i,
             "text": q.get("question", q.get("text", "")),
             "golds_set": set(g for g in q["expected_doc_ids"] if g in expected)}
            for i, q in enumerate(qs) if any(g in expected for g in q["expected_doc_ids"])]


def find_doc_path(doc_id):
    bare = doc_id.split("/", 1)[1] if "/" in doc_id else doc_id
    return _DOC_PATH.get(bare)


def load_doc_text(doc_id, max_chars=1500):
    p = find_doc_path(doc_id)
    if p is None: return ""
    return p.read_text(encoding="utf-8", errors="replace")[:max_chars]


def build_pool(q, aplus, k_a=200, k_b=100, k_cl=500, per_cl=50, max_pool=500):
    cands, seen = [], set()
    for d in aplus.get(q["qid"], [])[:k_a]:
        if d not in seen: seen.add(d); cands.append(d)
    for d in _BM25.get(q["qid"], [])[:k_b]:
        if d not in seen: seen.add(d); cands.append(d)
    bm25_clusters = set()
    for d in _BM25.get(q["qid"], [])[:200]:
        cl = _DOC_CLUSTER.get(d, -1)
        if cl >= 0: bm25_clusters.add(cl)
    row = q["row_idx"]
    vec = np.asarray(np.load(QEMB, mmap_mode="r")[row])
    cs = _CENTS @ vec
    n_cl = len(_CENTS)
    cl_score = np.zeros(n_cl, dtype=np.float32)
    for cl in range(n_cl):
        if cs[cl] < 0.1: continue
        size = _CLUSTER_SIZES[cl]
        boost = 1.5 if cl in bm25_clusters else 1.0
        cl_score[cl] = cs[cl] * np.log1p(size) * boost
    top_cls = np.argsort(-cl_score)[:k_cl]
    for cl in top_cls:
        for d_id in _CLUSTER_DOC_LISTS.get(int(cl), [])[:per_cl]:
            if d_id not in seen:
                seen.add(d_id); cands.append(d_id)
    return cands[:max_pool]


def call_minimax_yesno(query, snippets, max_tokens=300, timeout=120):
    """Send snippets, get back JSON array of yes/no (or 1/0) of same length."""
    doc_block = "\n\n".join(f"[{i}] {title}\n{text[:1000]}" for i, (title, text) in enumerate(snippets))
    user = (
        f"Query: {query}\n\n"
        f"For each of the {len(snippets)} documents below, answer YES if it contains information "
        f"directly relevant to answering the query, or NO otherwise. "
        f"Return ONLY a JSON array of {len(snippets)} strings — each \"yes\" or \"no\" in order. "
        f"Example: [\"yes\", \"no\", \"yes\", ...]\n\n"
        f"Documents:\n{doc_block}"
    )
    body = {"model": _MODEL, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}]}
    req = urllib.request.Request(
        f"{_BASE_URL}/v1/messages", data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": _API_KEY,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    text = resp["content"][0]["text"].strip()
    # Try JSON array (string or numeric) anywhere
    m = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
    if not m:
        # Fallback: find "yes"/"no" tokens in free text
        yes_count = len(re.findall(r"\byes\b", text, re.IGNORECASE))
        no_count = len(re.findall(r"\bno\b", text, re.IGNORECASE))
        if yes_count + no_count >= len(snippets) * 0.5:
            # Naive first-N yes, then no
            scores = [1.0] * yes_count + [0.0] * (len(snippets) - yes_count)
            return scores, resp.get("usage", {})
        raise ValueError(f"no JSON array: {text[:200]}")
    raw = m.group(0)
    # Take quoted strings, else bare words
    quoted = re.findall(r'"([^"]+)"', raw)
    if len(quoted) >= len(snippets) * 0.5:
        tokens = quoted
    else:
        tokens = re.findall(r"\b\d+\b|[a-zA-Z]+", raw)
    scores = []
    for j in range(len(snippets)):
        if j < len(tokens):
            v = tokens[j].strip().lower()
            if v in ("yes", "y", "1", "true", "relevant", "✓"):
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)
    return scores, resp.get("usage", {})


def main():
    global _CENTS, _CLUSTER_SIZES, _CLUSTER_DOC_LISTS, _BM25, _DOC_CLUSTER, _DOC_PATH
    global _BASE_URL, _API_KEY, _MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--max_pool", type=int, default=100)
    ap.add_argument("--chunk_size", type=int, default=20)
    ap.add_argument("--max_doc_chars", type=int, default=1500)
    args = ap.parse_args()

    _API_KEY = os.environ.get("MINIMAX_API_KEY")
    _BASE_URL = (os.environ.get("MINIMAX_BASE_URL") or "http://167.233.22.91:19950/").rstrip("/")
    _MODEL = os.environ.get("MINIMAX_MODEL") or "MiniMax-M3"
    if not _API_KEY: print("ERROR: MINIMAX_API_KEY not set", file=sys.stderr); sys.exit(1)

    print(f"[load] model={_MODEL}", flush=True)
    aplus = {json.loads(ln)["question_id"]: json.loads(ln)["ranked_ids"] for ln in open(A_PLUS)}
    _BM25 = {}
    for ln in open(BM25):
        r = json.loads(ln)
        _BM25[r["question_id"]] = r["top_1000_ids"] if "top_1000_ids" in r else r["top_100_ids"][:1000]
    eval_qs = load_eval_qs()
    if args.num < len(eval_qs): eval_qs = eval_qs[:args.num]

    print("[load] cluster index", flush=True)
    _DOC_CLUSTER = json.load(open(INDEX / "doc_id_to_cluster.json"))
    _CLUSTER_DOC_LISTS = {}
    for d, c in _DOC_CLUSTER.items():
        _CLUSTER_DOC_LISTS.setdefault(c, []).append(d)
    raw = np.load(INDEX / "centroids.npy")
    _CENTS = (raw / np.linalg.norm(raw, axis=1, keepdims=True).clip(min=1e-8)).astype(np.float32)
    _CLUSTER_SIZES = np.array([len(_CLUSTER_DOC_LISTS.get(c, [])) for c in range(len(_CENTS))], dtype=np.float32)
    print("[load] doc path index", flush=True)
    _DOC_PATH = {}
    for f in CORPUS.rglob("*"):
        if f.is_file(): _DOC_PATH.setdefault(f.name, f)
    print(f"  {len(_DOC_PATH)} files", flush=True)

    per_q_rows = []
    pool_recall = []
    orig_pool_size = []
    total_in = total_out = 0
    t0 = time.time()
    for i, q in enumerate(eval_qs):
        cands = build_pool(q, aplus, k_a=200, k_b=100, k_cl=500, per_cl=50,
                           max_pool=args.max_pool)
        orig_pool_size.append(len(cands))
        pool_has_gold = any(get_dsid(d) in q["golds_set"] for d in cands)
        pool_recall.append(int(pool_has_gold))

        # Load snippets
        snippets = []
        for d in cands:
            text = load_doc_text(d, args.max_doc_chars)
            title = text.split("\n", 1)[0][:100] if text else d.split("/")[-1]
            snippets.append((title, text))

        # Batched yes/no in chunks
        all_scores = [0.0] * len(cands)
        n_chunks = (len(snippets) + args.chunk_size - 1) // args.chunk_size
        for ci in range(n_chunks):
            chunk = snippets[ci*args.chunk_size:(ci+1)*args.chunk_size]
            try:
                scores, usage = call_minimax_yesno(q["text"], chunk)
                in_t = usage.get("input_tokens", 0); out_t = usage.get("output_tokens", 0)
                total_in += in_t; total_out += out_t
                for j, s in enumerate(scores):
                    all_scores[ci*args.chunk_size + j] = s
            except Exception as e:
                print(f"  [{i}] c{ci} ERROR: {e}", flush=True)

        # Build reranked: yes-docs first (preserving RRF order), then no-docs
        reranked = []
        for j, d in enumerate(cands):
            if all_scores[j] > 0.5:
                reranked.append(d)
        for j, d in enumerate(cands):
            if all_scores[j] <= 0.5:
                reranked.append(d)

        row = {"question_id": q["qid"], "pool_size": len(cands), "pool_has_gold": pool_has_gold,
               "n_yes": int(sum(1 for s in all_scores if s > 0.5))}
        for k in K_VALUES:
            top_k = reranked[:k]
            hit = any(get_dsid(d) in q["golds_set"] for d in top_k)
            row[f"hit@{k}"] = int(hit)
        per_q_rows.append(row)

        if (i + 1) % 1 == 0:
            elapsed_total = time.time() - t0
            eta = elapsed_total / (i + 1) * (len(eval_qs) - i - 1)
            acc_now = {k: sum(r[f"hit@{k}"] for r in per_q_rows) / (i + 1) * 100 for k in K_VALUES}
            print(f"  [{i+1}/{len(eval_qs)}] q={q['qid']} pool={len(cands)} n_yes={row['n_yes']} | "
                  f"K@5={acc_now[5]:.1f}% K@10={acc_now[10]:.1f}% K@15={acc_now[15]:.1f}% K@20={acc_now[20]:.1f}% | "
                  f"pool_recall={sum(pool_recall)/len(pool_recall)*100:.1f}% | "
                  f"ETA={eta/60:.1f}min | tok_in={total_in} tok_out={total_out}",
                  flush=True)

    out_per_q = OUT_DIR / "round27_per_question.csv"
    out_summary = OUT_DIR / "round27_summary.csv"
    with open(out_per_q, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q_rows[0].keys()))
        w.writeheader(); w.writerows(per_q_rows)
    with open(out_summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "n_questions", "acc_pct", "pool_recall_pct", "avg_pool_size", "total_in_tok", "total_out_tok"])
        for k in K_VALUES:
            n = sum(r[f"hit@{k}"] for r in per_q_rows)
            w.writerow([k, len(per_q_rows), f"{n/len(per_q_rows)*100:.1f}",
                        f"{sum(pool_recall)/len(pool_recall)*100:.1f}",
                        int(sum(orig_pool_size)/len(orig_pool_size)), total_in, total_out])

    print(f"\n[done] {len(per_q_rows)} questions in {(time.time()-t0)/60:.1f}min", flush=True)
    print(f"\nFinal:", flush=True)
    for k in K_VALUES:
        n = sum(r[f"hit@{k}"] for r in per_q_rows)
        print(f"  K@{k}: {n/len(per_q_rows)*100:.1f}% ({n}/{len(per_q_rows)})", flush=True)


if __name__ == "__main__":
    main()