#!/usr/bin/env python3
"""
Round 24: MiniMax (Claude-Sonnet-4.5) listwise rerank on expanded pool.

Pool per question = A+ top-200 ∪ BM25 top-100 ∪ cluster expansion top-500 (centroid × size × BM25 rate).
Then prompt Claude to listwise-rank the pool to top-10.

Goal: K@10 ≥ 95% on 470 evaluable.

Why MiniMax:
- Anthropic API direct returned 401 (placeholder key)
- OpenRouter cohere/rerank-4-fast depleted ($10/10)
- MINIMAX_API_KEY works at 167.233.22.91:19950 (Anthropic-compatible Claude Sonnet 4.5)
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
from collections import Counter

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

# Module caches
_CENTS = _CLUSTER_SIZES = _CLUSTER_DOC_LISTS = _BM25 = _DOC_CLUSTER = _DOC_PATH = None
_BASE_URL = _API_KEY = _MODEL = None


def get_dsid(p):
    if not p: return ""
    m = DSID_RE.search(str(p))
    return m.group(0) if m else ""


def load_eval_qs():
    qs = [json.loads(l) for l in open(QUESTIONS)]
    expected = set(json.load(open(EXPECTED))
)
    return [{"qid": q["question_id"], "row_idx": i,
             "text": q.get("question", q.get("text", "")),
             "golds_set": set(g for g in q["expected_doc_ids"] if g in expected)}
            for i, q in enumerate(qs) if any(g in expected for g in q["expected_doc_ids"])]


def find_doc_path(doc_id: str) -> Path | None:
    bare = doc_id.split("/", 1)[1] if "/" in doc_id else doc_id
    return _DOC_PATH.get(bare)


def load_doc_text(doc_id: str, max_chars: int = 1500) -> str:
    p = find_doc_path(doc_id)
    if p is None:
        return ""
    return p.read_text(encoding="utf-8", errors="replace")[:max_chars]


def build_pool(q, aplus, k_a=200, k_b=100, k_cl=500, per_cl=50, max_pool=300):
    """A+200 ∪ BM25_100 ∪ cluster_500×50 → max 300."""
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


def call_minimax(query: str, docs: list[tuple[str, str]], max_tokens: int = 4000,
                 timeout: int = 120, top_k: int = 100,
                 prompt_mode: str = "topk") -> tuple[list[int], dict]:
    """Call MiniMax rerank. Returns (list of doc indices in rank order, usage)."""
    doc_block = "\n\n".join(f"[{i}] {title}\n{text}" for i, (title, text) in enumerate(docs))
    if prompt_mode == "topk":
        user = (
            f"Query: {query}\n\n"
            f"You have {len(docs)} candidate documents below. Your job is to identify the {top_k} "
            f"that are MOST relevant to answering the query. "
            f"Return ONLY a JSON array of exactly {top_k} integer indices in best-to-worst "
            f"relevance order. Example: [42, 17, 3, ...]\n\n"
            f"Documents:\n{doc_block}"
        )
    elif prompt_mode == "score":
        user = (
            f"Query: {query}\n\n"
            f"Rate each candidate's relevance 0-10 (10 = perfectly answers the query). "
            f"Return ONLY a JSON array of {len(docs)} integers in the same order as the docs. "
            f"Example: [9, 2, 7, 0, 5, ...]\n\n"
            f"Documents:\n{doc_block}"
        )
    else:
        user = (
            f"Query: {query}\n\n"
            f"Candidates ({len(docs)} documents):\n{doc_block}\n\n"
            "Return ONLY a JSON array of integers — the indices of the documents in best-to-worst "
            "relevance order for answering the query. Include all indices, no comments."
        )
    body = {
        "model": _MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        f"{_BASE_URL}/v1/messages",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": _API_KEY,
                 "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    text = resp["content"][0]["text"].strip()
    m = re.search(r"\[([\d,\s.\-eE]+)\]", text)
    if not m:
        raise ValueError(f"no JSON array in response: {text[:200]}")
    vals = [v.strip() for v in m.group(1).split(",") if v.strip()]
    parsed = []
    for v in vals:
        try:
            parsed.append(float(v))
        except ValueError:
            pass
    if prompt_mode == "score":
        # Each doc gets a score; sort indices by score desc
        scores = parsed + [0.0] * (len(docs) - len(parsed))
        order = sorted(range(len(docs)), key=lambda i: -scores[i])
    else:
        # topk or rank: each value is an index
        seen = set()
        order = []
        for v in parsed:
            idx = int(v)
            if 0 <= idx < len(docs) and idx not in seen:
                order.append(idx)
                seen.add(idx)
    return order, resp.get("usage", {})


def main():
    global _CENTS, _CLUSTER_SIZES, _CLUSTER_DOC_LISTS, _BM25, _DOC_CLUSTER, _DOC_PATH
    global _BASE_URL, _API_KEY, _MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--k_a", type=int, default=200)
    ap.add_argument("--k_b", type=int, default=100)
    ap.add_argument("--k_cl", type=int, default=500)
    ap.add_argument("--per_cl", type=int, default=50)
    ap.add_argument("--max_pool", type=int, default=300)
    ap.add_argument("--max_doc_chars", type=int, default=1500)
    ap.add_argument("--max_input_docs", type=int, default=100,
                    help="cap docs sent to LLM (top by RRF order)")
    ap.add_argument("--top_k", type=int, default=100,
                    help="top-K to ask from prompt_mode=topk")
    ap.add_argument("--prompt_mode", type=str, default="topk",
                    choices=["topk", "score", "rank"])
    args = ap.parse_args()

    _API_KEY = os.environ.get("MINIMAX_API_KEY")
    _BASE_URL = (os.environ.get("MINIMAX_BASE_URL") or "http://167.233.22.91:19950/").rstrip("/")
    _MODEL = os.environ.get("MINIMAX_MODEL") or "MiniMax-M3"
    if not _API_KEY:
        print("ERROR: MINIMAX_API_KEY not set", file=sys.stderr); sys.exit(1)

    print(f"[load] model={_MODEL}, base={_BASE_URL}", flush=True)
    aplus = {json.loads(ln)["question_id"]: json.loads(ln)["ranked_ids"] for ln in open(A_PLUS)}
    _BM25 = {}
    for ln in open(BM25):
        r = json.loads(ln)
        _BM25[r["question_id"]] = r["top_1000_ids"] if "top_1000_ids" in r else r["top_100_ids"][:1000]
    eval_qs = load_eval_qs()
    if args.num < len(eval_qs):
        eval_qs = eval_qs[:args.num]

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
        if f.is_file():
            _DOC_PATH.setdefault(f.name, f)
    print(f"  {len(_DOC_PATH)} files", flush=True)

    out_per_q = OUT_DIR / "round24_per_question.csv"
    out_summary = OUT_DIR / "round24_summary.csv"

    per_q_rows = []
    pool_recall = []
    orig_pool_size = []
    total_in = total_out = 0
    t0 = time.time()
    for i, q in enumerate(eval_qs):
        cands = build_pool(q, aplus, k_a=args.k_a, k_b=args.k_b, k_cl=args.k_cl,
                           per_cl=args.per_cl, max_pool=args.max_pool)
        orig_pool_size.append(len(cands))
        pool_has_gold = any(get_dsid(d) in q["golds_set"] for d in cands)
        pool_recall.append(int(pool_has_gold))

        # Cap to max_input_docs
        send_cands = cands[:args.max_input_docs]
        # Load first snippet of each
        snippets = []
        for d in send_cands:
            text = load_doc_text(d, args.max_doc_chars)
            # Title: first line
            title = text.split("\n", 1)[0][:100] if text else d.split("/")[-1]
            snippets.append((title, text))

        try:
            t1 = time.time()
            order, usage = call_minimax(q["text"], snippets, top_k=args.top_k,
                                        prompt_mode=args.prompt_mode)
            elapsed = time.time() - t1
            in_t = usage.get("input_tokens", 0); out_t = usage.get("output_tokens", 0)
            total_in += in_t; total_out += out_t
            # Re-order cands
            reranked = []
            for idx in order:
                if 0 <= idx < len(send_cands):
                    reranked.append(send_cands[idx])
            # Append any missing in original order
            seen = set(reranked)
            for d in send_cands:
                if d not in seen: reranked.append(d); seen.add(d)
            reranked_ids = reranked
            err = ""
        except Exception as e:
            print(f"  [{i}] q={q['qid']} ERROR: {e}", flush=True)
            reranked_ids = send_cands
            elapsed = 0.0
            err = str(e)[:200]

        row = {"question_id": q["qid"], "pool_size": len(cands), "pool_has_gold": pool_has_gold,
               "elapsed": round(elapsed, 2), "in_tok": in_t, "out_tok": out_t, "error": err}
        for k in K_VALUES:
            top_k = reranked_ids[:k]
            hit = any(get_dsid(d) in q["golds_set"] for d in top_k)
            row[f"hit@{k}"] = int(hit)
        per_q_rows.append(row)

        if (i + 1) % 1 == 0:
            elapsed_total = time.time() - t0
            eta = elapsed_total / (i + 1) * (len(eval_qs) - i - 1)
            acc_now = {k: sum(r[f"hit@{k}"] for r in per_q_rows) / (i + 1) * 100 for k in K_VALUES}
            print(f"  [{i+1}/{len(eval_qs)}] q={q['qid']} pool={len(cands)} sent={len(send_cands)} "
                  f"t={elapsed:.1f}s in={in_t} out={out_t} | K@10={acc_now[10]:.1f}% K@15={acc_now[15]:.1f}% | "
                  f"pool_recall={sum(pool_recall)/len(pool_recall)*100:.1f}% | "
                  f"total_in={total_in} total_out={total_out} | ETA={eta/60:.1f}min",
                  flush=True)

    with open(out_per_q, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q_rows[0].keys()))
        w.writeheader(); w.writerows(per_q_rows)

    with open(out_summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "n_questions", "acc_pct", "pool_recall_pct", "avg_pool_size",
                    "total_in_tok", "total_out_tok"])
        for k in K_VALUES:
            n = sum(r[f"hit@{k}"] for r in per_q_rows)
            w.writerow([k, len(per_q_rows), f"{n/len(per_q_rows)*100:.1f}",
                        f"{sum(pool_recall)/len(pool_recall)*100:.1f}",
                        int(sum(orig_pool_size)/len(orig_pool_size)),
                        total_in, total_out])

    print(f"\n[done] {len(per_q_rows)} questions in {(time.time()-t0)/60:.1f}min", flush=True)
    print(f"\nFinal:", flush=True)
    for k in K_VALUES:
        n = sum(r[f"hit@{k}"] for r in per_q_rows)
        print(f"  K@{k}: {n/len(per_q_rows)*100:.1f}% ({n}/{len(per_q_rows)})", flush=True)


if __name__ == "__main__":
    main()