#!/usr/bin/env python3
"""
Round 30: Smaller chunks (5) + 3-level rating + 2-pass refinement.

Smaller chunks should give better per-chunk accuracy. Cost: 2x more calls.
Total expected time per query: ~25s (vs 20s for chunks of 10).
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


def load_doc_text(doc_id, max_chars=2500):
    p = find_doc_path(doc_id)
    if p is None: return ""
    return p.read_text(encoding="utf-8", errors="replace")[:max_chars]


def build_pool(q, aplus, k_a=200, k_b=100, k_cl=500, per_cl=50, max_pool=100):
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


def call_minimax_rate(query, snippets, max_tokens=300, timeout=120):
    """3-level: 0/1/2."""
    doc_block = "\n\n".join(f"[{i}] {title}\n{text[:1500]}" for i, (title, text) in enumerate(snippets))
    user = (
        f"Query: {query}\n\n"
        f"Rate each document for relevance:\n"
        f"  2 = directly relevant (contains the answer)\n"
        f"  1 = tangentially relevant (mentions related concepts)\n"
        f"  0 = not relevant\n\n"
        f"Return ONLY a JSON array of {len(snippets)} integers. Example: [2, 0, 1, 2, ...]\n\n"
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
    m = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
    if not m: raise ValueError(f"no JSON: {text[:200]}")
    vals = re.findall(r"\b\d+\b", m.group(0))
    scores = [min(2, max(0, int(vals[j]))) if j < len(vals) else 0
              for j in range(len(snippets))]
    return scores, resp.get("usage", {})


def main():
    global _CENTS, _CLUSTER_SIZES, _CLUSTER_DOC_LISTS, _BM25, _DOC_CLUSTER, _DOC_PATH
    global _BASE_URL, _API_KEY, _MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--max_pool", type=int, default=100)
    ap.add_argument("--chunk_size", type=int, default=5)
    ap.add_argument("--max_doc_chars", type=int, default=2500)
    ap.add_argument("--pass2_size", type=int, default=20)
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
        cands = build_pool(q, aplus, max_pool=args.max_pool)
        orig_pool_size.append(len(cands))
        pool_has_gold = any(get_dsid(d) in q["golds_set"] for d in cands)
        pool_recall.append(int(pool_has_gold))

        snippets = []
        for d in cands:
            text = load_doc_text(d, args.max_doc_chars)
            title = text.split("\n", 1)[0][:100] if text else d.split("/")[-1]
            snippets.append((title, text))

        # Pass 1: 3-level rate per chunk
        all_scores = [0] * len(cands)
        n_chunks = (len(snippets) + args.chunk_size - 1) // args.chunk_size
        for ci in range(n_chunks):
            chunk = snippets[ci*args.chunk_size:(ci+1)*args.chunk_size]
            try:
                scores, usage = call_minimax_rate(q["text"], chunk)
                in_t = usage.get("input_tokens", 0); out_t = usage.get("output_tokens", 0)
                total_in += in_t; total_out += out_t
                for j, s in enumerate(scores):
                    all_scores[ci*args.chunk_size + j] = s
            except Exception as e:
                print(f"  [{i}] p1c{ci} ERROR: {e}", flush=True)

        # Pass 2: re-rate top-20 by score
        order_idx = sorted(range(len(cands)), key=lambda j: (-all_scores[j], j))
        top_idx = order_idx[:args.pass2_size]
        pass2_scores = [all_scores[j] for j in top_idx]
        if top_idx:
            pass2_snippets = [snippets[j] for j in top_idx]
            n_chunks2 = (len(pass2_snippets) + args.chunk_size - 1) // args.chunk_size
            for ci in range(n_chunks2):
                chunk = pass2_snippets[ci*args.chunk_size:(ci+1)*args.chunk_size]
                try:
                    scores, usage = call_minimax_rate(q["text"], chunk)
                    in_t = usage.get("input_tokens", 0); out_t = usage.get("output_tokens", 0)
                    total_in += in_t; total_out += out_t
                    for j, s in enumerate(scores):
                        pass2_scores[ci*args.chunk_size + j] = max(pass2_scores[ci*args.chunk_size + j], s)
                except Exception as e:
                    print(f"  [{i}] p2c{ci} ERROR: {e}", flush=True)

        # Final: rank by max(pass1, pass2) score, ties by RRF
        score_map = {cands[j]: all_scores[j] for j in range(len(cands))}
        for k, j in enumerate(top_idx):
            score_map[cands[j]] = max(score_map[cands[j]], pass2_scores[k])
        reranked = sorted(cands, key=lambda d: (-score_map[d], cands.index(d)))

        row = {"question_id": q["qid"], "pool_size": len(cands), "pool_has_gold": pool_has_gold,
               "n_high": sum(1 for s in score_map.values() if s >= 2),
               "n_mid": sum(1 for s in score_map.values() if s == 1)}
        for k in K_VALUES:
            top_k = reranked[:k]
            hit = any(get_dsid(d) in q["golds_set"] for d in top_k)
            row[f"hit@{k}"] = int(hit)
        per_q_rows.append(row)

        if (i + 1) % 1 == 0:
            elapsed_total = time.time() - t0
            eta = elapsed_total / (i + 1) * (len(eval_qs) - i - 1)
            acc_now = {k: sum(r[f"hit@{k}"] for r in per_q_rows) / (i + 1) * 100 for k in K_VALUES}
            print(f"  [{i+1}/{len(eval_qs)}] q={q['qid']} pool={len(cands)} n_high={row['n_high']} n_mid={row['n_mid']} | "
                  f"K@5={acc_now[5]:.1f}% K@10={acc_now[10]:.1f}% K@15={acc_now[15]:.1f}% K@20={acc_now[20]:.1f}% | "
                  f"pool_recall={sum(pool_recall)/len(pool_recall)*100:.1f}% | "
                  f"ETA={eta/60:.1f}min | tok_in={total_in} tok_out={total_out}",
                  flush=True)

    out_per_q = OUT_DIR / "round30_per_question.csv"
    out_summary = OUT_DIR / "round30_summary.csv"
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