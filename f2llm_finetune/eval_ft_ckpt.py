#!/usr/bin/env python3
"""Evaluate any F2LLM checkpoint on the 470-question 5K-corpus benchmark.

Identical protocol to the baseline run (eval_f2llm_5k.py):
  - docs = raw content, no prompt, last-token pooling, L2-norm
  - queries = QUERY_PROMPT + question
  - exact cosine search (float32 dot on L2-normalized vectors), full ranking
  - hit@K (acc_any) for K = 1,5,10,20,50,75,100,200,500,600,1000,2000, n=470
  - per-type breakdown + per-question ranks

Usage:
  eval_ft_ckpt.py --model <hf_id_or_dir> [--tag NAME] [--max-length 8192]
Outputs: /workspace/rag5k/eval_out/<tag>_summary.json and <tag>_per_question.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

HF_DATASET = "onyx-dot-app/EnterpriseRAG-Bench"
HF_REVISION = "69916e31c68aa5963c00248fd7f0bc12d04fd235"
IDS_PATH = "/workspace/rag5k/corpus_5k_doc_ids.json"
QUESTIONS_PATH = "/workspace/rag5k/questions.jsonl"
OUT_DIR = "/workspace/rag5k/eval_out"
KS = [1, 5, 10, 20, 50, 75, 100, 200, 500, 600, 1000, 2000]
QUERY_PROMPT = (
    "Instruct: Given a question, retrieve passages that can help answer "
    "the question.\nQuery: "
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_corpus(ids: list[str]) -> tuple[list[str], list[str]]:
    ds = load_dataset(HF_DATASET, "documents", split="test", revision=HF_REVISION)
    idx: dict[str, int] = {}
    for i, d in enumerate(ds["doc_id"]):
        if d not in idx:
            idx[d] = i
    missing = [d for d in ids if d not in idx]
    if missing:
        raise SystemExit(f"{len(missing)} corpus ids missing: {missing[:5]}")
    sub = ds.select([idx[d] for d in ids])
    return list(sub["content"]), list(sub["doc_id"])


@torch.inference_mode()
def encode(model, tokenizer, texts: list[str], max_length: int, token_budget: int,
           desc: str) -> np.ndarray:
    """Length-sorted token-budget batching; returns float32 L2-normed (N, D)."""
    enc_all = tokenizer(texts, add_special_tokens=True)
    lens = np.array([len(x) for x in enc_all["input_ids"]])
    order = np.argsort(lens)
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    i, nb = 0, 0
    t0 = time.time()
    while i < len(order):
        j = i
        while j < len(order):
            width = min(int(lens[order[j]]), max_length)
            if (j - i + 1) * width > token_budget and j > i:
                break
            j += 1
        bidx = order[i:j]
        enc = tokenizer([texts[k] for k in bidx], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(model.device)
        hs = model(**enc).last_hidden_state
        eos_pos = enc.attention_mask.sum(dim=1) - 1
        e = hs[torch.arange(len(bidx), device=model.device), eos_pos]
        out[bidx] = F.normalize(e.float(), p=2, dim=1).cpu().numpy()
        nb += 1
        if nb % 25 == 0 or j == len(order):
            log(f"  {desc}: {j}/{len(order)} ({time.time() - t0:.0f}s)")
        i = j
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--token-budget", type=int, default=16384)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    ids = json.load(open(IDS_PATH))

    log(f"loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16,
                                      device_map={"": 0})
    model.eval()

    contents, doc_ids = load_corpus(ids)
    log(f"corpus: {len(contents)} docs; encoding (max_length={args.max_length}) ...")
    doc_emb = encode(model, tokenizer, contents, args.max_length,
                     args.token_budget, "docs")

    qs = [json.loads(l) for l in open(QUESTIONS_PATH)]
    answerable = [q for q in qs if q.get("expected_doc_ids")]
    log(f"questions: {len(qs)} total, {len(answerable)} answerable")
    q_emb = encode(model, tokenizer,
                   [QUERY_PROMPT + q["question"] for q in answerable],
                   2048, 8192, "queries")

    pos = {d: i for i, d in enumerate(doc_ids)}
    sims = q_emb @ doc_emb.T
    ranks_desc = np.argsort(-sims, axis=1, kind="stable")

    hits = {k: 0 for k in KS}
    mrr = 0.0
    rows = []
    by_type = defaultdict(lambda: {"n": 0, **{f"hit@{k}": 0 for k in KS}})
    for r, q in enumerate(answerable):
        ranking = ranks_desc[r]
        best = min(int(np.where(ranking == pos[d])[0][0]) for d in q["expected_doc_ids"])
        rank1 = best + 1
        mrr += 1.0 / rank1
        t = by_type[q.get("question_type", "unknown")]
        t["n"] += 1
        for k in KS:
            if best < k:
                hits[k] += 1
                t[f"hit@{k}"] += 1
        rows.append({"question_id": q["question_id"],
                     "question_type": q.get("question_type", ""),
                     "gold_rank": rank1})

    n = len(answerable)
    summary = {
        "tag": args.tag, "model": args.model, "n": n,
        "mrr": mrr / n,
        "hit@K": {k: {"count": hits[k], "pct": 100 * hits[k] / n} for k in KS},
        "per_type": {t: {"n": v["n"],
                         **{f"hit@{k}": 100 * v[f"hit@{k}"] / v["n"] for k in (1, 10, 100)}}
                     for t, v in sorted(by_type.items())},
        "elapsed_s": time.time() - t0,
    }
    spath = f"{OUT_DIR}/{args.tag}_summary.json"
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    cpath = f"{OUT_DIR}/{args.tag}_per_question.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "question_type", "gold_rank"])
        w.writeheader()
        w.writerows(rows)

    log(f"MRR={mrr / n:.4f}   " + "  ".join(
        f"hit@{k}={100 * hits[k] / n:.2f}%" for k in (1, 5, 10, 20, 100, 1000)))
    log(f"saved {spath} + {cpath}   TOTAL {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
