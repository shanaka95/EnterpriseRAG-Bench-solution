#!/usr/bin/env python3
"""Build F2LLM-format finetuning data for the 5K EnterpriseRAG corpus.

Runs ON THE REMOTE server (needs GPU + cached HF datasets). Produces:
  /workspace/rag5k/train_data/corpus.parquet           [doc_id, input_ids]
  /workspace/rag5k/train_data/enterprise_query.parquet [query_input_ids, passage_input_ids,
                                                        negative_1..24_input_ids]

Convention follows the original F2LLM repo (tokenize_data_qwen.py + run.py):
  - all texts tokenized with truncation, then EOS appended (repo appends EOS
    even though the Qwen tokenizer already adds one -> training data ends
    with the same double-EOS the model was originally trained on)
  - passage_input_ids / negative_i_input_ids are DOC-ID STRINGS referencing
    corpus.parquet (run.py resolves them via get_corpus_ids)
  - queries get the retrieval instruction prompt prepended, matching the
    eval protocol used for the 470-question benchmark

Hard negatives: 24 per query, mined with the BASE F2LLM-v2-80M against the
pre-computed 5K doc embeddings (f2llm_v2_80m_5k.npz, rows aligned to
corpus_5k_doc_ids.json): top-ranked docs excluding the positive.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

HF_DATASET = "onyx-dot-app/EnterpriseRAG-Bench"
HF_REVISION = "69916e31c68aa5963c00248fd7f0bc12d04fd235"
MODEL_PATH = "codefuse-ai/F2LLM-v2-80M"
IDS_PATH = "/workspace/rag5k/corpus_5k_doc_ids.json"
QUESTIONS_PARQUET = "/workspace/rag5k/enterprise_rag_questions.parquet"
DOC_EMB_NPZ = "/workspace/rag5k/f2llm_v2_80m_5k.npz"
OUT_DIR = "/workspace/rag5k/train_data"
QUERY_PROMPT = (
    "Instruct: Given a question, retrieve passages that can help answer "
    "the question.\nQuery: "
)
MAX_SEQ_LENGTH = 4096      # doc truncation at tokenize time (matches training config)
QUERY_MAX_LEN = 511        # synthetic questions are short; +EOS -> <= 512
NUM_NEG = 24               # run.py samples from exactly 24 negatives
SEED = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tok_ids(tokenizer, text: str, max_length: int) -> np.ndarray:
    """Repo convention: truncate, then append EOS."""
    ids = tokenizer(text, max_length=max_length, truncation=True)["input_ids"]
    return np.array(ids + [tokenizer.eos_token_id], dtype=np.int64)


def load_questions(ids5k: set[str]) -> pd.DataFrame:
    """25K (doc_id, question) rows; sample 5 per doc deterministically."""
    df = pd.read_parquet(QUESTIONS_PARQUET)
    df = df[df["doc_id"].isin(ids5k)].reset_index(drop=True)
    log(f"questions for 5K docs: {len(df)} rows, {df['doc_id'].nunique()} docs")
    rng = random.Random(SEED)
    keep_idx = []
    for doc_id, grp in df.groupby("doc_id", sort=False):
        idxs = grp.index.tolist()
        keep_idx.extend(idxs if len(idxs) <= 5 else rng.sample(idxs, 5))
    df = df.loc[sorted(keep_idx)].reset_index(drop=True)
    counts = df["doc_id"].value_counts()
    assert (counts == 5).all(), "every doc must have exactly 5 questions"
    log(f"kept {len(df)} training pairs (5 per doc)")
    return df[["doc_id", "question"]]


def load_docs(ids5k: list[str]) -> dict[str, str]:
    """doc_id -> content, using the cached pinned documents split."""
    ds = load_dataset(HF_DATASET, "documents", split="test", revision=HF_REVISION)
    log(f"documents split: {len(ds):,} rows")
    wanted = set(ids5k)
    texts: dict[str, str] = {}
    for d in ds:
        if d["doc_id"] in wanted and d["doc_id"] not in texts:
            texts[d["doc_id"]] = d["content"]
    missing = wanted - set(texts)
    if missing:
        raise SystemExit(f"{len(missing)} docs missing from dataset: {list(missing)[:5]}")
    log(f"doc texts collected: {len(texts)}")
    return texts


@torch.inference_mode()
def embed_questions(model, tokenizer, texts: list[str], batch_size: int = 128,
                    max_length: int = 512) -> np.ndarray:
    embs = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(model.device)
        hs = model(**enc).last_hidden_state
        eos_pos = enc.attention_mask.sum(dim=1) - 1
        e = hs[torch.arange(len(chunk), device=model.device), eos_pos]
        embs[i:i + len(chunk)] = F.normalize(e.float(), p=2, dim=1).cpu().numpy()
        if (i // batch_size) % 25 == 0:
            log(f"  question embeddings: {min(i + batch_size, len(texts))}/{len(texts)}")
    return embs


def mine_hard_negatives(q_emb: np.ndarray, doc_emb: np.ndarray,
                        doc_ids: list[str], gold_doc: list[str]) -> tuple[list[list[str]], dict]:
    """Top-24 docs per query excluding the positive. Also retrieval diagnostics."""
    pos_of = {d: i for i, d in enumerate(doc_ids)}
    gold_idx = np.array([pos_of[d] for d in gold_doc])
    log("computing 25K x 5K similarity matrix ...")
    sims = q_emb @ doc_emb.T  # (N, 5000)
    log("ranking ...")
    order = np.argsort(-sims, axis=1)  # descending per row
    negs: list[list[str]] = []
    ranks_of_pos = np.zeros(len(gold_idx), dtype=np.int32)
    for r in range(len(gold_idx)):
        gi = gold_idx[r]
        ranks_of_pos[r] = int(np.where(order[r] == gi)[0][0])
        picked = []
        for c in order[r]:
            if c == gi:
                continue
            picked.append(doc_ids[int(c)])
            if len(picked) == NUM_NEG:
                break
        negs.append(picked)
    n = len(gold_idx)
    diag = {
        "n": n,
        "pos_rank_mean": float(ranks_of_pos.mean()),
        "pos_rank_median": float(np.median(ranks_of_pos)),
        "hit@1": float((ranks_of_pos == 0).mean()),
        "hit@5": float((ranks_of_pos < 5).mean()),
        "hit@10": float((ranks_of_pos < 10).mean()),
        "hit@100": float((ranks_of_pos < 100).mean()),
    }
    log("base-model retrieval of the 25K training questions (diagnostic): "
        + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in diag.items()))
    return negs, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    ids5k = json.load(open(IDS_PATH))
    assert len(ids5k) == 5000 and len(set(ids5k)) == 5000

    # ---- questions -------------------------------------------------------
    qdf = load_questions(set(ids5k))

    # ---- docs + tokenizer ------------------------------------------------
    log(f"loading tokenizer + model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                      device_map={"": 0})
    model.eval()

    texts = load_docs(ids5k)
    log("tokenizing 5K docs ...")
    corpus_rows = {d: tok_ids(tokenizer, texts[d], MAX_SEQ_LENGTH - 1) for d in ids5k}
    lens = np.array([len(v) for v in corpus_rows.values()])
    log(f"corpus tokens: mean={lens.mean():.0f} p95={np.percentile(lens, 95):.0f} "
        f"max={lens.max()} (cap {MAX_SEQ_LENGTH})")

    # ---- embed questions for mining --------------------------------------
    log("embedding 25K training questions (with prompt) ...")
    q_emb = embed_questions(model, tokenizer,
                            [QUERY_PROMPT + q for q in qdf["question"]])

    # ---- mine hard negatives ---------------------------------------------
    z = np.load(DOC_EMB_NPZ)
    assert list(z["doc_ids"]) == ids5k, "npz rows must match corpus_5k_doc_ids.json"
    negs, diag = mine_hard_negatives(q_emb, z["embeddings"], ids5k,
                                     qdf["doc_id"].tolist())
    for r in range(len(negs)):
        assert qdf["doc_id"][r] not in negs[r], "positive leaked into negatives"

    # ---- tokenize queries --------------------------------------------------
    log("tokenizing 25K queries ...")
    q_tokens = [tok_ids(tokenizer, QUERY_PROMPT + q, QUERY_MAX_LEN)
                for q in qdf["question"]]
    qlens = np.array([len(x) for x in q_tokens])
    log(f"query tokens: mean={qlens.mean():.0f} max={qlens.max()}")

    # ---- write parquets ----------------------------------------------------
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    corpus_df = pd.DataFrame({
        "doc_id": ids5k,
        "input_ids": [corpus_rows[d] for d in ids5k],
    })
    corpus_df.to_parquet(f"{args.out_dir}/corpus.parquet", index=False)
    log(f"wrote corpus.parquet: {len(corpus_df)} docs")

    qrec = {
        "query_input_ids": q_tokens,
        "passage_input_ids": qdf["doc_id"].tolist(),
    }
    for i in range(NUM_NEG):
        qrec[f"negative_{i+1}_input_ids"] = [negs[r][i] for r in range(len(negs))]
    qout = pd.DataFrame(qrec)
    qout.to_parquet(f"{args.out_dir}/enterprise_query.parquet", index=False)
    log(f"wrote enterprise_query.parquet: {len(qout)} rows x {len(qout.columns)} cols")

    # ---- verify round-trip ---------------------------------------------------
    back = pd.read_parquet(f"{args.out_dir}/enterprise_query.parquet")
    assert len(back) == len(qout)
    assert isinstance(back["query_input_ids"][0], (list, np.ndarray))
    assert back["passage_input_ids"].isin(ids5k).all()
    for i in range(1, NUM_NEG + 1):
        assert back[f"negative_{i}_input_ids"].isin(ids5k).all()
    cback = pd.read_parquet(f"{args.out_dir}/corpus.parquet")
    assert set(cback["doc_id"]) == set(ids5k)
    log("round-trip verification OK")

    with open(f"{args.out_dir}/build_info.json", "w") as f:
        json.dump({
            "num_pairs": len(qout), "num_docs": len(ids5k), "num_neg": NUM_NEG,
            "max_seq_length": MAX_SEQ_LENGTH, "query_max_len": QUERY_MAX_LEN,
            "query_prompt": QUERY_PROMPT, "seed": SEED,
            "base_model_mining_diagnostics": diag,
            "corpus_token_len_mean": float(lens.mean()),
            "corpus_token_len_max": int(lens.max()),
        }, f, indent=2)
    log(f"TOTAL {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
