"""
Round 22: MMR-style dense-diversity rerank.

For each question:
1. Take RRF top-100
2. Iteratively pick highest-scoring doc
3. Penalize remaining docs by their max dense sim to already-picked set
4. This forces semantic diversity in top-10

If gold is in top-100 but not top-10 due to high-similarity near-duplicates,
MMR should surface it.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
import numpy as np

ROOT = Path("/data/projects/rag")
A_PLUS = ROOT / "data/exp_mq_rrf_500/results/A+_rankings.jsonl"
BM25_TOPK = ROOT / "data/bm25_topk_docids.jsonl"
QEMB = ROOT / "data/jina_v5_query_emb.npy"
QUESTIONS = ROOT / "data/questions.jsonl"
EXPECTED = ROOT / "data/expected_doc_ids.json"
INDEX = ROOT / "data/exp_full_cluster_v5/index"
META_NPY = Path("/tmp/v5_meta.npy")
VEC_NPY = Path("/tmp/v5_vectors.npy")

DSID_RE = re.compile(r"(dsid_[a-f0-9]+)")
def get_dsid(p): return (DSID_RE.search(p) or [None]).group(1) or ""


def load_eval_qs():
    qs = [json.loads(l) for l in open(QUESTIONS)]
    expected = set(json.load(open(EXPECTED)))
    return [{"qid": q["question_id"], "row_idx": i,
             "golds_set": set(g for g in q["expected_doc_ids"] if g in expected)}
            for i, q in enumerate(qs) if any(g in expected for g in q["expected_doc_ids"])]


def acc_any(top_k, golds, k):
    return any(get_dsid(t) in golds for t in top_k[:k])


METRICS_AT_K = (5, 10, 15, 20, 50, 100)


def main():
    t0 = time.time()
    print("[load]", flush=True)
    aplus = {json.loads(ln)["question_id"]: json.loads(ln)["ranked_ids"] for ln in open(A_PLUS)}
    bm25 = {json.loads(ln)["question_id"]: json.loads(ln)["top_100_ids"] for ln in open(BM25_TOPK)}
    eval_qs = load_eval_qs()
    qemb = np.load(QEMB, mmap_mode="r")
    meta_npy = np.load(META_NPY, allow_pickle=True)
    n = len(meta_npy)
    meta_ids = np.empty(n, dtype=object)
    for i in range(n): meta_ids[i] = str(meta_npy[i][0])
    doc_idx_lookup = {meta_ids[i]: i for i in range(n)}
    vecs = np.load(VEC_NPY, mmap_mode="r")

    def rrf_top(qid, ap_p=200, bm_p=500, ks=30, w_ap=1.5, w_bm=1.0):
        a = aplus[qid][:ap_p]; b = bm25[qid][:bm_p]
        s = {}
        for i, d in enumerate(a): s[d] = s.get(d, 0) + w_ap / (ks + i + 1)
        for i, d in enumerate(b): s[d] = s.get(d, 0) + w_bm / (ks + i + 1)
        return sorted(s.items(), key=lambda x: -x[1])

    def evaluate(strategy_fn, params):
        rows = [(q, strategy_fn(q)) for q in eval_qs]
        n_total = len(rows)
        m = {"strategy": strategy_fn.__name__, "params": params, "n": n_total}
        for k in METRICS_AT_K:
            m[f"acc_any@{k}"] = sum(acc_any(r, q["golds_set"], k) for q, r in rows) / n_total * 100
        return m

    def s_baseline(q):
        return [d for d, _ in rrf_top(q["qid"])][:200]

    def s_mmr_dense(q, lam=0.5, top_n=200, take_n=200):
        """MMR with RRF score and dense similarity penalty.
        score(d) = lam * rrf_score(d) - (1-lam) * max_dense_sim(d, picked)
        """
        ranked = rrf_top(q["qid"])[:top_n]
        cands = [d for d, _ in ranked]
        rrf_scores_arr = np.array([s for _, s in ranked], dtype=np.float32)
        rrf_norm = (rrf_scores_arr - rrf_scores_arr.min()) / (rrf_scores_arr.max() - rrf_scores_arr.min() + 1e-9)
        # Pre-compute dense vectors for all candidates
        cand_arr = np.array([doc_idx_lookup.get(d, -1) for d in cands], dtype=np.int64)
        valid = cand_arr >= 0
        cand_vecs = np.asarray(vecs[cand_arr[valid]])
        # Dense similarity matrix
        sims = cand_vecs @ cand_vecs.T  # NxN
        np.fill_diagonal(sims, -1)  # don't self-penalize
        # MMR iteration
        picked = []
        remaining = list(range(len(cands)))
        max_sim_to_picked = np.zeros(len(cands), dtype=np.float32)
        for _ in range(min(take_n, len(cands))):
            if not remaining:
                break
            scores = lam * rrf_norm - (1 - lam) * max_sim_to_picked
            # Mask picked
            scores[picked] = -np.inf
            best = remaining[np.argmax(scores[remaining])]
            picked.append(best)
            remaining.remove(best)
            # Update max sim to picked
            max_sim_to_picked = np.maximum(max_sim_to_picked, sims[best])
        return [cands[i] for i in picked]

    def s_mmr_query(q, lam=0.5, top_n=200, take_n=200):
        """MMR with query similarity as penalty.
        score(d) = lam * rrf_score(d) - (1-lam) * max_dense_sim_to_query_of_picked
        """
        ranked = rrf_top(q["qid"])[:top_n]
        cands = [d for d, _ in ranked]
        rrf_scores_arr = np.array([s for _, s in ranked], dtype=np.float32)
        rrf_norm = (rrf_scores_arr - rrf_scores_arr.min()) / (rrf_scores_arr.max() - rrf_scores_arr.min() + 1e-9)
        cand_arr = np.array([doc_idx_lookup.get(d, -1) for d in cands], dtype=np.int64)
        valid = cand_arr >= 0
        cand_vecs = np.asarray(vecs[cand_arr[valid]])
        vec = np.asarray(qemb[q["row_idx"]])
        sims_to_q = cand_vecs @ vec
        picked = []
        remaining = list(range(len(cands)))
        max_sim_to_picked = np.zeros(len(cands), dtype=np.float32)
        for _ in range(min(take_n, len(cands))):
            if not remaining:
                break
            scores = lam * rrf_norm - (1 - lam) * max_sim_to_picked
            scores[picked] = -np.inf
            best = remaining[np.argmax(scores[remaining])]
            picked.append(best)
            remaining.remove(best)
            max_sim_to_picked = np.maximum(max_sim_to_picked, sims_to_q)
        return [cands[i] for i in picked]

    def s_boost_neg(q, lam=0.5, top_n=200, take_n=200):
        """Penalize docs similar to TOP-K RRF docs."""
        ranked = rrf_top(q["qid"])[:top_n]
        cands = [d for d, _ in ranked]
        rrf_scores_arr = np.array([s for _, s in ranked], dtype=np.float32)
        rrf_norm = (rrf_scores_arr - rrf_scores_arr.min()) / (rrf_scores_arr.max() - rrf_scores_arr.min() + 1e-9)
        cand_arr = np.array([doc_idx_lookup.get(d, -1) for d in cands], dtype=np.int64)
        valid = cand_arr >= 0
        cand_vecs = np.asarray(vecs[cand_arr[valid]])
        sims = cand_vecs @ cand_vecs.T
        np.fill_diagonal(sims, -1)
        # For each doc, score = rrf_norm - lam * max_sim_to_top_k(rrf)
        top_k_for_neg = 10
        neg_ref = sims[:top_k_for_neg].max(axis=0)
        scores = rrf_norm - lam * neg_ref
        order = np.argsort(-scores)
        return [cands[i] for i in order]

    results = []
    print("\n[baseline RRF]", flush=True)
    r = evaluate(s_baseline, {})
    results.append(r)
    print(f"  K@10={r['acc_any@10']:5.1f}  K@15={r['acc_any@15']:5.1f}  K@100={r['acc_any@100']:5.1f}", flush=True)

    print("\n[MMR dense]", flush=True)
    for lam in [0.3, 0.5, 0.7, 0.9]:
        r = evaluate(s_mmr_dense, {"lam": lam})
        results.append(r)
        print(f"  λ={lam}  K@10={r['acc_any@10']:5.1f}  K@15={r['acc_any@15']:5.1f}  K@100={r['acc_any@100']:5.1f}", flush=True)

    print("\n[MMR query]", flush=True)
    for lam in [0.1, 0.3, 0.5, 0.7]:
        r = evaluate(s_mmr_query, {"lam": lam})
        results.append(r)
        print(f"  λ={lam}  K@10={r['acc_any@10']:5.1f}  K@15={r['acc_any@15']:5.1f}  K@100={r['acc_any@100']:5.1f}", flush=True)

    print("\n[Boost-negative]", flush=True)
    for lam in [0.05, 0.1, 0.2, 0.5]:
        r = evaluate(s_boost_neg, {"lam": lam})
        results.append(r)
        print(f"  λ={lam}  K@10={r['acc_any@10']:5.1f}  K@15={r['acc_any@15']:5.1f}  K@100={r['acc_any@100']:5.1f}", flush=True)

    print(f"\n[done] {len(results)} strategies in {time.time()-t0:.1f}s", flush=True)
    by_k10 = sorted(results, key=lambda r: -(r["acc_any@10"] + r["acc_any@15"]))
    print("\nTop 15 by K@10 + K@15:", flush=True)
    for r in by_k10[:15]:
        print(f"  {r['strategy']:24s} {json.dumps(r['params']):40s}  "
              f"K@10={r['acc_any@10']:5.1f}  K@15={r['acc_any@15']:5.1f}  K@100={r['acc_any@100']:5.1f}",
              flush=True)

    with open(ROOT / "data/exp_cluster_filter_rerank/round22_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()