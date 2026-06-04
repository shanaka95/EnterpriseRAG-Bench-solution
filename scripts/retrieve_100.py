#!/usr/bin/env python3
"""
End-to-end retrieval: user question → top-100 refined doc IDs.

Pipeline (RRF fusion of jina-v3 dense + BM25 sparse):
  1. User question (text)
  2. jina-embeddings-v3 (task=retrieval.query) → vector
  3. LanceDB flat L2 search over 511,962 docs → top-2000 jv-ranked doc IDs
  4. BM25 (bm25s Lucene) on full corpus → top-2000 bm-ranked doc IDs
  5. RRF fusion: score(d) = 1/(60 + rank_jv(d)) + 1/(60 + rank_bm(d))
  6. Sort by RRF score desc → take top-K (default 100)
  7. Output: list of dicts with doc_id, source, path, rrf_score, jv_rank, bm_rank

This is the recommended production pipeline for top-100 high-precision
retrieval (83.4% hit@100, vs 78.2% for ColBERT rerank at the same K).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/data/projects/rag/backend")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# RRF best config from §32 of the report
JV_TOP_N = 2000
BM_TOP_N = 2000
K0 = 60
K_VALUES = (10, 20, 50, 100, 200, 500, 1000)
BM_K_VALUES = (100, 500, 1000, 2000, 5000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="user question")
    ap.add_argument("--k", type=int, default=100, help="top-K to return (default 100)")
    ap.add_argument("--show-text", action="store_true",
                    help="include the first 200 chars of each doc's text")
    ap.add_argument("--all-ks", action="store_true",
                    help="print hit@K for all K_VALUES against the same RRF ranking")
    ap.add_argument("--from-cached", action="store_true",
                    help="use the saved RRF full ranking file (qid via --qid)")
    ap.add_argument("--qid", help="for --from-cached: question id from questions.jsonl")
    ap.add_argument("--corpus-dir", default="/data/projects/rag/data/all_documents")
    ap.add_argument("--questions", default="/data/projects/rag/data/questions.jsonl")
    args = ap.parse_args()

    t_total = time.time()

    # ---------- Stage 1: jina-v3 dense retrieval ----------
    if args.from_cached:
        # Use the pre-computed RRF ranking from the saved JSONL
        log("loading saved RRF full ranking (jina-v3 + BM25 already computed) …")
        path = "/data/projects/rag/data/rrf_full_ranking_N2000_k060.jsonl"
        target_qid = args.qid
        # Find the question by text match if --qid not given
        if not target_qid:
            with open(args.questions) as f:
                for ln in f:
                    q = json.loads(ln)
                    if q.get("question", "").strip() == args.query.strip():
                        target_qid = q.get("question_id")
                        break
        if not target_qid:
            log(f"ERROR: could not find question. Use --qid or pass exact text.")
            sys.exit(1)
        with open(path) as f:
            found = False
            for ln in f:
                rec = json.loads(ln)
                if rec["question_id"] == target_qid:
                    ranked_ids = rec["ranked_doc_ids"]
                    scores_list = rec["rrf_scores"]
                    jv_rank_map = bm_rank_map = {}  # not in saved file
                    expected = rec["expected_doc_ids"]
                    log(f"  found {target_qid}: {rec['question'][:80]}")
                    found = True
                    break
        if not found:
            log(f"ERROR: qid {target_qid} not found in {path}")
            sys.exit(1)
        t_jv = t_bm = 0.05  # cached, essentially free
        t_rrf = time.time() - t_total

    else:
        # ---------- Stage 1A: jina-v3 dense retrieval ----------
        log("=== Stage 1A: jina-embeddings-v3 (retrieval.query) ===")
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        import lancedb
        model = SentenceTransformer("jinaai/jina-embeddings-v3",
                                     trust_remote_code=True, device="cpu")
        t_load_jv = time.time() - t0
        log(f"  model loaded in {t_load_jv:.1f}s")

        t0 = time.time()
        qvec = model.encode(
            [args.query], task="retrieval.query", batch_size=1,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )[0]
        t_enc_jv = time.time() - t0
        log(f"  query encoded in {t_enc_jv*1000:.0f}ms (1024-dim)")

        t0 = time.time()
        table = lancedb.connect("/data/projects/rag/data/dense_index/db").open_table("documents")
        hits = table.search(qvec.tolist()).limit(JV_TOP_N).to_list()
        jv_ranked = [h["id"] for h in hits]
        t_jv = time.time() - t0
        log(f"  LanceDB search over 511,962 docs → {len(jv_ranked)} candidates in {t_jv*1000:.0f}ms")

        # ---------- Stage 1B: BM25 sparse retrieval ----------
        log("=== Stage 1B: BM25 (Lucene, k1=1.5, b=0.75) ===")
        t0 = time.time()
        import bm25s
        # Build index (one-time, then cached)
        cache = "/tmp/bm25s_index_cache"
        if not os.path.exists(cache):
            log("  building BM25 index (one-time, ~2 min)…")
            texts, ids = [], []
            for fp in sorted(Path("/data/projects/rag/data/all_documents").rglob("*.txt")):
                ids.append(fp.relative_to("/data/projects/rag/data/all_documents").as_posix())
                texts.append(fp.read_text(encoding="utf-8", errors="replace"))
            corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
            bm25 = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
            bm25.index(corpus_tokens, show_progress=False)
            with open(os.path.join(cache, "ids.json"), "w") as f:
                json.dump(ids, f)
            del texts
        else:
            ids = json.load(open(os.path.join(cache, "ids.json")))
            # We don't actually cache the index here (bm25s has no native save);
            # for simplicity, always rebuild. Production should use a persistent store.
            log("  (rebuilding BM25 index — no persistent cache yet)")
            texts = []
            for did in ids:
                with open(f"/data/projects/rag/data/all_documents/{did}", encoding="utf-8", errors="replace") as fh:
                    texts.append(fh.read())
            corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
            bm25 = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
            bm25.index(corpus_tokens, show_progress=False)
        t_load_bm = time.time() - t0
        log(f"  BM25 index ready ({len(ids):,} docs) in {t_load_bm:.1f}s")

        t0 = time.time()
        qtok = bm25s.tokenize([args.query], stopwords="en", show_progress=False)
        bm_results = bm25.retrieve(qtok, corpus=ids, k=BM_TOP_N, show_progress=False)
        bm_ranked = [str(d) for d in bm_results.documents[0]]
        t_bm = time.time() - t0
        log(f"  BM25 scored {len(bm_ranked)} candidates in {t_bm*1000:.0f}ms")

        # ---------- Stage 2: RRF fusion ----------
        log("=== Stage 2: RRF fusion (k0=60) ===")
        t0 = time.time()
        scores: dict[str, float] = defaultdict(float)
        for rank, d in enumerate(jv_ranked, 1):
            scores[d] += 1.0 / (K0 + rank)
        for rank, d in enumerate(bm_ranked, 1):
            scores[d] += 1.0 / (K0 + rank)
        jv_rank_map = {d: i for i, d in enumerate(jv_ranked, 1)}
        bm_rank_map = {d: i for i, d in enumerate(bm_ranked, 1)}
        ranked_ids = sorted(scores.keys(),
                            key=lambda d: (-scores[d], jv_rank_map.get(d, 1e9), bm_rank_map.get(d, 1e9)))
        scores_list = [scores[d] for d in ranked_ids]
        t_rrf = time.time() - t0
        log(f"  RRF fused {len(ranked_ids)} unique candidates in {t_rrf*1000:.0f}ms")

        expected = None

    # ---------- Stage 3: top-K selection ----------
    log(f"=== Stage 3: top-{args.k} selection ===")
    top = ranked_ids[:args.k]

    # ---------- Output ----------
    t_total = time.time() - t_total
    log(f"=== TOTAL: {t_total*1000:.0f}ms ===")

    print()
    print("=" * 90)
    print(f"RETRIEVED top-{args.k} docs for query: {args.query!r}")
    print("=" * 90)
    print()
    for j, (doc_id, sc) in enumerate(zip(top, scores_list[:args.k]), 1):
        source = doc_id.split("/")[0] if "/" in doc_id else "?"
        jv_r = jv_rank_map.get(doc_id, "—")
        bm_r = bm_rank_map.get(doc_id, "—")
        exp_marker = "  ★ EXPECTED" if (expected and any(e in doc_id for e in expected)) else ""
        print(f"  {j:>3}. RRF={sc:.5f}  jv#{jv_r} bm#{bm_r}  [{source}]  {doc_id}{exp_marker}")

    if args.show_text:
        print()
        print("=" * 90)
        print("Document text previews (first 200 chars each):")
        print("=" * 90)
        for j, doc_id in enumerate(top, 1):
            fp = os.path.join(args.corpus_dir, doc_id)
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    text = f.read(200).replace("\n", " ").strip()
            except FileNotFoundError:
                text = "(file not found)"
            print(f"\n  {j}. {doc_id}\n     {text}…")

    if args.all_ks and expected:
        print()
        print("=" * 90)
        print("hit@K against expected doc(s) — for ALL K values from same RRF ranking:")
        print("=" * 90)
        for k in K_VALUES:
            top_k = ranked_ids[:k]
            hit = any(any(e in d for e in expected) for d in top_k)
            mark = "✓" if hit else "✗"
            print(f"  {mark}  K={k:>4}  {'HIT' if hit else 'miss'}")


if __name__ == "__main__":
    main()
