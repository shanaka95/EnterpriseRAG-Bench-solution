#!/usr/bin/env python3
"""
Compare the two chunking algorithms on N docs from the viz corpus.

For each chosen doc, runs both `recurse_chunk` (token-level MaxSim) and
`recurse_chunk_semantic` (sentence-level cosine distance) and prints:

  - Number of leaves, avg tokens/leaf, max depth
  - Per-leaf end-character (does it end on a sentence terminator?)
  - Side-by-side preview of the first 2 leaves from each algorithm

Output is human-readable, not a unit test. Run after the build to see
qualitative differences.

Usage:
    /data/projects/rag/backend/venv/bin/python scripts/compare_chunkers.py
    /data/projects/rag/backend/venv/bin/python scripts/compare_chunkers.py --n-docs 10
    /data/projects/rag/backend/venv/bin/python scripts/compare_chunkers.py --doc-id dsid_abc...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make backend importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import numpy as np
import pyarrow.parquet as pq

from app.ml.raptor_chunking import (
    recurse_chunk, recurse_chunk_semantic, make_node_id,
    COLBERT_MAX_SEQ,
)
from build_raptor_viz_graph import (
    fetch_colbert_for_doc, get_jina_tokenizer, VIZ_DIR,
)


def _ends_on_terminator(text: str) -> bool:
    """Return True if `text` ends on a sentence terminator (broadened).

    In addition to standard '.!?', we accept:
      - newline (typical end-of-slack-message boundary)
      - close-paren/bracket that closes a sentence like "(see https://x.com)"
      - colon (end of a label like "Key snippets:")
      - em-dash (end of a clause that was interrupted)
    """
    if not text:
        return False
    stripped = text.rstrip()
    if not stripped:
        return False
    last = stripped[-1]
    if last in '.!?")]}':
        return True
    if last == '\n' and len(stripped) > 1:
        # newline-terminated: only count if previous char was a sentence-end-ish char
        prev = stripped[-2]
        if prev in '.!?":—-*`':
            return True
        # Otherwise: it's a message boundary in slack/jira which is also fine
        return True
    return last in ':—-'


def _chunks_per_doc(text: str, colbert_vecs, char_offsets, doc_id: str,
                    chunking: str):
    """Run the requested chunker and return the flat list of ChunkNodes."""
    if chunking == "semantic":
        return recurse_chunk_semantic(
            text=text,
            start_char=0, end_char=len(text),
            colbert_vecs=colbert_vecs,
            char_offsets=char_offsets,
            level=0, parent_id=None,
            sibling_idx=0, n_siblings=1,
            doc_id=doc_id,
        )
    else:
        return recurse_chunk(
            start_tok=0, end_tok=len(char_offsets),
            colbert_vecs=colbert_vecs,
            char_offsets=char_offsets,
            start_char=0, end_char=len(text),
            level=0, parent_id=None,
            sibling_idx=0, n_siblings=1,
            doc_id=doc_id,
        )


def _stats(nodes, text: str) -> dict:
    leaves = [n for n in nodes if n.is_leaf]
    leaf_texts = [text[n.start_char:n.end_char] for n in leaves]
    leaf_tok_counts = [n.n_tokens_colbert for n in leaves]
    ends_on_term = sum(1 for lt in leaf_texts if _ends_on_terminator(lt))
    max_level = max((n.level for n in nodes), default=0)
    return {
        "total_nodes": len(nodes),
        "leaves": len(leaves),
        "avg_tokens": float(np.mean(leaf_tok_counts)) if leaf_tok_counts else 0,
        "min_tokens": min(leaf_tok_counts) if leaf_tok_counts else 0,
        "max_tokens": max(leaf_tok_counts) if leaf_tok_counts else 0,
        "max_level": max_level,
        "leaves_ending_on_terminator": ends_on_term,
        "leaves_pct_on_terminator": (ends_on_term / max(1, len(leaves))) * 100,
    }


def _format_leaves(nodes, text: str, n: int = 3) -> str:
    """Format the first N leaves for display."""
    leaves = sorted([nd for nd in nodes if nd.is_leaf], key=lambda x: x.start_char)
    lines = []
    for i, leaf in enumerate(leaves[:n]):
        chunk = text[leaf.start_char:leaf.end_char].replace("\n", " ").strip()
        term = "✓" if _ends_on_terminator(chunk) else "✗"
        snippet = chunk[:80] + ("…" if len(chunk) > 80 else "")
        lines.append(f"    L{leaf.level} tok={leaf.n_tokens_colbert:4d} end-on-term={term}  {snippet!r}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-docs", type=int, default=3,
                    help="Number of docs to compare (default 3)")
    ap.add_argument("--doc-id", type=str, default=None,
                    help="Specific doc_id to compare (overrides --n-docs)")
    ap.add_argument("--source", type=str, default=None,
                    help="Filter docs by source_type (e.g. 'slack', 'github')")
    ap.add_argument("--skip-both", action="store_true",
                    help="Skip docs where both algorithms produce identical results")
    ap.add_argument("--print-leaves", type=int, default=3,
                    help="Number of leaves to preview per algorithm per doc (default 3)")
    args = ap.parse_args()

    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    log(f"Loading docs from {VIZ_DIR}/docs.parquet …")
    docs_path = Path(VIZ_DIR) / "docs.parquet"
    if not docs_path.exists():
        log(f"FATAL: {docs_path} not found. Run build_raptor_viz_graph.py first.")
        sys.exit(1)
    table = pq.read_table(docs_path)
    docs = {r["doc_id"]: dict(r) for r in table.to_pylist()}
    log(f"  loaded {len(docs)} docs")

    # Pick which docs to compare
    if args.doc_id:
        if args.doc_id not in docs:
            log(f"FATAL: doc_id {args.doc_id} not in corpus")
            sys.exit(1)
        targets = [args.doc_id]
    else:
        all_ids = list(docs.keys())
        if args.source:
            all_ids = [d for d in all_ids if docs[d].get("source_type") == args.source]
        # Sort by n_chars desc so we test on meaty docs first
        all_ids.sort(key=lambda d: docs[d].get("n_chars", 0), reverse=True)
        targets = all_ids[:args.n_docs]
    log(f"Comparing {len(targets)} doc(s)")

    # Tokenizer
    tok = get_jina_tokenizer()

    grand_total_token_time = 0.0
    grand_total_token_leaves = 0
    grand_total_semantic_time = 0.0
    grand_total_semantic_leaves = 0
    grand_token_terminators = 0
    grand_semantic_terminators = 0
    grand_token_leaf_count = 0
    grand_semantic_leaf_count = 0

    for doc_id in targets:
        d = docs[doc_id]
        text = d["content"]
        if not text or not text.strip():
            log(f"  {doc_id}: SKIP (empty text)")
            continue
        log(f"")
        log(f"{'=' * 80}")
        log(f"doc_id: {doc_id}")
        log(f"  title: {d.get('title', '?')[:60]}")
        log(f"  source: {d.get('source_type', '?')}")
        log(f"  n_chars: {len(text):,}")

        # Fetch ColBERT
        colbert_vecs = fetch_colbert_for_doc(doc_id)
        if colbert_vecs is None or colbert_vecs.shape[0] == 0:
            log(f"  SKIP (no ColBERT vectors)")
            continue

        # Tokenize
        enc = tok(
            text, return_offsets_mapping=True, add_special_tokens=False,
            truncation=True, max_length=COLBERT_MAX_SEQ,
        )
        offsets = enc["offset_mapping"]
        n_tokens = min(len(offsets), colbert_vecs.shape[0])
        offsets = offsets[:n_tokens]

        # ---- TOKEN-LEVEL CHUNKER ----
        t0 = time.time()
        try:
            token_nodes = _chunks_per_doc(text, colbert_vecs, offsets, doc_id, "token")
        except Exception as e:
            log(f"  token chunker ERROR: {e}")
            token_nodes = []
        token_time = time.time() - t0
        token_stats = _stats(token_nodes, text)

        # ---- SEMANTIC CHUNKER ----
        t0 = time.time()
        try:
            semantic_nodes = _chunks_per_doc(text, colbert_vecs, offsets, doc_id, "semantic")
        except Exception as e:
            log(f"  semantic chunker ERROR: {e}")
            semantic_nodes = []
        semantic_time = time.time() - t0
        semantic_stats = _stats(semantic_nodes, text)

        # Report
        log(f"")
        log(f"  TOKEN chunker:    {token_stats['leaves']} leaves, "
            f"avg {token_stats['avg_tokens']:.0f} tok/leaf, "
            f"max_depth {token_stats['max_level']}, "
            f"{token_stats['leaves_pct_on_terminator']:.0f}% end on terminator, "
            f"{token_time:.2f}s")
        log(f"  SEMANTIC chunker: {semantic_stats['leaves']} leaves, "
            f"avg {semantic_stats['avg_tokens']:.0f} tok/leaf, "
            f"max_depth {semantic_stats['max_level']}, "
            f"{semantic_stats['leaves_pct_on_terminator']:.0f}% end on terminator, "
            f"{semantic_time:.2f}s")

        log(f"")
        log(f"  --- TOKEN chunker first {args.print_leaves} leaves ---")
        print(_format_leaves(token_nodes, text, args.print_leaves))
        log(f"")
        log(f"  --- SEMANTIC chunker first {args.print_leaves} leaves ---")
        print(_format_leaves(semantic_nodes, text, args.print_leaves))

        # Aggregate
        grand_total_token_time += token_time
        grand_total_token_leaves += token_stats['leaves']
        grand_token_terminators += token_stats['leaves_ending_on_terminator']
        grand_token_leaf_count += token_stats['leaves']
        grand_total_semantic_time += semantic_time
        grand_total_semantic_leaves += semantic_stats['leaves']
        grand_semantic_terminators += semantic_stats['leaves_ending_on_terminator']
        grand_semantic_leaf_count += semantic_stats['leaves']

    log(f"")
    log(f"{'=' * 80}")
    log(f"SUMMARY ({len(targets)} docs)")
    log(f"  TOKEN    : {grand_total_token_time:.2f}s total, "
        f"{grand_total_token_leaves} leaves, "
        f"{grand_token_terminators}/{grand_token_leaf_count} "
        f"({(grand_token_terminators / max(1, grand_token_leaf_count)) * 100:.0f}%) end on terminator")
    log(f"  SEMANTIC : {grand_total_semantic_time:.2f}s total, "
        f"{grand_total_semantic_leaves} leaves, "
        f"{grand_semantic_terminators}/{grand_semantic_leaf_count} "
        f"({(grand_semantic_terminators / max(1, grand_semantic_leaf_count)) * 100:.0f}%) end on terminator")


if __name__ == "__main__":
    main()
