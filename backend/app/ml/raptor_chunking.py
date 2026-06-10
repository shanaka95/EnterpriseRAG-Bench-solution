"""
Pure chunking logic for the RAPTOR-style hierarchical chunk graph.

Extracted from scripts/build_raptor_graph.py (originally lines 113-339) so
that BOTH the production build script and the new viz build script
(scripts/build_raptor_viz_graph.py) can share the exact same algorithm.

What lives here (no I/O, no LanceDB, no HF):
  - make_node_id:            content-addressed node id (sha1)
  - ChunkNode:               dataclass
  - adjacent_window_maxsim:  per-token ColBERT MaxSim across adjacent windows
  - find_valleys:            pick deepest local minima of the sim array
  - recurse_chunk:           the token-level recursive chunker (production)
  - semantic_chunk_sentences: sentence-level semantic chunker (viz)
  - recurse_chunk_semantic:  the sentence-level recursive chunker
  - split_token_to_char:     token->char offset helper
  - split_token_idx_for_char: char->token offset helper

What does NOT live here (it stays in the build scripts because it depends
on the specific data source / target):
  - ColBERT fetch (LanceDB)  — scripts/open their own table
  - HF dataset download       — scripts/open their own connection
  - LanceDB writer            — each script writes to its own table
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ----------------------------- constants -------------------------------------

# Chunking parameters (mirror build_raptor_graph.py defaults)
MIN_LEAF_TOKENS = 128
MAX_LEAF_TOKENS = 256
MAX_SPLIT_CHUNK_TOKENS = 2048   # aim for child <= this many tokens
WINDOW_TOKENS = 64              # ColBERT adjacent-window size
WINDOW_STRIDE = 32              # 50% overlap
MIN_VALLEY_SCORE_GAP = 0.10     # require a valley to be at least 0.10 below the max sim
MIN_SPLIT_GAP_TOKENS = MIN_LEAF_TOKENS   # never split closer than this
COLBERT_MAX_SEQ = 8192          # jina-colbert-v2 max seq len

# Semantic-chunker parameters (sentence-level)
SENTENCE_WINDOW = 3             # sentences per window
PERCENTILE_THRESHOLD = 0.90     # breakpoints above the 90th percentile
MIN_SEMANTIC_LEAF_TOKENS = MIN_LEAF_TOKENS
TARGET_SEMANTIC_LEAF_TOKENS = MAX_LEAF_TOKENS


# ----------------------------- helpers ---------------------------------------

def make_node_id(doc_id: str, level: int, start_char: int, end_char: int) -> str:
    """Content-addressed node id (sha1 prefix, 16 hex chars)."""
    h = hashlib.sha1()
    h.update(doc_id.encode("utf-8"))
    h.update(b"|")
    h.update(f"{level}:{start_char}:{end_char}".encode("utf-8"))
    return h.hexdigest()[:16]


def adjacent_window_maxsim(colbert_vecs, W: int = WINDOW_TOKENS, S: int = WINDOW_STRIDE):
    """Compute MaxSim between adjacent non-overlapping windows of tokens.

    colbert_vecs: (N, D) float32, L2-normalized per token.
    Returns: (K,) array of sims, one per adjacent-window pair, where
        K = (N // S) - 1
        sims[i] = MaxSim(window starting at i*S, window starting at (i+1)*S)
    """
    N = colbert_vecs.shape[0]
    starts = list(range(0, N, S))
    if len(starts) < 2:
        return np.zeros(0, dtype=colbert_vecs.dtype)
    windows = []
    for s in starts:
        e = min(s + W, N)
        windows.append(colbert_vecs[s:e])
    K = len(windows) - 1
    out = np.empty(K, dtype=colbert_vecs.dtype)
    # MaxSim(window_a, window_b) = sum over i of max over j of dot(a_i, b_j)
    for i in range(K):
        A = windows[i]
        B = windows[i + 1]
        sim = A @ B.T                # (|A|, |B|)
        out[i] = sim.max(axis=1).sum()
    return out


def find_valleys(sims, k: int, min_gap_idx: int) -> list[int]:
    """Find the k deepest valleys in `sims` with at least min_gap_idx spacing.

    Returns list of indices (length <= k). If fewer than k valleys exist,
    returns however many were found. Valleys must be a local minimum and
    below `sims.max() - MIN_VALLEY_SCORE_GAP`.
    """
    if sims.size == 0 or k <= 0:
        return []
    thresh = sims.max() - MIN_VALLEY_SCORE_GAP

    # Local minima: sims[i] < sims[i-1] and sims[i] < sims[i+1]
    candidates = []
    for i in range(1, len(sims) - 1):
        if sims[i] < sims[i - 1] and sims[i] < sims[i + 1] and sims[i] < thresh:
            candidates.append((float(sims[i]), i))
    if not candidates:
        return []
    # Sort by sim ASC (deepest valleys first)
    candidates.sort()
    chosen: list[int] = []
    for _, i in candidates:
        if all(abs(i - j) >= min_gap_idx for j in chosen):
            chosen.append(i)
        if len(chosen) >= k:
            break
    chosen.sort()
    return chosen


def split_token_to_char(tok_idx: int, char_offsets: list[tuple[int, int]]) -> int:
    """Return the start char of token `tok_idx` (or len(text) if past end)."""
    if tok_idx >= len(char_offsets):
        return char_offsets[-1][1] if char_offsets else 0
    return char_offsets[tok_idx][0]


def split_token_idx_for_char(char_pos: int, char_offsets: list[tuple[int, int]],
                             lo_tok: int, hi_tok: int) -> int:
    """Find the smallest tok index in [lo_tok, hi_tok] whose start_char >= char_pos.
    Fallback: hi_tok."""
    for i in range(lo_tok, min(hi_tok + 1, len(char_offsets))):
        if char_offsets[i][0] >= char_pos:
            return i
    return hi_tok


# ----------------------------- chunk node ------------------------------------

@dataclass
class ChunkNode:
    level: int
    start_char: int
    end_char: int
    n_tokens_colbert: int
    is_leaf: bool
    boundary_score: float | None   # higher = stronger
    parent_id: str | None
    sibling_idx: int
    n_siblings: int
    node_id: str = ""                       # filled after creation
    first_child_id: str | None = None       # filled during wiring
    next_sibling_id: str | None = None      # filled during wiring


# ----------------------------- recursive chunker -----------------------------

def recurse_chunk(start_tok: int, end_tok: int,
                  colbert_vecs,
                  char_offsets: list[tuple[int, int]],
                  start_char: int, end_char: int,
                  level: int, parent_id: str | None,
                  sibling_idx: int, n_siblings: int,
                  doc_id: str) -> list[ChunkNode]:
    """Recursively chunk the token range [start_tok, end_tok).

    Returns a flat list of ChunkNodes for the whole subtree (including `this`
    parent and all descendants). Children are appended in order.

    Char offsets passed in (start_char, end_char) are absolute char positions
    in the original doc. Token positions [start_tok, end_tok) are absolute
    token positions in the original doc.
    """
    n_tok = end_tok - start_tok
    # A node is a leaf if:
    #  - It's small enough (≤ MAX_LEAF_TOKENS), OR
    #  - Splitting it would produce sub-MIN_SPLIT_GAP_TOKENS children
    is_too_small_to_split = n_tok < 2 * MIN_SPLIT_GAP_TOKENS
    node = ChunkNode(
        level=level,
        start_char=start_char,
        end_char=end_char,
        n_tokens_colbert=n_tok,
        is_leaf=(n_tok <= MAX_LEAF_TOKENS) or is_too_small_to_split,
        boundary_score=None,
        parent_id=parent_id,
        sibling_idx=sibling_idx,
        n_siblings=n_siblings,
    )

    if node.is_leaf:
        return [node]

    # Decide k (# children) and find k-1 valleys
    k = max(2, math.ceil(n_tok / MAX_SPLIT_CHUNK_TOKENS))
    sub_vecs = colbert_vecs[start_tok:end_tok]
    sims = adjacent_window_maxsim(sub_vecs)
    sub_valleys = find_valleys(sims, k - 1, max(1, MIN_SPLIT_GAP_TOKENS // WINDOW_STRIDE))
    sub_valley_toks = [min(n_tok, (v + 1) * WINDOW_STRIDE) for v in sub_valleys]
    sub_valley_toks = sorted([
        v for v in sub_valley_toks
        if v >= MIN_SPLIT_GAP_TOKENS and v <= n_tok - MIN_SPLIT_GAP_TOKENS
    ])
    if not sub_valley_toks:
        if n_tok // k >= MIN_SPLIT_GAP_TOKENS:
            sub_valley_toks = [(i + 1) * n_tok // k for i in range(k - 1)]
        else:
            node.is_leaf = True
            return [node]
    # Re-validate: ensure minimum spacing between adjacent valleys in token space
    spaced = [sub_valley_toks[0]]
    for v in sub_valley_toks[1:]:
        if v - spaced[-1] >= MIN_SPLIT_GAP_TOKENS:
            spaced.append(v)
    if not spaced:
        node.is_leaf = True
        return [node]
    sub_valley_toks = spaced

    # Build child token ranges
    prev_end_tok = start_tok
    child_ranges: list[tuple[int, int, int]] = []   # (abs_start_tok, abs_end_tok, boundary_score)
    for sub_v_tok in sub_valley_toks:
        abs_v = start_tok + sub_v_tok
        if abs_v <= prev_end_tok:
            abs_v = prev_end_tok + MIN_SPLIT_GAP_TOKENS
        if abs_v >= end_tok - MIN_SPLIT_GAP_TOKENS:
            continue
        bsims_i = sub_v_tok // WINDOW_STRIDE - 1
        bscore = float(1.0 - sims[bsims_i]) if 0 <= bsims_i < len(sims) else 0.0
        child_ranges.append((prev_end_tok, abs_v, bscore))
        prev_end_tok = abs_v
    if prev_end_tok < end_tok - MIN_SPLIT_GAP_TOKENS:
        child_ranges.append((prev_end_tok, end_tok, 0.0))
    elif prev_end_tok < end_tok:
        ps, pe, pb = child_ranges[-1]
        child_ranges[-1] = (ps, end_tok, pb)
    if not child_ranges:
        node.is_leaf = True
        return [node]
    if len(child_ranges) == 1:
        node.is_leaf = True
        return [node]

    # Convert token ranges to char ranges
    children: list[ChunkNode] = []
    for ci, (cs, ce, bs) in enumerate(child_ranges):
        c_start_char = split_token_to_char(cs, char_offsets)
        c_end_char = split_token_to_char(ce, char_offsets)
        if c_end_char <= c_start_char:
            c_end_char = c_start_char + max(1, ce - cs)
        children.append(ChunkNode(
            level=level + 1,
            start_char=c_start_char,
            end_char=c_end_char,
            n_tokens_colbert=ce - cs,
            is_leaf=False,
            boundary_score=bs,
            parent_id=None,
            sibling_idx=ci,
            n_siblings=len(child_ranges),
        ))

    all_nodes: list[ChunkNode] = [node]
    for c, (cs, ce, _bs) in zip(children, child_ranges):
        sub = recurse_chunk(
            start_tok=cs,
            end_tok=ce,
            colbert_vecs=colbert_vecs,
            char_offsets=char_offsets,
            start_char=c.start_char,
            end_char=c.end_char,
            level=level + 1,
            parent_id=None,
            sibling_idx=c.sibling_idx,
            n_siblings=c.n_siblings,
            doc_id=doc_id,
        )
        all_nodes.extend(sub)

    return all_nodes


# ----------------------------- semantic (sentence-level) chunker -------------

def _char_to_tok(char_pos: int, char_offsets: list[tuple[int, int]],
                 fallback: int) -> int:
    """Return the smallest token index whose start_char >= char_pos.

    Used to map a sentence's char range -> token range. `fallback` is
    returned if no token matches (e.g. char_pos is past the end of the
    tokenized text — happens when a sentence is longer than the colbert
    tokenization covered).
    """
    # Binary search would be faster; for ~thousands of tokens linear is fine
    for i, (s, _e) in enumerate(char_offsets):
        if s >= char_pos:
            return i
    return fallback


# Average characters per JinaBERT token. Used to estimate token counts for
# chunks that fall partly outside the ColBERT-covered token range (which is
# shorter than the JinaBERT range for long docs, since we truncate ColBERT
# to the first n_colbert tokens). 4.0 is a good middle ground for English
# enterprise text; the error bar is wide (±50%) but we only use this as
# a fallback when the exact count isn't available.
_CHARS_PER_TOKEN_ESTIMATE = 4.0


def _count_tokens_in_range(start_char: int, end_char: int,
                            colbert_vecs: np.ndarray,
                            char_offsets: list[tuple[int, int]]) -> int:
    """Count the ColBERT/JinaBERT tokens whose char range intersects
    [start_char, end_char).

    Two cases:
      1. Chunk is fully inside the ColBERT-covered region: use the offset
         map directly. Count = (first tok with start >= end_char) -
                              (first tok with start >= start_char).
      2. Chunk extends past the last ColBERT token: use the exact count
         for the part inside the ColBERT region, then ESTIMATE the tail
         from its char count (~CHARS_PER_TOKEN_ESTIMATE).
    """
    n_tok_total = colbert_vecs.shape[0]
    if n_tok_total == 0 or not char_offsets:
        # No tokens to count; estimate from char count
        return max(1, int((end_char - start_char) / _CHARS_PER_TOKEN_ESTIMATE))

    start_tok_idx = _char_to_tok(start_char, char_offsets, n_tok_total)
    end_tok_idx = _char_to_tok(end_char, char_offsets, n_tok_total)

    # The exact count: end_tok_idx - start_tok_idx
    if end_tok_idx > start_tok_idx:
        exact = end_tok_idx - start_tok_idx
    else:
        exact = 0

    # If the chunk extends past the last ColBERT token (end_tok_idx ==
    # n_tok_total fallback), add an estimate for the tail.
    if end_tok_idx >= n_tok_total and char_offsets:
        last_covered_char = char_offsets[-1][1]
        if end_char > last_covered_char:
            tail_chars = end_char - max(start_char, last_covered_char)
            tail_tok_est = max(1, int(tail_chars / _CHARS_PER_TOKEN_ESTIMATE))
            return exact + tail_tok_est

    if end_tok_idx == start_tok_idx and (end_char - start_char) > 0:
        # Whitespace gap: 1 token to be safe
        return 1
    return exact


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D or 2D vector. Returns zeros for zero-norm input."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-9, 1.0, n)
    return v / n


def semantic_chunk_sentences(
    text: str,
    sentences: List[Tuple[int, int]],
    colbert_vecs: np.ndarray,
    char_offsets: list[tuple[int, int]],
    target_chunk_tokens: int = TARGET_SEMANTIC_LEAF_TOKENS,
    min_chunk_tokens: int = MIN_SEMANTIC_LEAF_TOKENS,
    percentile_threshold: float = PERCENTILE_THRESHOLD,
    window_size: int = SENTENCE_WINDOW,
) -> List[Tuple[int, int, float]]:
    """Sentence-level semantic chunker (Kamradt / LlamaIndex / LangChain style).

    Args:
        text: the document text (for reference / debugging)
        sentences: list of (start_char, end_char) sentence ranges, in order
        colbert_vecs: (N_tokens, D) float32 ColBERT vectors (L2-normalized per token)
        char_offsets: list of (start_char, end_char) per token
        target_chunk_tokens: aim for chunks of this many tokens (default 256)
        min_chunk_tokens: never produce chunks smaller than this (default 128)
        percentile_threshold: breakpoints are distances above this percentile
                             of all consecutive-window distances (default 0.90)
        window_size: number of sentences per window (default 3)

    Returns:
        list of (start_char, end_char, boundary_score) tuples. The list covers
        the full text range with no gaps. `boundary_score` is the cosine
        distance (0..2) at the chunk boundary — higher = stronger break.

    Algorithm:
      1. Map each sentence -> token range via char_offsets
      2. Embed each sentence by mean-pooling its ColBERT tokens (L2-normalize)
      3. Sliding window of W sentences: distance[i] = 1 - cos_sim(
            mean(window[i:i+W]), mean(window[i+1:i+1+W]))
      4. Find breakpoints where distance >= percentile_threshold quantile
      5. Group consecutive sentences between breakpoints into chunks
      6. Merge chunks smaller than min_chunk_tokens with the following chunk
    """
    n_sent = len(sentences)
    if n_sent == 0:
        return [(0, len(text), 0.0)]
    if n_sent == 1:
        return [(sentences[0][0], sentences[0][1], 0.0)]

    n_tok = colbert_vecs.shape[0]
    if n_tok == 0 or len(char_offsets) == 0:
        return [(s, e, 0.0) for s, e in sentences]

    D = colbert_vecs.shape[1]
    fallback_tok = n_tok

    # 1. Map sentences to token ranges
    sent_tok_ranges: List[Tuple[int, int]] = []
    for s_char, e_char in sentences:
        ts = _char_to_tok(s_char, char_offsets, fallback_tok)
        te = _char_to_tok(e_char, char_offsets, fallback_tok)
        # Clip to actual token range
        ts = max(0, min(ts, n_tok))
        te = max(0, min(te, n_tok))
        if te < ts:
            te = ts
        sent_tok_ranges.append((ts, te))

    # 2. Embed each sentence by mean-pooling its ColBERT tokens
    sent_embs = np.zeros((n_sent, D), dtype=colbert_vecs.dtype)
    for i, (ts, te) in enumerate(sent_tok_ranges):
        if te > ts:
            v = colbert_vecs[ts:te].mean(axis=0)
            sent_embs[i] = v
    sent_embs = _l2_normalize(sent_embs)

    # 3. Sliding-window cosine distance
    # We compute one distance per adjacent window pair. If n_sent < window_size+1,
    # use the maximum possible window count and adapt.
    W = min(window_size, max(1, n_sent - 1))
    distances: List[float] = []
    for i in range(n_sent - W):
        a = sent_embs[i:i + W].mean(axis=0)
        b = sent_embs[i + 1:i + 1 + W].mean(axis=0)
        a = _l2_normalize(a.reshape(1, -1)).flatten()
        b = _l2_normalize(b.reshape(1, -1)).flatten()
        cos_sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
        distances.append(1.0 - cos_sim)  # cosine distance

    if not distances:
        return [(s, e, 0.0) for s, e in sentences]

    # 4. Find breakpoints: distances above the threshold percentile
    threshold = float(np.percentile(distances, percentile_threshold * 100))
    breakpoints = [i for i, d in enumerate(distances) if d >= threshold]
    if not breakpoints:
        # No semantic breaks — keep as one chunk
        return [(sentences[0][0], sentences[-1][1], 0.0)]

    # 5. Build chunks: consecutive sentences between breakpoints
    # The convention: distance[i] is between window[i:i+W] and window[i+1:i+1+W].
    # A breakpoint at index `bp` means the last sentence of the previous chunk
    # is sentence index `bp + W - 1` (the last sentence of the first window).
    chunks: List[Tuple[int, int, float]] = []
    prev_sent = 0
    for bp in breakpoints:
        last_sent = min(bp + W - 1, n_sent - 2)  # last sentence of the first window
        if last_sent < prev_sent:
            continue
        s_char = sentences[prev_sent][0]
        e_char = sentences[last_sent][1]
        chunks.append((s_char, e_char, float(distances[bp])))
        prev_sent = last_sent + 1
    # Trailing chunk
    if prev_sent < n_sent:
        s_char = sentences[prev_sent][0]
        e_char = sentences[-1][1]
        chunks.append((s_char, e_char, 0.0))

    # 6. Merge small chunks with the following chunk, iteratively
    #    (a merge can still be too small, so re-check)
    merged: List[Tuple[int, int, float]] = []
    i = 0
    while i < len(chunks):
        s, e, b = chunks[i]
        # Compute token count for this chunk to enforce min_chunk_tokens
        ts = _char_to_tok(s, char_offsets, fallback_tok)
        te = _char_to_tok(e, char_offsets, fallback_tok)
        n_tok_chunk = max(0, min(te, n_tok) - max(0, min(ts, n_tok)))
        # If this is the last chunk, accept it as-is
        if i == len(chunks) - 1:
            merged.append((s, e, b))
            break
        # If chunk is too small, merge with next and re-check
        if n_tok_chunk < min_chunk_tokens and i + 1 < len(chunks):
            ns, ne, nb = chunks[i + 1]
            # Replace chunk[i+1] with the merged version and re-process it
            chunks[i + 1] = (s, ne, max(b, nb))
            i += 1
            continue
        merged.append((s, e, b))
        i += 1
    return merged


def recurse_chunk_semantic(
    text: str,
    start_char: int,
    end_char: int,
    colbert_vecs: np.ndarray,
    char_offsets: list[tuple[int, int]],
    level: int,
    parent_id: str | None,
    sibling_idx: int,
    n_siblings: int,
    doc_id: str,
    target_chunk_tokens: int = TARGET_SEMANTIC_LEAF_TOKENS,
    min_chunk_tokens: int = MIN_SEMANTIC_LEAF_TOKENS,
    percentile_threshold: float = PERCENTILE_THRESHOLD,
    window_size: int = SENTENCE_WINDOW,
) -> list[ChunkNode]:
    """Sentence-level recursive chunker.

    Mirrors `recurse_chunk` in signature so the build script can swap one for
    the other behind a CLI flag. The difference: splits only happen at
    sentence boundaries, and boundary detection is a percentile-threshold on
    sliding-window cosine distance (mean-pooled ColBERT per sentence) rather
    than per-token MaxSim valleys.

    Args:
        text: the full document text (used to slice sentence ranges)
        start_char, end_char: char range of this chunk within `text`
        colbert_vecs, char_offsets: token-level ColBERT vectors and char offsets
        level, parent_id, sibling_idx, n_siblings, doc_id: same as recurse_chunk

    Returns:
        flat list of ChunkNode covering the subtree rooted at this chunk.
    """
    # Lazy import: avoid circular dep with sentence_segmenter.py at module-load
    from app.ml.sentence_segmenter import segment_sentences

    n_tok_total = colbert_vecs.shape[0]
    n_tok = n_tok_total
    # Count tokens whose char range falls inside [start_char, end_char).
    # Uses the offset map for the ColBERT-covered region and a chars/token
    # estimate for any tail that extends past the last ColBERT token (long
    # docs get truncated by `min(n_tokens, n_colbert)` in the build script,
    # so the JinaBERT offsets are also truncated to keep indices aligned).
    n_tok_in_range = _count_tokens_in_range(
        start_char, end_char, colbert_vecs, char_offsets,
    )
    n_tok = n_tok_in_range

    is_too_small_to_split = n_tok < 2 * min_chunk_tokens
    node = ChunkNode(
        level=level,
        start_char=start_char,
        end_char=end_char,
        n_tokens_colbert=n_tok,
        is_leaf=(n_tok <= target_chunk_tokens) or is_too_small_to_split,
        boundary_score=None,
        parent_id=parent_id,
        sibling_idx=sibling_idx,
        n_siblings=n_siblings,
    )

    if node.is_leaf:
        return [node]

    # Segment sentences within this range
    sub_text = text[start_char:end_char]
    sub_sents = segment_sentences(sub_text)
    # Convert to absolute char positions
    abs_sents = [(s + start_char, e + start_char) for s, e in sub_sents]
    if len(abs_sents) < 2:
        # Can't chunk by sentence in this range
        node.is_leaf = True
        return [node]

    # Compute chunks
    chunks = semantic_chunk_sentences(
        text=sub_text,
        sentences=sub_sents,  # relative — semantic_chunk_sentences uses these
        colbert_vecs=colbert_vecs,
        char_offsets=char_offsets,
        target_chunk_tokens=target_chunk_tokens,
        min_chunk_tokens=min_chunk_tokens,
        percentile_threshold=percentile_threshold,
        window_size=window_size,
    )
    if not chunks:
        node.is_leaf = True
        return [node]

    # Convert chunk ranges back to absolute
    abs_chunks = [(s + start_char, e + start_char, b) for s, e, b in chunks]

    # If the chunker returned a single chunk AND that chunk is too big,
    # fall back to character-wise splitting. This handles docs that have
    # no strong semantic breaks within them (rare, but happens for long
    # monologues or single-topic technical docs).
    if len(abs_chunks) == 1 and n_tok_in_range > 2 * target_chunk_tokens:
        n_pieces = max(2, n_tok_in_range // target_chunk_tokens)
        piece_chars = (end_char - start_char) // n_pieces
        abs_chunks = []
        for i in range(n_pieces):
            cs = start_char + i * piece_chars
            ce = start_char + (i + 1) * piece_chars if i < n_pieces - 1 else end_char
            abs_chunks.append((cs, ce, 0.0))

    if len(abs_chunks) == 1:
        node.is_leaf = True
        return [node]

    # Build children
    children: list[ChunkNode] = []
    for ci, (cs, ce, bscore) in enumerate(abs_chunks):
        if ce <= cs:
            continue
        children.append(ChunkNode(
            level=level + 1,
            start_char=cs,
            end_char=ce,
            n_tokens_colbert=0,  # filled in by recursion
            is_leaf=False,
            boundary_score=bscore,
            parent_id=None,
            sibling_idx=ci,
            n_siblings=len(abs_chunks),
        ))

    all_nodes: list[ChunkNode] = [node]
    for c in children:
        # Cap recursion depth: extremely long / finely-segmented docs can
        # recurse arbitrarily deep (e.g. a doc with thousands of 1-sentence
        # bullets). MAX_RECURSION_DEPTH=8 gives us up to 8 levels of hierarchy.
        if level + 1 > 8:
            c.is_leaf = True
            # Count this child's own tokens (not a parent-derived estimate).
            c.n_tokens_colbert = _count_tokens_in_range(
                c.start_char, c.end_char, colbert_vecs, char_offsets,
            )
            all_nodes.append(c)
            continue
        sub = recurse_chunk_semantic(
            text=text,
            start_char=c.start_char,
            end_char=c.end_char,
            colbert_vecs=colbert_vecs,
            char_offsets=char_offsets,
            level=level + 1,
            parent_id=None,
            sibling_idx=c.sibling_idx,
            n_siblings=c.n_siblings,
            doc_id=doc_id,
            target_chunk_tokens=target_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
            percentile_threshold=percentile_threshold,
            window_size=window_size,
        )
        all_nodes.extend(sub)
    return all_nodes
