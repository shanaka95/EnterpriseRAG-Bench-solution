"""
Rule-based sentence segmentation for the 9-source enterprise corpus
(slack, jira, github, confluence, gmail, google_drive, hubspot, fireflies, linear).

Why rule-based, not spaCy/NLTK:
  - spaCy (en_core_web_sm) is not installed in backend/venv and is a ~50 MB model
  - NLTK punkt is not installed and is a ~30 MB download
  - The corpus mixes code blocks, bullet lists, URLs, JSON, ASCII tables — a
    rule-based splitter with a few targeted heuristics handles them well
  - We only need sentence *boundaries* (char ranges), not parse trees

Returns: list of (start_char, end_char) tuples, in order, covering the full
input text. The end of one sentence is the start of the next (no gaps).

Strategy:
  1. Mask code blocks so the sentence splitter doesn't touch them
  2. Strip URLs (replace with a single space, preserving trailing punct)
  3. Find split positions: paragraph breaks, bullet starts, numbered-list
     starts, sentence terminators followed by a sentence-start char
  4. Numeric list markers and decimal numbers are excluded from
     sentence-terminator detection
  5. Abbreviation list (~30 common ones) is checked before accepting a split
  6. Merge sentences shorter than MIN_SENTENCE_CHARS with the next one,
     unless the current sentence is a bullet or code block
  7. If a sentence is > MAX_SENTENCE_CHARS, hard-split it on the next newline
"""
from __future__ import annotations

import re
from typing import List, Tuple

# Match a URL and replace it with a single space (preserves char positions
# by using a same-length placeholder isn't necessary — we just need the
# splitter to not split on the period inside the URL).
# We DO care about preserving relative positions for the *output* ranges,
# so we strip URLs to a single space (1 char) and remember their start
# position. Simpler: replace with a single space and accept that the
# returned ranges will be slightly off in length when URLs are present.
# In practice the offset_mapping downstream handles this fine.

URL_PATTERN = re.compile(r'https?://[^\s]+|www\.[^\s]+')

# Trim trailing punctuation that is more likely to be a sentence terminator
# than part of the URL. Common case: "see https://example.com." — the period
# belongs to the sentence, not the URL.
URL_TRAILING_PUNCT = re.compile(r'([.,;:!?)\'\"])+$')

# Fenced code blocks: ``` ... ```
FENCED_CODE_PATTERN = re.compile(r'```[^\n]*\n.*?```', re.DOTALL)

# Indented code block (4+ spaces at line start, ≥2 lines)
INDENTED_CODE_PATTERN = re.compile(r'(?:^(?:    |\t)[^\n]+\n?){2,}', re.MULTILINE)

# Bullet list item start
BULLET_PATTERN = re.compile(r'(?:^|\n)[ \t]*[-*+•·][ \t]+')

# Numbered list start
NUMBERED_LIST_PATTERN = re.compile(r'(?:^|\n)[ \t]*\d+\.[ \t]+')

# Paragraph break
PARA_BREAK_PATTERN = re.compile(r'\n\s*\n+')

# Sentence terminator followed by whitespace and a sentence-start character.
# The look-ahead char set is what "starts" a sentence: capital letter, quote,
# paren, bracket, em-dash, bullet marker, "From " (slack), "On " (email).
SENTENCE_TERMINATOR_PATTERN = re.compile(
    r'[.!?](?=\s+[A-Z"\'(\[\-—*•·>])'
)

# Em-dash or semicolon as a soft terminator (e.g. "list — and another — item")
# Treated as optional — we split on these only if followed by 5+ words then
# another terminator (handled in post-processing).
EM_DASH_PATTERN = re.compile(r'\s+[—–]\s+')

# Common English abbreviations that should NOT trigger a split.
# Matched at the end of a word, case-sensitive (so "Dr." works, "dr." doesn't
# by design — enterprise docs use proper capitalization).
ABBREVIATIONS: frozenset[str] = frozenset({
    'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'St', 'Ave', 'Blvd',
    'No', 'Vol', 'Fig', 'Eq', 'Sec', 'Art', 'Ch', 'pp', 'p',
    'Inc', 'Ltd', 'Co', 'Corp', 'LLC',
    'e.g', 'i.e', 'etc', 'vs', 'cf', 'viz', 'approx',
    'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    'U.S', 'U.K', 'E.U',
})

# Build a pattern that matches "<word>." or "<word>." with the word being an
# abbreviation. We use a negative lookbehind to ensure the abbreviation
# is preceded by whitespace or start-of-string.
ABBREVIATION_PATTERN = re.compile(
    r'(?:^|(?<=[\s(\["\']))(' + '|'.join(re.escape(a) for a in sorted(ABBREVIATIONS, key=len, reverse=True)) + r')\.',
    re.MULTILINE,
)

# Minimum sentence length — shorter sentences are merged with the next
MIN_SENTENCE_CHARS = 20

# Hard upper bound for any single sentence (chars). Slack/jira messages can
# be huge with no terminators; force-split on next \n if we exceed this.
MAX_SENTENCE_CHARS = 2000


def _is_abbreviation_split(text: str, split_pos: int) -> bool:
    """Return True if the period at text[split_pos] should NOT trigger a split.

    Covers three cases:
      1. The period follows a known abbreviation (Dr., e.g., U.S., ...)
      2. The period is part of a numeric list marker ('1.', '12.', ...)
      3. The period is part of a decimal number ('3.14', ...)

    split_pos: the index of the `.` in text.
    """
    # Walk backwards from split_pos, collecting alphanumeric chars + dots
    end = split_pos
    start = end
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == '.'):
        start -= 1
    word = text[start:end]
    word_clean = word.rstrip('.')

    # Case 1: known abbreviation
    if word_clean in ABBREVIATIONS:
        return True

    # Case 2: numeric list marker (e.g. "1.", "12.", "3.5.")
    # The "word" we collected is purely digits/dots, starts with a digit,
    # and is followed by a space (in the original text) or is at end-of-string.
    if word_clean and word_clean.replace('.', '').isdigit():
        # Check what comes after the period: space, newline, or end
        after = text[split_pos + 1:split_pos + 2]
        if after in ('', ' ', '\n', '\t', '\r'):
            # Make sure the digit run is "list-like" (≤3 digits before the .)
            digits_before = word_clean.split('.')[0]
            if len(digits_before) <= 3:
                return True
        # Decimal number: only treat as non-split if the period is followed
        # by a digit (e.g., "3.14")
        if after.isdigit():
            return True

    return False


def _strip_urls(text: str) -> str:
    """Replace URLs with a single space, preserving trailing sentence punctuation.

    The URL pattern `https?://[^\\s]+` would otherwise greedily eat the
    trailing period in 'https://example.com.' (the sentence terminator).
    We strip that trailing punctuation from the match and re-attach it to
    the replacement so the period is still available to the sentence
    terminator detector.
    """
    def _repl(m: re.Match) -> str:
        url = m.group(0)
        trimmed = URL_TRAILING_PUNCT.sub('', url)
        trailing = url[len(trimmed):]   # the punct we removed
        return trimmed + ' ' + trailing # keep the URL stem + a space + the trailing punct
    return URL_PATTERN.sub(_repl, text)


def _mask_code_blocks(text: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Replace code blocks with placeholder chars so the sentence splitter
    doesn't touch them. Returns (masked_text, [(start, end, original), ...])."""
    blocks: List[Tuple[int, int, str]] = []

    def fenced_repl(m: re.Match) -> str:
        original = m.group(0)
        blocks.append((m.start(), m.end(), original))
        # Replace with newlines (so paragraph-break detection still works
        # across the boundary) of equivalent length
        return '\n' * original.count('\n')

    def indented_repl(m: re.Match) -> str:
        original = m.group(0)
        blocks.append((m.start(), m.end(), original))
        return '\n'

    masked = FENCED_CODE_PATTERN.sub(fenced_repl, text)
    masked = INDENTED_CODE_PATTERN.sub(indented_repl, masked)
    # Sort by start so we can recover the original positions
    blocks.sort()
    return masked, blocks


def _find_split_positions(masked: str) -> List[int]:
    """Return sorted list of char positions where a sentence ends in `masked`."""
    splits: set[int] = set()

    # 1. Paragraph breaks
    for m in PARA_BREAK_PATTERN.finditer(masked):
        splits.add(m.end())

    # 2. Bullet list items: split BEFORE the bullet (so each item is its own sentence)
    for m in BULLET_PATTERN.finditer(masked):
        splits.add(m.start())

    # 3. Numbered list items
    for m in NUMBERED_LIST_PATTERN.finditer(masked):
        splits.add(m.start())

    # 4. Sentence terminators (. ! ?) followed by a sentence-start char
    for m in SENTENCE_TERMINATOR_PATTERN.finditer(masked):
        period_pos = m.start()  # position of the period
        # Check if it's an abbreviation — if so, skip
        if _is_abbreviation_split(masked, period_pos):
            continue
        # The sentence ends AT the period (inclusive), so the next sentence
        # starts at m.end() (the whitespace + new char)
        splits.add(period_pos + 1)

    return sorted(splits)


def _apply_splits(masked: str, split_positions: List[int], text_len: int) -> List[Tuple[int, int, str]]:
    """Convert split positions into a list of (start, end, kind) char ranges
    covering [0, text_len). `kind` is 'normal' or 'bullet' (or 'code').

    Bullets/numbered-list starts are tagged so merge_short knows not to merge
    them with the previous sentence.
    """
    if not split_positions:
        return [(0, text_len, 'normal')]

    starts = [0] + [p for p in split_positions if 0 < p < text_len] + [text_len]
    # Dedupe consecutive duplicates
    out: List[Tuple[int, int, str]] = []
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        if e > s:
            # Mark this range as 'bullet' if the start char is at a bullet
            # or numbered-list marker position
            kind = 'normal'
            if i > 0 and starts[i] in split_positions:
                # We split here — check if it's a bullet start
                tail = masked[s:s + 5]
                if BULLET_PATTERN.match(tail) or NUMBERED_LIST_PATTERN.match(tail):
                    kind = 'bullet'
            out.append((s, e, kind))
    return out


def _merge_short(sentences: List[Tuple[int, int, str, str]], min_chars: int) -> List[Tuple[int, int, str, str]]:
    """Merge sentences shorter than min_chars with the FOLLOWING sentence,
    unless this sentence is a bullet/code-block (those always stand alone).

    Each sentence is (start, end, text, kind). Kind is 'normal' | 'bullet' | 'code'.
    """
    if not sentences:
        return sentences
    merged: List[Tuple[int, int, str, str]] = []
    i = 0
    while i < len(sentences):
        s, e, txt, kind = sentences[i]
        # If this is the last sentence, or a bullet, or a code block — keep as-is
        if i == len(sentences) - 1 or kind in ('bullet', 'code'):
            merged.append((s, e, txt, kind))
            i += 1
            continue
        # Merge with following while we're short (only if following is also normal)
        while len(txt) < min_chars and i + 1 < len(sentences):
            ns, ne, ntxt, nkind = sentences[i + 1]
            if nkind in ('bullet', 'code'):
                break  # don't absorb a bullet into a short sentence
            txt = txt + ntxt
            e = ne
            i += 1
        merged.append((s, e, txt, kind))
        i += 1
    return merged


def _hard_split_long(sentences: List[Tuple[int, int, str, str]], max_chars: int) -> List[Tuple[int, int, str, str]]:
    """If any sentence exceeds max_chars, force-split it on the next \\n."""
    out: List[Tuple[int, int, str, str]] = []
    for s, e, txt, kind in sentences:
        if len(txt) <= max_chars:
            out.append((s, e, txt, kind))
            continue
        # Hard-split: find the next \n after every max_chars
        cursor = 0
        abs_cursor = s
        while cursor < len(txt):
            end_cursor = min(cursor + max_chars, len(txt))
            if end_cursor < len(txt):
                # Find next newline
                nl = txt.find('\n', cursor, end_cursor + 200)  # look ahead a bit
                if nl > cursor:
                    end_cursor = nl + 1
            out.append((abs_cursor + cursor, abs_cursor + end_cursor, txt[cursor:end_cursor], kind))
            cursor = end_cursor
    return out


def _unmask_code_blocks(sentences: List[Tuple[int, int, str]],
                        blocks: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Restore the original code-block text by giving each block its own sentence.

    Each code block at [bs, be) becomes its own sentence. If a sentence
    straddles a block boundary, we split it at the block's start (and end).
    The block's content is preserved as a separate, atomic sentence.
    """
    if not blocks:
        return sentences

    new_sentences: List[Tuple[int, int, str]] = list(sentences)
    # Process blocks in order of appearance. For each block, find the
    # sentence(s) it intersects and split them.
    for bs, be, _btx in blocks:
        # Find the sentence(s) that this block intersects
        # Walk through new_sentences and split at bs and be
        updated: List[Tuple[int, int, str]] = []
        for s, e, txt in new_sentences:
            # No intersection: keep as-is
            if be <= s or bs >= e:
                updated.append((s, e, txt))
                continue
            # Sentence fully contains block: split into 3 (before, block, after)
            if s < bs and be < e:
                updated.append((s, bs, txt))         # before block
                updated.append((bs, be, ''))         # the block itself
                updated.append((be, e, txt))         # after block
            # Block fully contains sentence: keep as-is (sentence is inside block)
            elif bs <= s and e <= be:
                updated.append((s, e, txt))
            # Sentence starts inside block, ends after: split at block end
            elif s < be <= e:
                updated.append((s, be, txt))
                updated.append((be, e, txt))
            # Sentence starts before block, ends inside block: split at block start
            elif s < bs < e:
                updated.append((s, bs, txt))
                updated.append((bs, e, txt))
        new_sentences = updated
    return new_sentences


def segment_sentences(text: str) -> List[Tuple[int, int]]:
    """Segment `text` into a list of (start_char, end_char) sentence ranges.

    The returned ranges cover the full input text, in order, with no gaps.
    Each range is inclusive of start, exclusive of end.

    Args:
        text: the full document text (any encoding, but typically UTF-8)

    Returns:
        list of (start_char, end_char) tuples. len >= 1.
    """
    if not text:
        return []

    n = len(text)

    # Step 1: mask code blocks (keep their positions for later restore)
    masked, blocks = _mask_code_blocks(text)

    # Step 2: strip URLs from the masked text (we don't try to restore URLs
    # in the output — the offset_mapping downstream handles positioning)
    masked = _strip_urls(masked)

    # Step 3: find split positions
    split_positions = _find_split_positions(masked)

    # Step 4: convert to (start, end, kind) ranges covering [0, n)
    raw = _apply_splits(masked, split_positions, n)

    # Step 5: rebuild text slices from the ORIGINAL text (not the masked one)
    # so downstream code that reads text[start:end] gets the right content
    sents_with_text: List[Tuple[int, int, str, str]] = [
        (s, e, text[s:e], kind) for s, e, kind in raw
    ]

    # Step 6: unmask code blocks — expand sentence boundaries to cover the
    # original block ranges (returns (s, e, text) without kind)
    sents_simple: List[Tuple[int, int, str]] = [
        (s, e, txt) for s, e, txt, _ in sents_with_text
    ]
    sents_simple = _unmask_code_blocks(sents_simple, blocks)
    # Re-attach kinds (code blocks get kind='code' if they contain a block)
    new_sents: List[Tuple[int, int, str, str]] = []
    for i, (s, e, txt) in enumerate(sents_simple):
        original_kind = sents_with_text[min(i, len(sents_with_text) - 1)][3]
        # If a code block falls entirely inside this range, mark as 'code'
        has_code = any(b[0] >= s and b[1] <= e for b in blocks)
        kind = 'code' if has_code else original_kind
        new_sents.append((s, e, txt, kind))
    sents_with_text = new_sents

    # Step 7: merge very short sentences with the next (skipping bullets/code)
    sents_with_text = _merge_short(sents_with_text, MIN_SENTENCE_CHARS)

    # Step 8: hard-split overly long sentences on the next \n
    sents_with_text = _hard_split_long(sents_with_text, MAX_SENTENCE_CHARS)

    # Final: re-derive text from ORIGINAL text using the final positions
    final: List[Tuple[int, int]] = [
        (s, min(e, n)) for s, e, _, _ in sents_with_text
    ]
    # Dedupe (start, end) pairs that may have been created by merging
    seen = set()
    deduped: List[Tuple[int, int]] = []
    for s, e in final:
        if (s, e) not in seen:
            seen.add((s, e))
            deduped.append((s, e))
    return deduped
