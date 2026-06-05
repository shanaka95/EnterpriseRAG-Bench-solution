#!/usr/bin/env python3
"""Run the reactive RAG agent on the benchmark questions and write the
answers to a JSONL file.

Output format (one JSON object per line, strict):

    {"question_id": "qst_0001",
     "answer": "Your answer text...",
     "document_ids": ["dsid_9550250a59e74f1bbd5612480b2e7100", "dsid_..."]}

The ``document_ids`` list contains only the bare ``dsid_<32hex>`` portion
of the cited document paths — no source prefix, no ``__<slug>`` suffix,
no file extension. So a citation like
``github/dsid_ae068ee4aa9640159427cd941bef0238__pr-18421-...txt`` becomes
``dsid_ae068ee4aa9640159427cd941bef0238``.

Usage:
    # First 5 questions
    ./backend/venv/bin/python scripts/run_batch_answers.py 5

    # First 50 questions
    ./backend/venv/bin/python scripts/run_batch_answers.py 50

    # All 500 (no arg)
    ./backend/venv/bin/python scripts/run_batch_answers.py

    # Custom output path
    ./backend/venv/bin/python scripts/run_batch_answers.py 5 --out data/my_answers.jsonl

The first question takes ~4 min (BM25 build); subsequent questions
take only LLM time (5-15s each on the LiteLLM-proxied gpt-5.4 endpoint,
or 10-30s on the MiniMax-M3 endpoint). Results stream to disk after
every question, so an interrupted run loses at most the last record.

Credentials
-----------
By default the script reads ``MINIMAX_API_KEY``, ``MINIMAX_BASE_URL``,
``MINIMAX_MODEL`` and ``MINIMAX_PROTOCOL`` from ``/data/projects/rag/.env``
if it exists. Existing environment variables take precedence over the
``.env`` file (so CI / shell exports still win). Pass ``--no-dotenv``
to skip the auto-load.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure the venv-loaded `agent` package is importable
BACKEND = "/data/projects/rag/backend"
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# Project root — the .env file lives here. We resolve relative to this
# script's location so the script works no matter where it's invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

# Vars the agent needs at minimum. .env may also set MODEL / BASE_URL /
# PROTOCOL — we load all of them.
DOTENV_VARS = ("MINIMAX_API_KEY", "MINIMAX_BASE_URL", "MINIMAX_MODEL",
               "MINIMAX_PROTOCOL", "OPENAI_API_KEY")


def load_dotenv(path: Path = DOTENV_PATH) -> dict[str, str]:
    """Minimal ``.env`` parser — no python-dotenv dependency.

    Reads each ``KEY=VALUE`` line, strips quotes, ignores blanks and
    ``#`` comments. Returns the parsed dict (also written into
    ``os.environ`` only for keys NOT already set — existing env wins).
    """
    if not path.is_file():
        return {}
    parsed: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip surrounding quotes (single or double)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            parsed[key] = val
    # Existing env vars win — only fill in blanks
    for k, v in parsed.items():
        os.environ.setdefault(k, v)
    return parsed

from agent import run_agent  # noqa: E402

QUESTIONS_PATH = "/data/projects/rag/data/questions.jsonl"
DEFAULT_OUT = "/data/projects/rag/data/agent_answers.jsonl"

# Max attempts per question when the agent produces no answer at all
# (exception, empty string, or pure whitespace). The agent's own
# "Question cannot be answered..." fallback counts as a real answer and
# is NOT retried.
MAX_RETRIES = 3

# A bare doc id is `dsid_` followed by exactly 32 lowercase hex chars.
# We match inside the longer agent citation (`source/dsid_<hex>__slug.ext`)
# and pull out the bare id. No `\b` anchor — the 32-hex requirement is
# already strict enough, and `\b` fails when the next char is `_`
# (since `_` is a word char, so no boundary exists after the hex run).
DSID_RE = re.compile(r"(dsid_[0-9a-f]{32})")


def bare_doc_id(path: str) -> str | None:
    """Extract ``dsid_<32hex>`` from a full citation path.

    Returns the bare id on success, ``None`` if the citation doesn't
    contain a recognisable dsid (e.g. it's a directory listing or
    non-corpus reference). The agent is only ever expected to cite
    corpus docs, so a ``None`` is unexpected and we drop it.
    """
    m = DSID_RE.search(path)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(
        description="Run the agent on the first N questions and write JSONL.",
    )
    ap.add_argument(
        "num_questions", type=int, nargs="?", default=None,
        help="How many of the first questions to run. Default: all 500.",
    )
    ap.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Output JSONL path (default: {DEFAULT_OUT}).",
    )
    ap.add_argument(
        "--questions", default=QUESTIONS_PATH,
        help=f"Path to questions.jsonl (default: {QUESTIONS_PATH}).",
    )
    ap.add_argument(
        "--no-dotenv", action="store_true",
        help=f"Skip auto-loading {DOTENV_PATH} (rely on existing env vars).",
    )
    ap.add_argument(
        "--no-trace", action="store_true",
        help="Disable Langfuse tracing for this run (overrides env config).",
    )
    args = ap.parse_args()

    # Load .env BEFORE the env-var check (existing env vars take precedence)
    if not args.no_dotenv:
        loaded = load_dotenv()
        if loaded:
            keys = [k for k in loaded if k in DOTENV_VARS]
            print(f"[env] loaded {len(keys)} vars from {DOTENV_PATH}: {keys}",
                  flush=True)

    # Decide whether to enable Langfuse tracing. Default ON when env vars
    # are set; --no-trace turns it off explicitly.
    if not args.no_trace:
        try:
            from agent.tracing import get_callbacks
            cbs = get_callbacks()
            if cbs:
                print(f"[trace] Langfuse tracing enabled "
                      f"({len(cbs)} handler, host={os.environ.get('LANGFUSE_BASE_URL', '?')})",
                      flush=True)
            else:
                print(f"[trace] Langfuse not configured "
                      f"(set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY + LANGFUSE_BASE_URL)",
                      flush=True)
        except Exception as e:
            print(f"[trace] init failed: {e}", flush=True)
    trace = not args.no_trace

    # Now check the API key (after .env has populated it)
    if not os.environ.get("MINIMAX_API_KEY"):
        print("ERROR: MINIMAX_API_KEY not set. Put it in .env or export it.",
              file=sys.stderr)
        sys.exit(1)

    # Load all questions
    with open(args.questions, encoding="utf-8") as f:
        questions = [json.loads(ln) for ln in f if ln.strip()]
    if args.num_questions is not None:
        if args.num_questions < 0:
            print("ERROR: num_questions must be >= 0", file=sys.stderr)
            sys.exit(2)
        questions = questions[: args.num_questions]
    n_total = len(questions)
    if n_total == 0:
        print("ERROR: no questions to run", file=sys.stderr)
        sys.exit(2)
    print(f"[run] {n_total} questions selected  out={args.out}", flush=True)

    # Per-run session id for Langfuse grouping. Every question in this
    # batch will share the same session_id, so the Langfuse "Sessions"
    # view will show them as a single conversation replay. Format:
    #   benchmark-YYYYMMDD-HHMMSS-<short uuid>
    # The date+time makes the session findable by when it ran; the
    # short uuid disambiguates simultaneous runs.
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_id = f"benchmark-{run_stamp}-{uuid.uuid4().hex[:8]}"
    print(f"[run] session_id={session_id}  (use this to find the run "
          f"in Langfuse Sessions view)", flush=True)

    # Truncate output file (start fresh)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")  # truncate

    t_total = time.time()
    n_done = 0
    n_hits = 0
    n_retries = 0
    n_gave_up = 0  # agent returned an explicit "Question cannot be answered"
    n_empty = 0    # agent returned no answer (after all retries)
    for i, q in enumerate(questions, 1):
        qid = q["question_id"]
        qtext = q["question"]
        print(f"\n[{i:>3}/{n_total}] {qid}: {qtext[:80]}", flush=True)

        # Retry up to 3 attempts when the agent produces no answer at
        # all (exception, empty string, or pure whitespace). The agent's
        # own "Question cannot be answered..." fallback is treated as a
        # real answer and is NOT retried.
        answer = ""
        supporting: list[str] = []
        last_error: str | None = None
        attempts = 0
        for attempt in range(1, MAX_RETRIES + 1):
            attempts = attempt
            t0 = time.time()
            try:
                final = run_agent(
                    qtext, question_id=qid, trace=trace,
                    session_id=session_id,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                print(f"  [attempt {attempt}/{MAX_RETRIES}] EXCEPTION: "
                      f"{last_error}", flush=True)
                continue
            elapsed = time.time() - t0

            answer = (final.get("final_answer") or "").strip()
            supporting = final.get("supporting_doc_ids") or []

            if answer:
                # Real answer (or an explicit "cannot be answered" —
                # both are acceptable). Stop retrying.
                break
            else:
                last_error = "empty final_answer"
                print(f"  [attempt {attempt}/{MAX_RETRIES}] empty answer "
                      f"(elapsed={elapsed:.1f}s)", flush=True)

        if not answer:
            # All attempts failed with no usable answer. Record the most
            # recent final (which may be partially populated: cited docs,
            # agent trace, etc.) but mark the answer as the explicit
            # fallback string so the downstream consumer knows the agent
            # gave up. This is the same string the agent itself would
            # produce; keeping it consistent makes the JSONL easier to
            # post-process.
            n_empty += 1
            answer = "Question cannot be answered with the available documents."
            print(f"  ⚠️  giving up on {qid} after {attempts} attempts "
                  f"(last: {last_error})", flush=True)
        elif answer.startswith("Question cannot be answered"):
            n_gave_up += 1

        if attempts > 1:
            n_retries += attempts - 1

        # Strip the citation paths down to bare dsid_ ids, preserving
        # the order the agent listed them in and de-duplicating.
        doc_ids: list[str] = []
        seen: set[str] = set()
        for path in supporting:
            bare = bare_doc_id(path)
            if bare and bare not in seen:
                doc_ids.append(bare)
                seen.add(bare)

        # Hit evaluation: any bare id from the gold set appears in the
        # agent's cited ids. The gold ids are already bare (per the
        # questions.jsonl schema), so a direct set intersection works.
        expected = q.get("expected_doc_ids", []) or []
        hit = bool(expected) and bool(set(expected) & set(doc_ids))
        if hit:
            n_hits += 1
        n_done += 1

        # Append the single record to disk — stream-friendly, no in-memory
        # accumulation, so a 500-question run is safe against crashes.
        rec = {
            "question_id": qid,
            "answer": answer,
            "document_ids": doc_ids,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  answer: {answer[:120]!r}", flush=True)
        print(f"  doc_ids: {doc_ids}", flush=True)
        print(f"  attempts={attempts}  hit={'✅' if hit else '❌'}",
              flush=True)

    total = time.time() - t_total
    print(f"\n[done] wrote {n_done} records to {out_path}", flush=True)
    if n_done:
        print(f"[done] hits: {n_hits}/{n_done}  total wall clock: {total:.1f}s "
              f"({total/60:.1f} min)", flush=True)
        print(f"[done] explicit give-ups: {n_gave_up}  "
              f"retries (extra attempts): {n_retries}  "
              f"forced empty after retries: {n_empty}", flush=True)

    # Flush any pending Langfuse events before the script exits. Without
    # this, the last 1-2 traces of a batch run can be lost because the
    # Langfuse SDK buffers and ships asynchronously.
    if not args.no_trace:
        try:
            from agent.tracing import shutdown
            shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
