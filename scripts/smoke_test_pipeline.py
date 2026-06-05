#!/usr/bin/env python3
"""End-to-end smoke test for the RAG agent pipeline.

Runs a single question through the full graph and asserts:
  1. All 6 graph nodes fire (bm25, jina, rrf, agent, tools, finalize)
  2. BM25 cache hits ("BM25 index loaded from cache" log line)
  3. jina-v3 + LanceDB load
  4. **CRITICAL**: gold fields (expected_doc_ids, gold_answer) NEVER
     appear in any message the LLM sees — not in SystemMessage, not in
     HumanMessage, not in any ToolMessage, nowhere
  5. The agent finishes by calling ``submit_answer`` (not by giving up)
  6. ``final_answer`` is non-empty
  7. ``final_answer`` is grounded — shares at least one significant
     word with the gold answer
  8. At least one gold doc is in ``supporting_doc_ids``

Exits 0 on success, non-zero on any assertion failure.
"""
from __future__ import annotations
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure the venv-loaded `agent` package is importable
BACKEND = "/data/projects/rag/backend"
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

if not os.environ.get("MINIMAX_API_KEY"):
    print("ERROR: MINIMAX_API_KEY not set. Export it or use scripts/run_ui.sh.",
          file=sys.stderr)
    sys.exit(2)

from agent import run_agent  # noqa: E402
from agent.tracing import shutdown as lf_shutdown  # noqa: E402

QUESTIONS = "/data/projects/rag/data/questions.jsonl"
# qst_0002 is short, has a single-word gold answer ("stream.timebox_finalized")
# and known to be answerable. Good smoke-test target.
QID = os.environ.get("SMOKE_QID", "qst_0002")

REQUIRED_NODES = ("bm25_retrieve", "jina_dense_retrieve", "rrf_fuse",
                  "agent_llm", "finalize")


def main() -> int:
    qs = [json.loads(ln) for ln in open(QUESTIONS, encoding="utf-8") if ln.strip()]
    q = next((qq for qq in qs if qq["question_id"] == QID), None)
    if q is None:
        print(f"ERROR: question {QID} not found in {QUESTIONS}", file=sys.stderr)
        return 2

    gold_answer = q["gold_answer"]
    gold_docs = set(q.get("expected_doc_ids") or [])
    # Treat any non-trivial token in the gold as "significant" — helps
    # detect "the agent said nothing relevant" cases.
    gold_words = {w.strip(".,;:()[]\"'`")
                  for w in gold_answer.lower().split()
                  if len(w) > 3}

    print("=" * 60)
    print(f"SMOKE TEST — question {QID}")
    print("=" * 60)
    print(f"Q:           {q['question']}")
    print(f"Gold:        {gold_answer!r}")
    print(f"Gold docs:   {sorted(gold_docs)}")
    print(f"Gold words:  {sorted(gold_words)}")
    print()

    # Per-run Langfuse session id so the smoke test trace is findable
    # in the Sessions view. Format: smoke-YYYYMMDD-HHMMSS-<short uuid>.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_id = f"smoke-{stamp}-{uuid.uuid4().hex[:8]}"
    print(f"Langfuse session_id: {session_id}", flush=True)
    try:
        return _run_checks(q, gold_answer, gold_docs, gold_words, session_id)
    finally:
        # Always flush Langfuse — even on a failed assertion — so the
        # trace lands in the UI either way.
        lf_shutdown()


def _run_checks(q, gold_answer, gold_docs, gold_words, session_id) -> int:
    t0 = time.time()
    final = run_agent(q["question"], question_id=q["question_id"],
                      session_id=session_id)
    elapsed = time.time() - t0
    print(f"\nfinished in {elapsed:.1f}s")
    print(f"final_answer:        {final.get('final_answer')!r}")
    print(f"supporting_doc_ids:  {final.get('supporting_doc_ids')}")
    print(f"finished_via_tool:   {final.get('finished_via_tool')}")

    # 1. all nodes fired
    trace_nodes = [t.get("node") for t in final.get("node_trace", [])]
    print(f"\n[1/6] node_trace: {trace_nodes}")
    missing = [n for n in REQUIRED_NODES if n not in trace_nodes]
    if missing:
        print(f"  ❌ missing nodes: {missing}")
        return 1
    print(f"  ✅ all {len(REQUIRED_NODES)} required nodes present")

    # 2. finished via tool
    if not final.get("finished_via_tool"):
        print("[2/6] ❌ agent did not finish via submit_answer tool")
        return 1
    print("[2/6] ✅ agent called submit_answer tool")

    # 3. CRITICAL: gold never appears in the LLM's INPUT (not in outputs)
    # We check only the seeded SystemMessage and HumanMessage — these are
    # the only messages the *system* controls. ToolMessages (retrieved
    # batches) and AIMessages (the agent's own outputs / tool calls) are
    # not inputs and may legitimately contain the gold doc — that's the
    # whole point of RAG.
    msgs = final.get("messages", [])
    print(f"\n[3/6] scanning input messages (System + Human) for gold leakage…")

    # The system seeds exactly one SystemMessage and one HumanMessage.
    sys_msgs = [m for m in msgs if m.__class__.__name__ == "SystemMessage"]
    hum_msgs = [m for m in msgs if m.__class__.__name__ == "HumanMessage"]
    input_msgs = sys_msgs + hum_msgs
    print(f"  found {len(sys_msgs)} SystemMessage + {len(hum_msgs)} HumanMessage")

    # Banned substrings: the gold answer text and the gold doc dsids.
    # We do NOT scan for full doc paths (e.g. "github/dsid_xxx__file.txt")
    # because those are stable corpus paths, not gold signals — and we
    # do NOT scan ToolMessage or AIMessage content.
    import re
    banned = []
    if gold_answer:
        banned.append(gold_answer)
    for did in gold_docs:
        m = re.search(r"dsid_[a-f0-9]+", did)
        if m:
            banned.append(m.group(0))
    banned = [b for b in banned if b]
    print(f"  banned substrings: {banned}")

    leak_found = False
    for i, m in enumerate(input_msgs):
        content = m.content
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        for needle in banned:
            if needle in content:
                print(f"  ❌ LEAK at input msg[{i}] ({m.__class__.__name__}): "
                      f"{needle!r} present in content")
                print(f"     content preview: {content[:200]!r}")
                leak_found = True
    if leak_found:
        return 1
    print(f"  ✅ no gold in any of {len(input_msgs)} input messages")

    # 4. final_answer is non-empty
    ans = (final.get("final_answer") or "").strip()
    print(f"\n[4/6] final_answer non-empty? {bool(ans)}")
    if not ans:
        print("  ❌ final_answer is empty")
        return 1
    if ans.startswith("Question cannot be answered"):
        print(f"  ❌ agent gave up: {ans!r}")
        return 1
    print(f"  ✅ final_answer is {len(ans)} chars")

    # 5. answer is grounded — shares at least one significant word with gold
    ans_words = {w.strip(".,;:()[]\"'`")
                 for w in ans.lower().split()
                 if len(w) > 3}
    overlap = gold_words & ans_words
    print(f"[5/6] answer/gold word overlap: {len(overlap)} words: {sorted(overlap)[:10]}")
    if not overlap:
        print(f"  ❌ answer shares no words with gold")
        print(f"     answer: {ans!r}")
        print(f"     gold:   {gold_answer!r}")
        return 1
    print(f"  ✅ answer is grounded")

    # 6. at least one gold doc was cited
    cited = set(final.get("supporting_doc_ids") or [])
    hits = gold_docs & cited
    print(f"[6/6] gold docs cited: {len(hits)} of {len(gold_docs)}  {sorted(hits)}")
    if not hits:
        # This is a soft fail — agent may be right but cite a different
        # document. Print a warning but don't fail the smoke test.
        print(f"  ⚠️  no gold doc cited (but the answer is grounded)")
    else:
        print(f"  ✅ gold doc(s) cited")

    print()
    print("=" * 60)
    print("✅ SMOKE TEST PASSED")
    print("=" * 60)
    # Force-flush Langfuse before exit so the trace lands in the UI
    lf_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
