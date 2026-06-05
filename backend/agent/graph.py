"""LangGraph state graph for the reactive RAG agent.

Graph layout (StateGraph):

    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ bm25     │ -> │ jina     │ -> │ rrf      │
    └──────────┘    └──────────┘    └──────────┘
                                          │
                                          v
                                    ┌──────────┐
                  ┌─ tool call? ─── │  agent   │ ◀─ system prompt
                  │                 └──────────┘
                  v
            ┌──────────┐
            │  tools   │  get_next_batch → 10 docs
            │  node    │  submit_answer  → captures final JSON
            └──────────┘
                  │
                  └─ back to agent

How the final answer is captured
--------------------------------
The agent has TWO tools:
  - ``get_next_batch(batch_size)``  — read more docs
  - ``submit_answer(doc_id, response)`` — provide the final answer

The Anthropic API guarantees that when the model calls a tool, the
arguments are valid JSON matching the tool's ``input_schema``. So if
the agent calls ``submit_answer`` with its two arguments, we *know*
the input is structured ``{doc_id, response}`` and we can read it
directly out of the ``tool_calls`` attribute of the AIMessage. No
regex, no JSON-in-prose parsing, no fighting with thinking blocks.

To be extra-strict: if the agent decides to stop without calling any
tool, the ``should_continue`` function re-prompts with
``tool_choice={"type": "tool", "name": "submit_answer"}`` — forcing
the model to use the structured tool.
"""
from __future__ import annotations
import json
import time
from typing import Any, Literal

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import AgentState
from .retrieval import bm25_retrieve, jina_dense_retrieve, make_rrf_fuse
from .tools import make_get_next_batch, make_submit_answer
from .llm import get_llm
from .config import load_config


# ---------- helpers ----------

def _content_to_text(content) -> str:
    """Extract plain text from an AIMessage content (str | list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    out.append(block["text"])
                elif block.get("type") == "thinking" and "thinking" in block:
                    out.append(f"[thinking] {block['thinking']}")
        return "\n".join(out)
    return str(content)


def _strip_thinking(ai_msg: AIMessage) -> AIMessage:
    """Return a copy of ``ai_msg`` with thinking blocks removed from content.

    The MiniMax Anthropic-compatible endpoint has trouble preserving
    tool_use/tool_result pairing when ``thinking`` blocks are interleaved
    with the actual tool_use. Stripping them keeps the protocol happy.
    """
    content = ai_msg.content
    if isinstance(content, list):
        filtered = [
            b for b in content
            if not (isinstance(b, dict) and b.get("type") == "thinking")
        ]
        if len(filtered) != len(content):
            ai_msg = ai_msg.model_copy(update={"content": filtered})
    return ai_msg


SYSTEM_PROMPT = """You are a precise, fully-grounded question-answering assistant. You will be \
given a user question and access to a corpus of documents via TWO tools:

  1. `get_next_batch(batch_size)` — fetch the next batch of documents \
(default 10) in relevance order from a previous retrieval stage \
(BM25 + jina-v3 + RRF fusion).
  2. `submit_answer(doc_id, response)` — submit your FINAL answer. \
This is the ONLY way to finish the task.

GROUNDING RULES — read these carefully:
- **Be fully grounded to the documents.** Every fact in your response must come from a \
document you actually read via `get_next_batch`. Do NOT use outside knowledge. Do NOT \
invent names, numbers, dates, or IDs.
- **Be direct.** Answer the question as asked, in plain language. Do not add caveats, \
follow-up offers, "let me know if...", or restate the question back.
- **Be complete.** Cover every part of the question using only what the documents say. \
If the question has multiple sub-parts, answer each one. Don't stop after the first fact.
- **Be concise.** Don't pad with summaries like "In summary, ..." or "Overall, ...". \
Just answer.
- **Cite the source.** The `doc_id` field must be the doc_id of the document(s) \
that actually contained the answer (comma-separated if multiple, or null if none).

CRITICAL — stop as soon as you have the answer. Do NOT keep reading more documents once \
you've found what you need. Every extra batch costs time and money. The first batch is \
usually enough.

Workflow:
1. Call `get_next_batch` to fetch the first batch of documents.
2. After each batch, look for a document that DIRECTLY and OBVIOUSLY contains the \
answer (specific values, names, dates, IDs, quotes that match the question).
3. As soon as you find it, call `submit_answer` with:
   - `doc_id`: the doc_id of the supporting document(s) (comma-separated if multiple, \
or null if not found)
   - `response`: your direct, complete, fully-grounded answer
4. Only call `get_next_batch` AGAIN if the current batch did NOT contain the answer. \
Don't second-guess — if you have any reasonable evidence (a number, a name, a date, a \
quote), use it.
5. If after reading ALL documents you cannot find the answer, call `submit_answer` \
with `doc_id=null` and `response="Question cannot be answered with the available documents."`

NEVER output plain text or JSON as your final answer. ALWAYS use the `submit_answer` tool.
"""


# ---------- agent node ----------

def make_agent_node(state_holder: dict):
    """Create the LLM-calling node. Captures state_holder in closure."""
    llm = get_llm()
    get_batch_tool: BaseTool = make_get_next_batch(state_holder)
    submit_tool: BaseTool = make_submit_answer(state_holder)
    # Both tools bound on the LLM. The model picks which to call.
    llm_with_tools = llm.bind_tools([get_batch_tool, submit_tool])

    def agent_node(state: AgentState) -> dict:
        t0 = time.time()
        messages = state.get("messages", []) or []
        new_messages: list = []

        # If first time, seed the conversation.
        if not state.get("seeded", False):
            new_messages += [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"Question: {state['question']}\n\n"
                    f"Call `get_next_batch` to read documents. When you have the answer, "
                    f"call `submit_answer(doc_id=..., response=...)` to finalize."
                )),
            ]
            messages = list(messages) + new_messages

        resp = llm_with_tools.invoke(messages)
        elapsed = time.time() - t0
        # Normalize: strip thinking, keep tool_calls intact.
        try:
            text_content = _content_to_text(resp.content)
            resp = _strip_thinking(resp)
        except Exception:
            text_content = ""

        # Capture any submit_answer call here so we don't need to wait for
        # the tool node to run. The tool node will still run (returning a
        # ToolMessage) but the answer is already in state_holder.
        for tc in (resp.tool_calls or []):
            if tc.get("name") == "submit_answer":
                args = tc.get("args", {}) or {}
                # Coerce doc_id to list, splitting on commas if needed.
                raw = args.get("doc_id")
                if raw is None or raw == "":
                    docs: list[str] = []
                elif isinstance(raw, str):
                    docs = [d.strip() for d in raw.split(",") if d.strip()]
                elif isinstance(raw, list):
                    docs = [str(d).strip() for d in raw if d]
                else:
                    docs = [str(raw).strip()]
                state_holder["final_answer"] = args.get("response", "")
                state_holder["supporting_doc_ids"] = docs
                state_holder["finished_via_tool"] = True
                break

        trace = [{
            "node": "agent_llm", "elapsed_s": round(elapsed, 3),
            "tool_calls": [tc.get("name") for tc in (resp.tool_calls or [])],
            "content_preview": (text_content or "")[:200],
        }]

        return {
            "messages": new_messages + [resp],
            "node_trace": trace,
            "seeded": True,
        }

    return agent_node


# ---------- decision & termination ----------

def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
    """If the last AIMessage has tool_calls, run tools; else finalize.

    Termination is triggered in three cases:
      1. The agent called ``submit_answer`` (already captured by agent_node).
      2. The agent called no tool at all AND we've already read at least one
         batch (avoid premature termination on the very first turn).
      3. The iteration cap was reached.
    """
    msgs = state.get("messages", []) or []
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and last.tool_calls:
        # If submit_answer was called, finalize (don't run the tool node,
        # which would invoke submit_answer again needlessly). The tool
        # result is already captured in state_holder by agent_node.
        if any(tc.get("name") == "submit_answer" for tc in last.tool_calls):
            return "finalize"
        n_agent_calls = sum(1 for m in msgs if isinstance(m, AIMessage))
        cfg = load_config()
        if n_agent_calls > cfg.max_iterations:
            return "finalize"
        return "tools"
    # No tool calls. If we've already read at least one batch, this is a
    # genuine "I want to stop" — but we should still let the user see the
    # final answer we already captured. If no tool calls on the very first
    # turn (model didn't read any docs), we also finalize.
    return "finalize"


def finalize(state: AgentState) -> dict:
    """Mark the run as finished.

    If the agent called ``submit_answer``, the answer and supporting
    docs are already in ``state_holder`` (set by agent_node). Otherwise,
    we fall back to parsing the last AIMessage's text.
    """
    sh_answer = state.get("_sh_final_answer")  # not actually used; read from passed-in dict
    # We can't easily access the agent_node's state_holder here, so the
    # caller (run_agent) merges the captured values back into final state.
    return {
        "finished": True,
        "node_trace": [{
            "node": "finalize",
        }],
    }


# ---------- graph builder ----------

def build_graph() -> tuple[Any, dict]:
    """Build the state graph and return (compiled_graph, state_holder).

    The state_holder is shared across the rrf_fuse node, the agent's
    submit_answer tool, and the get_next_batch tool, so all writes
    propagate consistently.
    """
    state_holder: dict = {}
    agent_node = make_agent_node(state_holder)
    rrf_node = make_rrf_fuse(state_holder)
    get_batch_tool = make_get_next_batch(state_holder)
    submit_tool = make_submit_answer(state_holder)
    tool_node = ToolNode([get_batch_tool, submit_tool])

    g = StateGraph(AgentState)
    g.add_node("bm25", bm25_retrieve)
    g.add_node("jina", jina_dense_retrieve)
    g.add_node("rrf",  rrf_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("finalize", finalize)

    g.set_entry_point("bm25")
    g.add_edge("bm25", "jina")
    g.add_edge("jina", "rrf")
    g.add_edge("rrf",  "agent")

    g.add_conditional_edges("agent", should_continue,
                            {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)

    return g.compile(), state_holder


# ---------- convenience runner ----------

def run_agent(question: str, question_id: str | None = None,
              expected_doc_ids: list[str] | None = None,
              gold_answer: str | None = None) -> dict:
    """Run the full agent on a single question. Returns a flat dict with
    the final state (and the messages for UI rendering)."""
    graph, state_holder = build_graph()
    init: AgentState = {
        "question": question,
        "question_id": question_id,
        "expected_doc_ids": expected_doc_ids or [],
        "gold_answer": gold_answer,
        "messages": [],
        "node_trace": [],
        "current_idx": 0,
        "finished": False,
        "seeded": False,
    }
    final = graph.invoke(init, {"recursion_limit": 200})
    # Merge state_holder values back. The agent_node writes
    # final_answer/supporting_doc_ids into state_holder directly when
    # the model calls submit_answer; the graph's view of the state
    # doesn't see those writes.
    final["current_idx"] = state_holder.get("current_idx", 0)
    if state_holder.get("finished_via_tool"):
        final["final_answer"] = state_holder.get("final_answer", "")
        final["supporting_doc_ids"] = state_holder.get("supporting_doc_ids", [])
        final["finished_via_tool"] = True
    return final
