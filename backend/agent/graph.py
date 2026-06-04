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
            │  tools   │ (next batch of 10 docs)
            └──────────┘
                  │
                  └─ back to agent

The agent decides when to call ``get_next_batch`` and when to give a
final answer (no more tool calls). We cap iterations at 25 batches
(250 docs) which is more than enough for the 100-doc refined list.
"""
from __future__ import annotations
import json
import time
from typing import Any, Literal

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import AgentState
from .retrieval import bm25_retrieve, jina_dense_retrieve, make_rrf_fuse
from .tools import make_get_next_batch
from .llm import get_llm
from .config import load_config


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


SYSTEM_PROMPT = """You are a precise question-answering assistant. You will be given a \
user question and access to a corpus of documents via a `get_next_batch` tool. The tool \
returns documents in batches (default 10 per call) in relevance order from a previous \
retrieval stage (BM25 + jina-v3 + RRF fusion).

Your job: read the documents in batches and answer the question.

CRITICAL — stop as soon as you have the answer. Do NOT keep reading more documents once \
you've found what you need. Every extra batch costs time and money. The first batch is \
usually enough.

Rules:
1. Call `get_next_batch` to fetch the first batch of documents.
2. After each batch, decide:
   - **If a document in the current batch directly and obviously contains the answer \
(values, names, dates, IDs, quotes that match the question), STOP immediately.** Output \
the JSON answer on the very next response. Do NOT call `get_next_batch` again.
   - Only call `get_next_batch` again if the current batch did NOT contain the answer.
3. If you have any reasonable evidence in the current batch — a number, a name, a date, a \
quote — that answers the question, USE IT. Don't second-guess and go looking for more.
4. Do NOT fabricate information. If after reading ALL documents you cannot find the \
answer, set doc_id to null and response to "I could not find the answer in the retrieved \
documents."

When you are ready to answer, output ONLY this JSON object and nothing else:

{
  "doc_id": "<the single doc_id that contained the answer, or comma-separated doc_ids if multiple, or null>",
  "response": "<your concise answer to the user's question>"
}

The JSON must be the LAST thing you output (no prose, no markdown fences). \
Call no more tools after the JSON.
"""


# ---------- agent node ----------

def make_agent_node(state_holder: dict):
    """Create the LLM-calling node. Captures state_holder in closure."""
    llm = get_llm()
    tool = make_get_next_batch(state_holder)
    llm_with_tools = llm.bind_tools([tool])

    def agent_node(state: AgentState) -> dict:
        t0 = time.time()
        messages = state.get("messages", []) or []

        new_messages: list = []

        # If first time, seed the conversation. We return ONLY the seed
        # messages + the new AIMessage; LangGraph appends to state.
        if not state.get("seeded", False):
            new_messages += [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"Question: {state['question']}\n\n"
                    f"Use the get_next_batch tool to read documents and answer. "
                    f"Begin with the first batch."
                )),
            ]
            messages = list(messages) + new_messages

        resp = llm_with_tools.invoke(messages)
        elapsed = time.time() - t0
        # Normalize the content for our downstream parser / UI
        try:
            text_content = _content_to_text(resp.content)
            # Strip thinking blocks — this endpoint has trouble pairing
            # tool_use with tool_result when thinking is present.
            resp = _strip_thinking(resp)
            if text_content and not resp.tool_calls:
                # Replace content with the extracted text so the parser works
                resp = resp.model_copy(update={"content": text_content})
        except Exception:
            text_content = ""
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


# ---------- parsing final answer ----------

def _parse_final(ai_msg: AIMessage) -> tuple[str | None, list[str], str | None]:
    """Extract (answer, supporting_doc_ids, reasoning) from the agent's final message.

    The agent is told to output strict JSON of the form::

        {"doc_id": "<id(s) or null>", "response": "<answer>"}

    We try JSON.parse first; if that fails we fall back to scanning the
    raw text for ``doc_id``-shaped strings and treat the whole content
    as the answer.
    """
    content = _content_to_text(ai_msg.content) if ai_msg.content else ""
    if not content:
        return None, [], None

    import json, re

    answer: str | None = None
    docs: list[str] = []
    reasoning: str | None = None

    # 1) Try strict JSON: look for the first '{' and try to parse the rest
    #    tolerantly (the model sometimes adds prose around the JSON).
    json_obj = None
    for opener in ("{", "[{"):
        idx = content.find(opener)
        if idx == -1:
            continue
        candidate = content[idx:].strip()
        # trim trailing junk after the closing brace
        depth = 0
        end_idx = -1
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        if end_idx > 0:
            candidate = candidate[:end_idx]
        try:
            json_obj = json.loads(candidate)
            break
        except Exception:
            continue

    if isinstance(json_obj, dict):
        resp = json_obj.get("response")
        if isinstance(resp, str):
            answer = resp.strip()
        doc_field = json_obj.get("doc_id")
        if isinstance(doc_field, str):
            # Comma-separated list of doc_ids is the common case
            docs = [d.strip() for d in doc_field.split(",") if d.strip()]
        elif isinstance(doc_field, list):
            docs = [str(d).strip() for d in doc_field if d]
        elif doc_field is None:
            docs = []
        # Optional 'reasoning' field (not required, just nice to have)
        reason = json_obj.get("reasoning")
        if isinstance(reason, str):
            reasoning = reason.strip()

    # 2) Fallback: scan the text for doc_id-shaped strings
    if not docs:
        ids = re.findall(r"dsid_[a-f0-9]{32}", content)
        docs = list(dict.fromkeys(ids))

    # 3) Last-resort: the whole content is the answer
    if answer is None:
        answer = content.strip()

    return answer, docs, reasoning


# ---------- decision & termination ----------

def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
    """If the last AIMessage has tool_calls, go to the tool node; else finalize."""
    msgs = state.get("messages", []) or []
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and last.tool_calls:
        # also enforce the iteration cap
        n_agent_calls = sum(1 for m in msgs if isinstance(m, AIMessage))
        cfg = load_config()
        if n_agent_calls > cfg.max_iterations:
            return "finalize"
        return "tools"
    return "finalize"


def finalize(state: AgentState) -> dict:
    """Mark the run as finished, parse out the final answer and supporting docs."""
    msgs = state.get("messages", []) or []
    last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
    answer, docs, reasoning = (None, [], None)
    if last_ai is not None:
        answer, docs, reasoning = _parse_final(last_ai)

    return {
        "final_answer": answer,
        "supporting_doc_ids": docs,
        "finished": True,
        "node_trace": [{
            "node": "finalize", "answer": answer, "n_docs": len(docs),
            "reasoning": reasoning,
        }],
    }


# ---------- graph builder ----------

def build_graph() -> tuple[Any, dict]:
    """Build the state graph and return (compiled_graph, state_holder).

    The state_holder is the same dict used inside the agent's tool closure
    AND the rrf_fuse node, so cursor advances and refined_doc_ids are
    visible to all of them.
    """
    state_holder: dict = {}
    agent_node = make_agent_node(state_holder)
    rrf_node = make_rrf_fuse(state_holder)
    tool = make_get_next_batch(state_holder)
    tool_node = ToolNode([tool])

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
    # The tool closure writes current_idx into state_holder; copy it back
    # since the graph's view of the state doesn't see the closure's writes.
    final["current_idx"] = state_holder.get("current_idx", 0)
    return final
