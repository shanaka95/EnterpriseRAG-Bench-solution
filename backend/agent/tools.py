"""Tools the reactive agent uses.

Two tools:
  * ``get_next_batch``  — paginate through the refined docs, 10 at a time.
  * ``submit_answer``   — the canonical way to provide the final answer.

Why a dedicated tool for the final answer? Because the Anthropic API
``tool_use`` mechanism guarantees that the arguments the model passes to
a tool are valid JSON matching the declared ``input_schema``. So if the
agent's final answer comes through ``submit_answer(doc_id, response)``,
we *know* the result is a structured object — no regex parsing of free
text, no dealing with markdown fences, no fighting with thinking blocks.

The graph also wires ``tool_choice={"type": "tool", "name": "submit_answer"}``
on the final turn so the model is forced to use this tool (and produce
structured JSON) when it's done reading.
"""
from __future__ import annotations
import json
import os
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from .config import load_config


class SubmitAnswerArgs(BaseModel):
    """The strict schema for the final answer.

    The Anthropic API validates the model's tool arguments against this
    schema before sending them back. If the model calls ``submit_answer``
    with these fields filled in, the input is guaranteed valid JSON.
    """
    doc_id: str | None = Field(
        default=None,
        description=(
            "Comma-separated list of doc_id(s) that contained the answer. "
            "Use null if the answer was not found in the retrieved documents. "
            "The model may return a string, but it will be coerced to a list "
            "by the parser when there are multiple doc_ids."
        ),
    )
    response: str = Field(
        ...,
        description="The final answer to the user's question. Be concise.",
    )


def make_get_next_batch(state_holder: dict) -> Callable:
    """Return a configured tool closure bound to ``state_holder``.

    The state_holder is the *same dict object* the graph mutates, so
    advancing ``current_idx`` here is visible to the rest of the graph.
    """
    cfg = load_config()

    @tool
    def get_next_batch(batch_size: int = 10) -> str:
        """Fetch the next batch of documents (doc_id + content) from the refined list.

        Call this repeatedly until the answer is found or all documents are
        exhausted. Each call returns up to ``batch_size`` documents and
        advances the internal cursor.

        Args:
            batch_size: How many documents to return (default 10).

        Returns:
            JSON string with the batch, current index, and remaining count.
            When no docs are left returns ``{"done": true, "exhausted": true}``.
        """
        refined = state_holder.get("refined_doc_ids") or []
        idx = int(state_holder.get("current_idx") or 0)
        batch_size = max(1, min(50, int(batch_size)))

        if idx >= len(refined):
            return json.dumps({"done": True, "exhausted": True,
                               "remaining": 0,
                               "message": "All documents have been read."})

        end = min(idx + batch_size, len(refined))
        batch = []
        for did in refined[idx:end]:
            fp = os.path.join(cfg.corpus_dir, did)
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except FileNotFoundError:
                content = "(file not found on disk)"
            # cap to 6000 chars per doc — keeps tool output ~ 60k chars max
            if len(content) > 6000:
                content = content[:6000] + "…[truncated]"
            batch.append({"doc_id": did, "content": content})

        # advance cursor (in-place; the graph sees the same dict)
        state_holder["current_idx"] = end

        return json.dumps({
            "batch": batch,
            "next_idx": end,
            "remaining": max(0, len(refined) - end),
            "index_range": [idx + 1, end],          # 1-indexed for display
            "total_refined": len(refined),
        }, ensure_ascii=False)

    return get_next_batch


def make_submit_answer(state_holder: dict) -> BaseTool:
    """Return a StructuredTool that captures the final answer.

    When the agent calls this tool, the args are validated by Pydantic
    (``SubmitAnswerArgs``) — so we *know* the input is structured JSON
    with ``doc_id`` and ``response`` fields. The tool then writes the
    captured answer into ``state_holder`` and returns a sentinel string
    so the ToolNode completes successfully.
    """
    cfg = load_config()

    def _run(doc_id: str | None, response: str) -> str:
        # Safety net: if the model submitted a blank/empty response,
        # store the explicit "cannot be answered" message instead. The
        # UI will show this as the final answer (not "(no answer)").
        if not response or not response.strip():
            response = "Question cannot be answered with the available documents."

        # Normalize doc_id: the schema says str or None, but the model
        # sometimes returns a comma-separated string for multiple docs.
        docs: list[str] = []
        if isinstance(doc_id, str) and doc_id.strip():
            docs = [d.strip() for d in doc_id.split(",") if d.strip()]
        state_holder["final_answer"] = response
        state_holder["supporting_doc_ids"] = docs
        state_holder["finished_via_tool"] = True
        return json.dumps({
            "status": "answer_submitted",
            "doc_id": doc_id,
            "response_preview": response[:200],
        }, ensure_ascii=False)

    # Use StructuredTool so the args_schema is enforced.
    from langchain_core.tools import StructuredTool
    return StructuredTool.from_function(
        func=_run,
        name="submit_answer",
        description=(
            "Submit your final answer to the user's question. Use this tool "
            "ONLY when you have enough information to answer. The tool takes "
            "two arguments: (1) the doc_id (or comma-separated doc_ids) of "
            "the document(s) that contained the answer — pass null if no "
            "document contained the answer; (2) your final response text. "
            "This is the ONLY way to finalize your answer. Do NOT output "
            "JSON as plain text — always call this tool."
        ),
        args_schema=SubmitAnswerArgs,
    )
