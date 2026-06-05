"""State schema for the LangGraph RAG agent.

The state is shared across all nodes. Fields are populated incrementally:
  - retrieval phase  : jv_ranked, bm_ranked, rrf_ranked, refined_doc_ids
  - agent phase      : messages, current_idx, final_answer, supporting_doc_ids
  - observability    : timings, node_trace

``messages`` uses ``Annotated[list, operator.add]`` so each node can return
*just the new message(s)* and LangGraph appends them — never return the
full conversation or you'll get duplicates and the API will reject the
second call with "tool result's tool id not found".
"""
from __future__ import annotations
from typing import Annotated, Any, TypedDict
import operator


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str
    question_id: str | None

    # --- retrieval phase ---
    jv_ranked: list[str]
    bm_ranked: list[str]
    rrf_ranked: list[str]
    rrf_scores: list[float]
    refined_doc_ids: list[str]            # top-100 from RRF

    # --- agent phase ---
    # Use operator.add so each node returns just the NEW message and
    # LangGraph appends. Don't return the full conversation.
    messages: Annotated[list, operator.add]
    current_idx: int                      # index into refined_doc_ids
    final_answer: str | None
    supporting_doc_ids: list[str]         # docs the agent cited
    finished: bool
    seeded: bool                          # True once [System, Human] are in

    # --- observability ---
    node_trace: Annotated[list[dict[str, Any]], operator.add]
    error: str | None
