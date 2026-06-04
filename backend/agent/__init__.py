"""Reactive RAG agent built on LangGraph.

The agent is composed of:
  1. BM25 sparse retrieval   (top-2000)
  2. jina-v3 dense retrieval (top-2000)
  3. RRF fusion               (top-100)
  4. Reactive LLM agent       (reads docs 10 at a time)

Stages 1-3 mirror the best production pipeline from the experiments
(83.4 % hit@100 on the 500-question dev set). Stage 4 is a ReAct-style
agent that calls a ``get_next_batch`` tool to paginate through the 100
docs in chunks of 10 and answers the question as soon as it sees
supporting evidence.
"""
from .state import AgentState
from .retrieval import bm25_retrieve, jina_dense_retrieve, make_rrf_fuse
from .tools import make_get_next_batch
from .graph import build_graph, run_agent

__all__ = ["AgentState", "bm25_retrieve", "jina_dense_retrieve", "make_rrf_fuse",
           "make_get_next_batch", "build_graph", "run_agent"]
