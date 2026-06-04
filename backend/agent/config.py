"""Configuration for the RAG agent.

Reads from environment variables (with sensible defaults). The API key is
sensitive — never log it. The model name and base URL are the
Anthropic-compatible endpoint for the MiniMax serving stack.
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    # LLM endpoint
    api_key: str
    base_url: str
    model: str

    # Retrieval hyperparameters (best from experiments)
    jv_top_n: int = 2000
    bm_top_n: int = 2000
    rrf_k0: int = 60
    rrf_top_k: int = 100

    # Agent loop
    batch_size: int = 10
    max_iterations: int = 25               # 25 batches × 10 = 250 lookups; well past 100
    max_tokens: int = 1024
    temperature: float = 0.0

    # Paths
    dense_index_dir: str = "/data/projects/rag/data/dense_index/db"
    corpus_dir: str = "/data/projects/rag/data/all_documents"
    questions_path: str = "/data/projects/rag/data/questions.jsonl"

    # jina-v3 model
    jina_model_id: str = "jinaai/jina-embeddings-v3"


def load_config() -> AgentConfig:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MINIMAX_API_KEY is not set. Put it in a .env file (see "
            ".env.example) or export it in your shell before running."
        )
    return AgentConfig(
        api_key=api_key,
        base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"),
        model=os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7"),
    )
