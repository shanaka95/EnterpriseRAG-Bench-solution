"""LLM wrapper for the MiniMax Anthropic-compatible endpoint.

Wraps ``langchain_anthropic.ChatAnthropic`` with our endpoint and model.
The class is what we'll bind tools to in the graph.
"""
from __future__ import annotations
from langchain_anthropic import ChatAnthropic

from .config import load_config


def get_llm(temperature: float | None = None) -> ChatAnthropic:
    """Return a ChatAnthropic pointed at the MiniMax endpoint."""
    cfg = load_config()
    return ChatAnthropic(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature if temperature is None else temperature,
    )
