"""LLM factory: returns a LangChain chat model for the chosen protocol.

The agent's graph and tools are protocol-agnostic — they only call
``llm.invoke(messages)`` and ``llm.bind_tools(...)``. The factory
selects between ``ChatAnthropic`` and ``ChatOpenAI`` based on the
``protocol`` argument. Both classes implement that interface.

Default protocol: ``"openai"`` (the new MiniMax endpoint at
``http://167.233.22.91:19950/`` serves OpenAI-format models like
``gpt-4o-mini`` and ``deepseek-v4-pro``). Override with
``MINIMAX_PROTOCOL=anthropic`` in the env, or pass ``protocol=`` to
:func:`get_llm` directly (used by the Streamlit UI).

Resolution order for each argument: explicit kwarg > env var
(``MINIMAX_API_KEY`` / ``MINIMAX_BASE_URL`` / ``MINIMAX_MODEL``).
"""
from __future__ import annotations
import os
from typing import Literal

from .config import load_config

Protocol = Literal["openai", "anthropic"]


def get_llm(
    protocol: Protocol | str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
):
    """Build a LangChain chat model. Resolves None args from env."""
    cfg = load_config()  # env vars are required for api_key/base_url/model

    # Explicit kwargs take precedence; fall back to env.
    api_key = api_key or cfg.api_key
    base_url = base_url or cfg.base_url
    model = model or cfg.model
    temperature = cfg.temperature if temperature is None else temperature

    # Protocol: explicit arg > MINIMAX_PROTOCOL env > default "openai"
    if protocol is None:
        protocol = os.environ.get("MINIMAX_PROTOCOL", "openai").lower()
    if protocol not in ("openai", "anthropic"):
        raise ValueError(
            f"Unknown protocol: {protocol!r}. Use 'openai' or 'anthropic'."
        )

    if protocol == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=cfg.max_tokens,
            temperature=temperature,
        )

    # protocol == "openai"
    from langchain_openai import ChatOpenAI
    # The OpenAI SDK appends "/v1/chat/completions" to base_url. If the
    # caller already included "/v1" (common with OpenAI-style endpoints
    # like the new MiniMax one), strip it to avoid "/v1/v1/...".
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base or None,
        max_tokens=cfg.max_tokens,
        temperature=temperature,
    )


def llm_info(llm) -> tuple[str, str]:
    """Return ``(protocol, model_name)`` for a LangChain chat model.

    Used by the tracing layer to tag Langfuse traces with the actual
    model + protocol that was used, even if the caller passed a custom
    LLM (e.g. from the Streamlit UI). The protocol is detected from
    the class — both ``ChatAnthropic`` and ``ChatOpenAI`` carry a
    ``model_name`` / ``model`` attribute. Falls back to
    ``("unknown", "<unknown>")`` if we can't tell.
    """
    cls = type(llm)
    qualname = f"{cls.__module__}.{cls.__qualname__}"
    if "langchain_anthropic" in qualname:
        protocol = "anthropic"
    elif "langchain_openai" in qualname:
        protocol = "openai"
    else:
        protocol = "unknown"
    # ChatAnthropic stores it as .model; ChatOpenAI as .model_name.
    name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "<unknown>"
    return protocol, str(name)
