"""Optional Langfuse tracing for the reactive RAG agent.

Auto-enabled when the three env vars below are set (the Langfuse SDK
itself keys off them). When any are missing, ``get_callbacks()``
returns ``[]`` and ``build_trace_config()`` returns just
``{"recursion_limit": ...}`` — the agent runs as before, with no
overhead and no noise.

    LANGFUSE_SECRET_KEY   (sk-lf-...)
    LANGFUSE_PUBLIC_KEY   (pk-lf-...)
    LANGFUSE_BASE_URL     (https://cloud.langfuse.com or self-hosted)

Usage from the agent runner::

    from agent.tracing import build_trace_config, flush

    invoke_config = build_trace_config(
        question_id="qst_0001",
        session_id="benchmark-2026-06-05-1",  # groups 10 traces per batch
        model="MiniMax-M3",
    )
    final = graph.invoke(init, invoke_config)
    flush()  # force a final sync flush before the script exits

The function returns a plain ``RunnableConfig`` dict that can be
passed to ``graph.invoke()``. LangGraph propagates the ``callbacks``,
``tags`` and ``metadata`` keys down to every nested ``invoke()``
(LLM, ToolNode, etc.), so the Langfuse handler attached at the graph
level captures the entire run as a single trace.

How Langfuse extracts trace-level attributes
--------------------------------------------
The Langfuse ``CallbackHandler`` (v3) reads these special keys out of
the **top-level** run's metadata and forwards them to
``propagate_attributes()`` internally — so they appear on every
nested observation automatically:

    metadata["langfuse_session_id"]  ->  session_id
    metadata["langfuse_user_id"]     ->  user_id
    metadata["langfuse_trace_name"]  ->  trace name (visible in UI)
    metadata["langfuse_tags"]        ->  list of tags
    metadata (other keys)            ->  observation metadata

That's why we put these "langfuse_*" keys in metadata, not at the
top level: the handler looks them up by name. We also set top-level
``tags`` so the LangChain run itself is discoverable in the LangSmith
debug view, but those are NOT the same as Langfuse tags — they don't
get forwarded to the Langfuse UI. The handler is the only one that
reads ``langfuse_tags``.

Source: langfuse/langchain/CallbackHandler.py:_parse_langfuse_trace_attributes

Cost
----
The Langfuse SDK batches and flushes events asynchronously. ``flush()``
forces a final sync flush — call it once at the end of a batch run or
at the end of a single UI invocation to avoid losing the tail.
"""
from __future__ import annotations
import os
from typing import Any

# Don't import langfuse at module load time — it's an optional dep and
# the user might not have it installed. The functions below handle
# ImportError gracefully and just return [] / no-op.

_HANDLER: Any = None  # cached langchain CallbackHandler
_CLIENT: Any = None   # cached raw Langfuse client (for flush / update)


# Standard tags we want on every trace regardless of run. Keep the
# list short — Langfuse charges nothing per tag but noisy tags
# clutter the filter UI.
_BASE_TAGS = ("rag-agent",)


def _is_configured() -> bool:
    """All three Langfuse env vars must be present and non-empty."""
    return bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
        and os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_BASE_URL")
    )


def get_callbacks() -> list:
    """Return a list with a single Langfuse CallbackHandler, or [].

    The returned list is what you pass to ``graph.invoke(..., config={"callbacks": ...})``
    or to any individual ``llm.invoke(..., config={"callbacks": ...})`` call.
    Cached on first call — safe to call many times.
    """
    global _HANDLER
    if _HANDLER is not None:
        return [_HANDLER]
    if not _is_configured():
        return []
    try:
        from langfuse.langchain import CallbackHandler
        _HANDLER = CallbackHandler()
        return [_HANDLER]
    except Exception as e:
        # Don't crash the agent run on a tracing misconfiguration
        # (e.g. the user typed a wrong host). Just log and continue.
        print(f"[tracing] Langfuse init failed: {type(e).__name__}: {e}",
              flush=True)
        return []


def get_client() -> Any:
    """Return the raw Langfuse client for manual updates (scores, etc.).

    Returns ``None`` if Langfuse is not configured or unavailable.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not _is_configured():
        return None
    try:
        from langfuse import get_client as _lf_get_client
        _CLIENT = _lf_get_client()
        return _CLIENT
    except Exception as e:
        print(f"[tracing] Langfuse client init failed: {type(e).__name__}: {e}",
              flush=True)
        return None


def build_trace_config(
    *,
    question_id: str | None = None,
    session_id: str | None = None,
    model: str | None = None,
    protocol: str | None = None,
    extra_tags: tuple[str, ...] = (),
    extra_metadata: dict | None = None,
    recursion_limit: int = 200,
) -> dict:
    """Build a RunnableConfig dict to pass to ``graph.invoke()``.

    Returns a config that has:

      - ``recursion_limit`` (always)
      - ``callbacks=[langfuse_handler]`` (when Langfuse is configured)
      - top-level ``tags`` (LangChain/LangSmith debug; not the same
        as Langfuse tags)
      - ``metadata`` with ``langfuse_*`` keys the handler forwards
        into ``propagate_attributes()`` (session_id, trace_name,
        tags), plus plain metadata keys the handler leaves on the
        trace's metadata.

    When Langfuse is not configured, returns just
    ``{"recursion_limit": ...}`` — no callbacks, no metadata.

    The returned dict is the only thing callers should pass into
    ``graph.invoke(..., invoke_config)``. Do not also call
    ``propagate_attributes()`` separately — the handler does that
    internally based on the metadata keys below.
    """
    config: dict = {"recursion_limit": recursion_limit}
    callbacks = get_callbacks()
    if not callbacks:
        return config
    config["callbacks"] = callbacks

    # Top-level tags: visible in LangChain debug / LangSmith, and
    # also merged into the Langfuse handler's parsed tags. We keep
    # the list short and stable.
    tags = list(_BASE_TAGS)
    if model:
        tags.append(f"model:{model}")
    if protocol:
        tags.append(f"protocol:{protocol}")
    tags.extend(extra_tags)
    config["tags"] = tags

    # The handler reads these "langfuse_*" keys from the TOP-LEVEL
    # run's metadata and calls propagate_attributes() internally,
    # so the values flow into every nested observation (LLM calls,
    # tool calls, retrievers). Plain metadata keys (no langfuse_
    # prefix) are kept as trace-level metadata on the root span.
    meta: dict[str, Any] = {}
    if session_id:
        meta["langfuse_session_id"] = session_id
    if question_id:
        # Trace name like "rag-qst_0007" — much more findable in the
        # Langfuse UI than the default auto-generated name.
        meta["langfuse_trace_name"] = f"rag-{question_id}"
    if tags:
        meta["langfuse_tags"] = tags
    # Plain metadata (kept on the trace, also pushed to child
    # observations by the handler):
    if model:
        meta["model"] = model
    if protocol:
        meta["protocol"] = protocol
    if question_id:
        meta["question_id"] = question_id
    if extra_metadata:
        # Caller's keys — don't let them stomp the langfuse_* keys.
        for k, v in extra_metadata.items():
            if k not in meta:
                meta[k] = v
    config["metadata"] = meta
    return config


def flush() -> None:
    """Force-flush any pending Langfuse events. Safe to call when off."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        # Flush failures shouldn't break the run
        print(f"[tracing] flush failed: {type(e).__name__}: {e}", flush=True)


def shutdown() -> None:
    """Flush + close the Langfuse client. Call at the very end of a batch."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
        client.shutdown()
    except Exception:
        pass
