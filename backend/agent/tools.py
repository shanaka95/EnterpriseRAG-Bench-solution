"""Tool the reactive agent calls to paginate through the refined docs.

The tool is a closure that captures the live graph state. When the agent
calls it, the tool returns the next ``batch_size`` documents in
(doc_id, content) format and advances the state's cursor. This avoids
threading the state dict through the tool's signature.
"""
from __future__ import annotations
import json
import os
from typing import Any, Callable

from langchain_core.tools import tool

from .config import load_config


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
