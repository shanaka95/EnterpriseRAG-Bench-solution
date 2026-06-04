"""
SSE (Server-Sent Events) endpoint for real-time tree updates.
Broadcasts cluster tree building events to connected clients.
"""
import json
import logging
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
import redis as redis_lib

from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["sse"])


async def event_generator() -> AsyncGenerator[dict, None]:
    """
    Subscribe to Redis pub/sub and yield SSE events.
    """
    r = redis_lib.Redis.from_url(settings.REDIS_URL)
    pubsub = r.pubsub()
    pubsub.subscribe("cluster_tree_updates")

    logger.info("SSE client connected, listening for tree updates...")

    try:
        loop = asyncio.get_event_loop()
        while True:
            # Run blocking Redis call in thread pool so the event loop stays responsive
            message = await loop.run_in_executor(None, pubsub.get_message, 1.0)
            if message and message["type"] == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                yield {
                    "event": "tree_update",
                    "data": data,
                }
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        logger.info("SSE client disconnected.")
    except Exception as e:
        logger.error(f"SSE error: {e}")
    finally:
        pubsub.unsubscribe()
        r.close()


@router.get("/stream")
async def stream_updates():
    """SSE endpoint for real-time cluster tree updates."""
    return EventSourceResponse(event_generator())
