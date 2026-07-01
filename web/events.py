"""Event bus — in-memory (single-process) and Redis (cross-process) backends."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)


# ── In-memory event bus (single-process / dev) ──

class RunEventBus:
    def __init__(self) -> None:
        self._queues: dict[int, set[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, run_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(run_id, set()).add(q)
        return q

    def unsubscribe(self, run_id: int, q: asyncio.Queue) -> None:
        if run_id in self._queues:
            self._queues[run_id].discard(q)
            if not self._queues[run_id]:
                del self._queues[run_id]

    def publish(self, run_id: int, message: dict) -> None:
        if self._loop is None:
            return
        queues = self._queues.get(run_id)
        if not queues:
            return
        for q in list(queues):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, message)
            except Exception:
                pass


# ── Redis event bus (cross-process / Celery) ──

class RedisEventBus:
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._redis = None
        self._async_redis = None

    def _get_sync_redis(self):
        if self._redis is None:
            import redis
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def _get_async_redis(self):
        if self._async_redis is None:
            import redis.asyncio as aioredis
            self._async_redis = aioredis.from_url(self.redis_url, decode_responses=True)
        return self._async_redis

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        pass  # Redis handles its own connections

    def publish(self, run_id: int, message: dict) -> None:
        r = self._get_sync_redis()
        try:
            r.publish(f"run:{run_id}", json.dumps(message))
        except Exception:
            logger.exception("Redis publish failed for run %d", run_id)

    async def subscribe(self, run_id: int) -> asyncio.Queue:
        r = await self._get_async_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe(f"run:{run_id}")
        q: asyncio.Queue = asyncio.Queue()

        async def _listener():
            try:
                async for msg in pubsub.listen():
                    if msg["type"] != "message":
                        continue
                    try:
                        data = json.loads(msg["data"])
                        await q.put(data)
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass
            finally:
                await pubsub.unsubscribe(f"run:{run_id}")
                await pubsub.close()

        task = asyncio.create_task(_listener())
        # Store task reference so caller can cancel it on unsubscribe
        q._listener_task = task
        return q

    async def unsubscribe(self, run_id: int, q: asyncio.Queue) -> None:
        task = getattr(q, "_listener_task", None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ── Factory ──

_event_bus_instance: RunEventBus | RedisEventBus | None = None


def get_event_bus(settings: "Settings") -> RunEventBus | RedisEventBus:
    """Return a singleton event bus — Redis-backed if redis_url is set, else in-memory."""
    global _event_bus_instance
    if _event_bus_instance is not None:
        return _event_bus_instance

    if settings.redis_url:
        bus: RunEventBus | RedisEventBus = RedisEventBus(settings.redis_url)
        logger.info("Using Redis event bus (redis_url=%s)", settings.redis_url)
    else:
        bus = RunEventBus()
        logger.info("Using in-memory event bus")
    _event_bus_instance = bus
    return bus


# ── Default (in-memory) instance for backward compatibility ──
event_bus = RunEventBus()
