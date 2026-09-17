import asyncio
import logging
import time

import pytest

from response_cache import (
    AnswerType,
    CacheTelemetry,
    CachedResponse,
    ResponseCache,
    build_cache_key,
    decide_cache_policy,
    question_hash,
)
from storage import TokenUsageStore


def test_cache_pipeline_end_to_end(tmp_path, caplog):
    db_path = tmp_path / "cache.db"
    sensitive = {"private-question", "private-answer", "private-source", "private-key"}

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        telemetry = CacheTelemetry()
        cache = ResponseCache(str(db_path), telemetry)
        provider_calls = 0
        search_calls = 0
        reservations = 0

        async def request(
            question: str,
            patch: str,
            *,
            answer_type: AnswerType = AnswerType.STATIC_FACT,
            history: bool = False,
            player_specific: bool = False,
            behavior: str = "success",
            ttl: int = 60,
        ) -> CachedResponse:
            nonlocal provider_calls, search_calls, reservations
            policy = decide_cache_policy(
                answer_type,
                depends_on_conversation_history=history,
                player_specific=player_specific,
                telemetry=telemetry,
            )
            key = build_cache_key(question, patch)

            async def search_and_provider() -> CachedResponse:
                nonlocal provider_calls, search_calls, reservations
                search_calls += 1
                reservations += 1
                if behavior == "cancel":
                    raise asyncio.CancelledError
                if behavior in {"fail", "partial", "refused"}:
                    raise RuntimeError(behavior)
                provider_calls += 1
                return CachedResponse(
                    f"answer for {question}",
                    [{"title": "private-source"}],
                    answer_type.value,
                    patch,
                )

            return await cache.load_and_cache(
                key,
                search_and_provider,
                question_hash=question_hash(question),
                ttl_seconds=ttl,
                cacheable=policy.cacheable,
            )

        first = await request("What is this?", "14.1")
        second = await request("  WHAT   IS THIS? ", "14.1")
        assert first == second
        assert search_calls == 1
        assert provider_calls == 1
        assert reservations == 1

        await request("What is this?", "14.2")
        assert provider_calls == 2

        await request("expires", "14.1", ttl=1)
        await asyncio.sleep(1.1)
        await request("expires", "14.1")
        assert provider_calls == 4

        reconnected = ResponseCache(str(db_path))
        assert await reconnected.get(build_cache_key("What is this?", "14.1")) is not None

        for kwargs in (
            {"player_specific": True},
            {"history": True},
        ):
            await request("private-question", "14.1", **kwargs)
        assert provider_calls == 6

        for behavior in ("fail", "cancel", "refused", "partial"):
            with pytest.raises((RuntimeError, asyncio.CancelledError)):
                await request(f"{behavior}-question", "14.1", behavior=behavior)
        assert await cache.get(build_cache_key("fail-question", "14.1")) is None

        original_get = cache.get

        async def failing_get(*args, **kwargs):
            raise OSError("private-key")

        cache.get = failing_get
        await request("read-failure", "14.1")
        cache.get = original_get

        original_put = cache.put

        async def failing_put(*args, **kwargs):
            raise OSError("private-key")

        cache.put = failing_put
        await request("write-failure", "14.1")
        cache.put = original_put

        await store.close()

    with caplog.at_level(logging.INFO, logger="response_cache"):
        asyncio.run(run())

    messages = " ".join(record.getMessage() for record in caplog.records)
    for value in sensitive:
        assert value not in messages


def test_different_keys_execute_concurrently(tmp_path):
    db_path = tmp_path / "cache.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        cache = ResponseCache(str(db_path))
        started = {"one": asyncio.Event(), "two": asyncio.Event()}

        async def load(key: str) -> CachedResponse:
            started[key].set()
            await asyncio.sleep(0.03)
            response = CachedResponse(key, [], AnswerType.STATIC_FACT.value, "14.1")
            await cache.put(key, "14.1", key, key, [], response.answer_type, 60)
            return response

        first = asyncio.create_task(cache.get_or_load("one", lambda: load("one")))
        second = asyncio.create_task(cache.get_or_load("two", lambda: load("two")))
        await asyncio.wait_for(
            asyncio.gather(started["one"].wait(), started["two"].wait()),
            0.2,
        )
        await asyncio.gather(first, second)
        await store.close()

    asyncio.run(run())