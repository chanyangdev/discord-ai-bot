import asyncio
import json
import logging

import aiosqlite

from response_cache import (
    AnswerType,
    CacheTelemetry,
    CachedResponse,
    ResponseCache,
    decide_cache_policy,
)
from storage import TokenUsageStore


def test_cache_events_and_hit_rate_do_not_expose_sensitive_values(tmp_path, caplog):
    db_path = tmp_path / "cache.db"
    sensitive_values = {
        "What is my secret question?",
        "secret answer",
        "user-123",
        "guild-456",
        "https://private.example/source",
        "cache-key-secret",
        "question-hash-secret",
    }

    async def run() -> CacheTelemetry:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        telemetry = CacheTelemetry()
        cache = ResponseCache(str(db_path), telemetry)
        await cache.put(
            "cache-key-secret",
            "14.1",
            "question-hash-secret",
            "secret answer",
            [{"url": "https://private.example/source"}],
            AnswerType.STATIC_FACT.value,
            60,
        )
        await cache.get("cache-key-secret")
        await cache.get("missing-key")
        decide_cache_policy(
            AnswerType.STATIC_FACT,
            depends_on_conversation_history=True,
            telemetry=telemetry,
        )
        await store.close()
        return telemetry

    with caplog.at_level(logging.INFO, logger="response_cache"):
        telemetry = asyncio.run(run())

    assert telemetry.counts["response_cache_write"] == 1
    assert telemetry.counts["response_cache_hit"] == 1
    assert telemetry.counts["response_cache_miss"] == 1
    assert telemetry.counts["response_cache_skip:conversation_history"] == 1
    assert telemetry.hit_rate() == 0.5
    messages = " ".join(record.getMessage() for record in caplog.records)
    for sensitive_value in sensitive_values:
        assert sensitive_value not in messages
    assert "answer_type=static_fact" in messages
    assert "patch_version=14.1" in messages


def test_hit_rate_handles_zero_division():
    assert CacheTelemetry().hit_rate() == 0.0


def test_keyed_recheck_does_not_duplicate_hit_or_miss_metrics(tmp_path):
    db_path = tmp_path / "cache.db"

    async def run() -> CacheTelemetry:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        telemetry = CacheTelemetry()
        cache = ResponseCache(str(db_path), telemetry)
        loader_started = asyncio.Event()
        release_loader = asyncio.Event()

        async def load() -> CachedResponse:
            loader_started.set()
            await release_loader.wait()
            response = CachedResponse("answer", [], AnswerType.STATIC_FACT.value, "14.1")
            await cache.put("same", "14.1", "hash", response.answer, [], response.answer_type, 60)
            return response

        leader = asyncio.create_task(cache.get_or_load("same", load))
        await loader_started.wait()
        follower = asyncio.create_task(cache.get_or_load("same", load))
        await asyncio.sleep(0)
        release_loader.set()
        await asyncio.gather(leader, follower)
        await store.close()
        return telemetry

    telemetry = asyncio.run(run())
    assert telemetry.counts["response_cache_miss"] == 2
    assert telemetry.counts.get("response_cache_hit", 0) == 0


def test_error_event_uses_only_operation_category(tmp_path, caplog):
    db_path = tmp_path / "cache.db"

    async def run() -> CacheTelemetry:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        telemetry = CacheTelemetry()
        cache = ResponseCache(str(db_path), telemetry)
        await cache.put(
            "private-cache-key",
            "14.1",
            "hash",
            "answer",
            [],
            "static_fact",
            60,
        )
        async with aiosqlite.connect(db_path) as connection:
            await connection.execute(
                "UPDATE response_cache SET sources_json = ? WHERE cache_key = ?",
                (json.dumps({"not": "a list"}), "private-cache-key"),
            )
            await connection.commit()
        await cache.get("private-cache-key")
        await store.close()
        return telemetry

    with caplog.at_level(logging.INFO, logger="response_cache"):
        telemetry = asyncio.run(run())

    assert telemetry.counts["response_cache_error:get"] == 1
    assert all("private-cache-key" not in record.getMessage() for record in caplog.records)