import asyncio
import json

import aiosqlite
import pytest

from response_cache import CachedResponse, ResponseCache
from storage import TokenUsageStore


def test_response_cache_repository(tmp_path):
    db_path = tmp_path / "cache.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        cache = ResponseCache(str(db_path))

        await cache.put(
            "key-1",
            "14.1",
            "hash-1",
            "こんにちは 🌟",
            [{"title": "Source"}],
            "static",
            60,
        )
        result = await cache.get("key-1")
        assert result == CachedResponse(
            "こんにちは 🌟", [{"title": "Source"}], "static", "14.1"
        )

        await cache.get("key-1")
        async with aiosqlite.connect(db_path) as connection:
            cursor = await connection.execute(
                "SELECT hit_count FROM response_cache WHERE cache_key = ?",
                ("key-1",),
            )
            assert (await cursor.fetchone())[0] == 2

        await cache.put("key-1", "14.2", "hash-2", "updated", [], "meta", 60)
        assert await cache.get("key-1") == CachedResponse("updated", [], "meta", "14.2")

        await store.close()

        reopened = ResponseCache(str(db_path))
        assert await reopened.get("key-1") == CachedResponse(
            "updated", [], "meta", "14.2"
        )

    asyncio.run(run())


def test_response_cache_expiration_and_cleanup(tmp_path):
    db_path = tmp_path / "cache.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        cache = ResponseCache(str(db_path))
        await cache.put("expired", "14.1", "hash", "old", [], "static", 1)
        await asyncio.sleep(1.1)
        assert await cache.get("expired") is None
        assert await cache.delete_expired() == 1
        await store.close()

    asyncio.run(run())


def test_response_cache_rejects_non_positive_ttl(tmp_path):
    cache = ResponseCache(str(tmp_path / "cache.db"))

    async def run() -> None:
        with pytest.raises(ValueError):
            await cache.put("key", "14.1", "hash", "answer", [], "static", 0)

    asyncio.run(run())


def test_response_cache_malformed_sources_are_misses(tmp_path):
    db_path = tmp_path / "cache.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        cache = ResponseCache(str(db_path))
        await cache.put("malformed", "14.1", "hash", "secret", [], "static", 60)
        async with aiosqlite.connect(db_path) as connection:
            await connection.execute(
                "UPDATE response_cache SET sources_json = ? WHERE cache_key = ?",
                (json.dumps({"not": "a list"}), "malformed"),
            )
            await connection.commit()
        assert await cache.get("malformed") is None
        await store.close()

    asyncio.run(run())


def test_response_cache_patch_invalidation(tmp_path):
    db_path = tmp_path / "cache.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        cache = ResponseCache(str(db_path))
        await cache.put("one", "14.1", "hash-1", "one", [], "static", 60)
        await cache.put("two", "14.2", "hash-2", "two", [], "static", 60)
        assert await cache.invalidate_patch("14.1") == 1
        assert await cache.get("one") is None
        assert await cache.get("two") is not None
        await store.close()

    asyncio.run(run())