import asyncio

from response_cache import CachedResponse, ResponseCache
from storage import TokenUsageStore


def test_identical_cache_requests_call_provider_once(tmp_path):
    db_path = tmp_path / "cache.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        cache = ResponseCache(str(db_path))
        provider_calls = 0

        async def load() -> CachedResponse:
            nonlocal provider_calls
            provider_calls += 1
            await asyncio.sleep(0.05)
            response = CachedResponse("answer", [], "static_fact", "14.1")
            await cache.put("same-key", "14.1", "hash", response.answer, [], "static_fact", 60)
            return response

        results = await asyncio.gather(
            *(cache.get_or_load("same-key", load) for _ in range(5))
        )

        assert provider_calls == 1
        assert results == [CachedResponse("answer", [], "static_fact", "14.1")] * 5
        await store.close()

    asyncio.run(run())


def test_different_cache_keys_proceed_concurrently(tmp_path):
    db_path = tmp_path / "cache.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        cache = ResponseCache(str(db_path))
        started = {"one": asyncio.Event(), "two": asyncio.Event()}
        provider_calls = []

        async def load(key: str) -> CachedResponse:
            provider_calls.append(key)
            started[key].set()
            await asyncio.sleep(0.05)
            response = CachedResponse(key, [], "static_fact", "14.1")
            await cache.put(key, "14.1", key, key, [], "static_fact", 60)
            return response

        first = asyncio.create_task(cache.get_or_load("one", lambda: load("one")))
        second = asyncio.create_task(cache.get_or_load("two", lambda: load("two")))
        await asyncio.wait_for(asyncio.gather(started["one"].wait(), started["two"].wait()), 0.2)
        await asyncio.gather(first, second)

        assert sorted(provider_calls) == ["one", "two"]
        await store.close()

    asyncio.run(run())