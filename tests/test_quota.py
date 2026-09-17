import asyncio

import pytest

from storage import DailyUsage, QuotaExceeded, TokenUsageStore


def test_persistence_and_independent_users(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        await store.reserve_tokens(1, 300, 1000)
        await store.commit_tokens(1, 300, 250, 50)
        await store.reserve_tokens(2, 1000, 1000)
        assert (await store.get_daily_usage(2)).reserved_tokens == 1000
        await store.close()

        reopened = TokenUsageStore(str(db_path), daily_limit=1000)
        await reopened.initialize()
        user_one = await reopened.get_daily_usage(1)
        assert user_one.prompt_tokens == 250
        assert user_one.completion_tokens == 50
        assert user_one.successful_requests == 1
        assert (await reopened.get_daily_usage(2)).reserved_tokens == 0
        await reopened.close()

    asyncio.run(run())


def test_startup_recovers_crashed_reservations(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        await store.reserve_tokens(3, 400, 1000)
        await store.close()

        recovered = TokenUsageStore(str(db_path), daily_limit=1000)
        await recovered.initialize()
        assert (await recovered.get_daily_usage(3)).reserved_tokens == 0
        await recovered.close()

    asyncio.run(run())


def test_repeated_initialization_preserves_live_reservations(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        await store.reserve_tokens(4, 400, 1000)
        await store.initialize()
        assert (await store.get_daily_usage(4)).reserved_tokens == 400
        await store.close()

    asyncio.run(run())


def test_cross_guild_aggregation_is_user_global(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        await store.reserve_tokens(5, 600, 1000)
        with pytest.raises(QuotaExceeded):
            await store.reserve_tokens(5, 500, 1000)
        assert (await store.get_daily_usage(5)).reserved_tokens == 600
        await store.close()

    asyncio.run(run())


def test_atomic_concurrent_reservations(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()

        async def reserve() -> bool:
            try:
                await store.reserve_tokens(7, 600, 1000)
                return True
            except QuotaExceeded:
                return False

        results = await asyncio.gather(reserve(), reserve())
        assert results.count(True) == 1
        assert (await store.get_daily_usage(7)).reserved_tokens == 600
        await store.close()

    asyncio.run(run())


def test_success_reconciliation_and_failure_release(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        await store.reserve_tokens(8, 400, 1000)
        assert await store.commit_tokens(8, 400, 250, 50) == 300
        usage = await store.get_daily_usage(8)
        assert usage.reserved_tokens == 0
        assert usage.prompt_tokens == 250
        assert usage.completion_tokens == 50
        await store.reserve_tokens(9, 400, 1000)
        await store.release_tokens(9, 400)
        assert (await store.get_daily_usage(9)).reserved_tokens == 0
        with pytest.raises(LookupError):
            await store.release_tokens(9, 400)
        await store.close()

    asyncio.run(run())


def test_cancellation_is_safe(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        await store.reserve_tokens(10, 400, 1000)
        task = asyncio.create_task(store.commit_tokens(10, 400, 1, 1))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await store.release_tokens(10, 400)
        assert (await store.get_daily_usage(10)).reserved_tokens == 0
        await store.close()

    asyncio.run(run())


def test_utc_day_isolation_and_empty_usage(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        usage = await store.get_daily_usage(11)
        assert isinstance(usage, DailyUsage)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.reserved_tokens == 0
        assert usage.usage_date
        await store.close()

    asyncio.run(run())
