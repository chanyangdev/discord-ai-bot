import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from storage import QuotaExceeded, TokenUsageStore


def _utc_day(offset_days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset_days)).strftime(
        "%Y-%m-%d"
    )


def test_persistence_after_reconnect(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        await store.reserve_tokens(101, 202, _utc_day(), 300)
        await store.apply_usage(101, 202, _utc_day(), 300, 250, 50)
        await store.close()

        reopened = TokenUsageStore(str(db_path), daily_limit=1000)
        await reopened.initialize()
        try:
            assert await reopened.usage_for_day(101, 202, _utc_day()) == 300
            with pytest.raises(QuotaExceeded):
                await reopened.reserve_tokens(101, 202, _utc_day(), 801)
        finally:
            await reopened.close()

    asyncio.run(run())


def test_concurrent_near_limit_reservations(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()

        async def reserve(value: int) -> bool:
            try:
                await store.reserve_tokens(5, 6, _utc_day(), value)
                return True
            except QuotaExceeded:
                return False

        results = await asyncio.gather(
            reserve(600),
            reserve(600),
        )

        assert results.count(True) == 1
        assert await store.usage_for_day(5, 6, _utc_day()) == 600
        await store.close()

    asyncio.run(run())


def test_failed_call_releases_reservation(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        today = _utc_day()
        await store.reserve_tokens(7, 8, today, 400)
        await store.release_tokens(7, 8, today, 400)
        assert await store.usage_for_day(7, 8, today) == 0
        await store.close()

    asyncio.run(run())


def test_dm_guild_id_handling(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        today = _utc_day()
        await store.reserve_tokens(9, 0, today, 120)
        assert await store.usage_for_day(9, 0, today) == 120
        await store.close()

    asyncio.run(run())


def test_utc_day_isolation(tmp_path):
    db_path = tmp_path / "quota.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        today = _utc_day()
        yesterday = _utc_day(-1)

        await store.reserve_tokens(11, 12, today, 300)
        await store.reserve_tokens(11, 12, yesterday, 250)

        assert await store.usage_for_day(11, 12, today) == 300
        assert await store.usage_for_day(11, 12, yesterday) == 250
        await store.close()

    asyncio.run(run())
