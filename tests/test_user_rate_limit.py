import asyncio
import importlib

import aiosqlite
import pytest

import storage
from storage import TokenUsageStore


def test_rate_limit_accepts_limit_then_rejects_without_new_event(monkeypatch, tmp_path):
    db_path = tmp_path / "rate-limit.db"
    monkeypatch.setattr(storage, "_utc_timestamp", lambda: 1_000)

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1_000)
        await store.initialize()
        results = [
            await store.acquire_user_rate_limit(
                user_id=1, limit=10, window_seconds=3_600
            )
            for _ in range(11)
        ]
        assert all(result.accepted for result in results[:10])
        assert not results[10].accepted
        assert results[10].request_count == 10
        assert results[10].retry_after_seconds == 3_600
        async with aiosqlite.connect(db_path) as connection:
            cursor = await connection.execute(
                "SELECT COUNT(*) FROM user_rate_limit_events WHERE user_id = 1"
            )
            assert await cursor.fetchone() == (10,)
        await store.close()

    asyncio.run(run())


def test_rate_limit_is_rolling_user_global_and_persistent(monkeypatch, tmp_path):
    db_path = tmp_path / "rate-limit.db"
    clock = {"value": 1_000}
    monkeypatch.setattr(storage, "_utc_timestamp", lambda: clock["value"])

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1_000)
        await store.initialize()
        assert (
            await store.acquire_user_rate_limit(user_id=1, limit=1, window_seconds=60)
        ).accepted
        assert not (
            await store.acquire_user_rate_limit(user_id=1, limit=1, window_seconds=60)
        ).accepted
        assert (
            await store.acquire_user_rate_limit(user_id=2, limit=1, window_seconds=60)
        ).accepted
        await store.close()

        reopened = TokenUsageStore(str(db_path), daily_limit=1_000)
        await reopened.initialize()
        assert not (
            await reopened.acquire_user_rate_limit(
                user_id=1, limit=1, window_seconds=60
            )
        ).accepted
        clock["value"] = 1_061
        assert (
            await reopened.acquire_user_rate_limit(
                user_id=1, limit=1, window_seconds=60
            )
        ).accepted
        await reopened.close()

    asyncio.run(run())


def test_concurrent_rate_limit_acquisitions_do_not_exceed_limit(tmp_path):
    async def run() -> None:
        store = TokenUsageStore(str(tmp_path / "rate-limit.db"), daily_limit=1_000)
        await store.initialize()
        results = await asyncio.gather(
            *(
                store.acquire_user_rate_limit(user_id=9, limit=10, window_seconds=60)
                for _ in range(20)
            )
        )
        assert sum(result.accepted for result in results) == 10
        await store.close()

    asyncio.run(run())


def test_rate_limit_cancellation_rolls_back_transaction(monkeypatch, tmp_path):
    class CancelledUuid:
        @property
        def hex(self):
            raise asyncio.CancelledError

    async def run() -> None:
        store = TokenUsageStore(str(tmp_path / "rate-limit.db"), daily_limit=1_000)
        await store.initialize()
        monkeypatch.setattr(storage.uuid, "uuid4", CancelledUuid)
        with pytest.raises(asyncio.CancelledError):
            await store.acquire_user_rate_limit(user_id=9, limit=1, window_seconds=60)
        status = await store.get_user_rate_limit_status(
            user_id=9, limit=1, window_seconds=60
        )
        assert status.request_count == 0
        await store.close()

    asyncio.run(run())


def test_rate_limit_cleanup_keeps_active_events(monkeypatch, tmp_path):
    db_path = tmp_path / "rate-limit.db"
    monkeypatch.setattr(storage, "_utc_timestamp", lambda: 1_000)

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1_000)
        await store.initialize()
        async with aiosqlite.connect(db_path) as connection:
            await connection.execute(
                "INSERT INTO user_rate_limit_events VALUES (?, ?, ?)",
                ("expired", 2, 900),
            )
            await connection.execute(
                "INSERT INTO user_rate_limit_events VALUES (?, ?, ?)",
                ("active", 3, 999),
            )
            await connection.commit()
        await store.acquire_user_rate_limit(user_id=1, limit=10, window_seconds=60)
        async with aiosqlite.connect(db_path) as connection:
            cursor = await connection.execute(
                "SELECT event_id FROM user_rate_limit_events ORDER BY event_id"
            )
            event_ids = [row[0] for row in await cursor.fetchall()]
            assert "expired" not in event_ids
            assert "active" in event_ids
            assert len(event_ids) == 2
        await store.close()

    asyncio.run(run())


def test_provider_failure_keeps_accepted_rate_limit_slot(monkeypatch, tmp_path):
    async def run() -> None:
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_MODELS", "qwen/qwen3.8-27b:free")
        import discord

        monkeypatch.setattr(discord.Client, "run", lambda *args, **kwargs: None)
        import bot

        bot = importlib.reload(bot)
        store = TokenUsageStore(str(tmp_path / "rate-limit.db"), daily_limit=1_000)
        await store.initialize()
        bot.discord_client.token_usage_store = store
        bot.discord_client.response_cache = None

        async def failing_provider(*args, **kwargs):
            raise RuntimeError("provider failure")

        monkeypatch.setattr(bot, "stream_openrouter_reply", failing_provider)
        try:
            await bot.generate_interactive_reply(
                object(), "conversation", "question", object(), user_id=4, guild_id=0
            )
        except RuntimeError:
            pass
        status = await store.get_user_rate_limit_status(
            user_id=4,
            limit=bot.USER_RATE_LIMIT_REQUESTS,
            window_seconds=bot.USER_RATE_LIMIT_WINDOW_SECONDS,
        )
        assert status.request_count == 1
        await store.close()

    asyncio.run(run())