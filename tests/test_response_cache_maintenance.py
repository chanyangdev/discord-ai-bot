import asyncio
import importlib

import pytest

from response_cache import parse_canonical_patch_version
from storage import TokenUsageStore


def _load_bot(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "cache.db"))

    import discord

    monkeypatch.setattr(discord.Client, "run", lambda *args, **kwargs: None)
    import bot

    return importlib.reload(bot)


def test_startup_and_periodic_cleanup(monkeypatch, tmp_path):
    bot = _load_bot(monkeypatch, tmp_path)
    bot.RESPONSE_CACHE_CLEANUP_INTERVAL_SECONDS = 0.01

    async def run() -> None:
        store = TokenUsageStore(bot.SQLITE_PATH, daily_limit=1000)
        await store.initialize()
        cache = bot.ResponseCache(bot.SQLITE_PATH)
        await cache.put("expired", "14.1", "hash", "old", [], "static", 1)
        await asyncio.sleep(1.1)

        client = bot.Bot(intents=bot.intents)
        await client.setup_hook()
        assert await cache.get("expired") is None

        await cache.put("later", "14.1", "hash", "old", [], "static", 1)
        await asyncio.sleep(1.1)
        await asyncio.sleep(0.03)
        assert await cache.get("later") is None
        await client.close()
        await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("patch_version", ["latest", "14", "14.1.0", "v14.1", ""])
def test_invalid_patch_input_is_rejected(patch_version):
    with pytest.raises(ValueError):
        parse_canonical_patch_version(patch_version)


def test_canonical_patch_input_and_patch_scoped_deletion(tmp_path):
    from response_cache import ResponseCache

    async def run() -> None:
        store = TokenUsageStore(str(tmp_path / "cache.db"), daily_limit=1000)
        await store.initialize()
        cache = ResponseCache(str(tmp_path / "cache.db"))
        await cache.put("one", "14.1", "hash-1", "one", [], "static", 60)
        await cache.put("two", "14.2", "hash-2", "two", [], "static", 60)
        assert await cache.invalidate_patch(parse_canonical_patch_version("14.1")) == 1
        assert await cache.get("one") is None
        assert await cache.get("two") is not None
        await store.close()

    asyncio.run(run())


def test_invalidation_command_requires_owner_check(monkeypatch, tmp_path):
    bot = _load_bot(monkeypatch, tmp_path)
    assert bot.invalidate_cache_command.checks == [bot.owner_only]

    async def run() -> None:
        owner = type("User", (), {"id": 123})()
        non_owner = type("User", (), {"id": 456})()

        async def application_info():
            return type(
                "ApplicationInfo",
                (),
                {"owner": type("User", (), {"id": owner.id})()},
            )()

        monkeypatch.setattr(bot.discord_client, "application_info", application_info)
        assert await bot.owner_only(type("Interaction", (), {"user": owner})())
        assert not await bot.owner_only(
            type("Interaction", (), {"user": non_owner})()
        )

    asyncio.run(run())
