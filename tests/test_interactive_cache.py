import asyncio
import importlib
from response_cache import AnswerType, ResponseCache, build_cache_key, question_hash
from storage import TokenUsageStore


class FakeStatus:
    def __init__(self):
        self.edits = []

    async def edit(self, *, content):
        self.edits.append(content)


class FakeChannel:
    async def send(self, content):
        return None


async def _setup_bot(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENROUTER_MODELS",
        "qwen/qwen3.8-27b:free,test/paid,test/reliable",
    )
    import discord

    monkeypatch.setattr(discord.Client, "run", lambda *args, **kwargs: None)
    import bot

    bot = importlib.reload(bot)
    store = TokenUsageStore(str(tmp_path / "cache.db"), daily_limit=1000)
    await store.initialize()
    bot.discord_client.token_usage_store = store
    bot.discord_client.response_cache = ResponseCache(str(tmp_path / "cache.db"))
    return bot, store


def test_cache_hit_bypasses_generation_and_history(monkeypatch, tmp_path):
    async def run():
        bot, store = await _setup_bot(monkeypatch, tmp_path)
        prompt = "safe build question"
        key = build_cache_key(prompt, bot.RESPONSE_CACHE_PATCH_VERSION)
        await bot.discord_client.response_cache.put(
            key,
            bot.RESPONSE_CACHE_PATCH_VERSION,
            question_hash(prompt),
            "cached answer",
            [],
            AnswerType.BUILD_META.value,
            60,
        )
        provider_calls = []

        async def should_not_call(*args, **kwargs):
            provider_calls.append(1)
            raise AssertionError("provider called on cache hit")

        monkeypatch.setattr(bot, "stream_openrouter_reply", should_not_call)
        answer, from_cache = await bot.generate_interactive_reply(
            FakeChannel(),
            "cache-conversation",
            prompt,
            FakeStatus(),
            user_id=1,
            guild_id=2,
            answer_type=AnswerType.BUILD_META,
        )
        assert (answer, from_cache) == ("cached answer", True)
        assert provider_calls == []
        assert list(bot.conversation_history.get("cache-conversation", ())) == []
        await store.close()

    asyncio.run(run())


def test_cache_miss_generates_and_writes_one_final_answer(monkeypatch, tmp_path):
    async def run():
        bot, store = await _setup_bot(monkeypatch, tmp_path)

        async def generate(*args, **kwargs):
            return "complete generated answer"

        monkeypatch.setattr(bot, "stream_openrouter_reply", generate)
        answer, from_cache = await bot.generate_interactive_reply(
            FakeChannel(),
            "cache-conversation",
            "cache miss question",
            FakeStatus(),
            user_id=1,
            guild_id=2,
            answer_type=AnswerType.BUILD_META,
        )
        key = build_cache_key("cache miss question", bot.RESPONSE_CACHE_PATCH_VERSION)
        cached = await bot.discord_client.response_cache.get(key)
        assert answer == "complete generated answer"
        assert from_cache is False
        assert cached.answer == answer
        await store.close()

    asyncio.run(run())


def test_duplicate_cache_misses_coalesce(monkeypatch, tmp_path):
    async def run():
        bot, store = await _setup_bot(monkeypatch, tmp_path)
        provider_calls = 0
        started = asyncio.Event()

        async def generate(*args, **kwargs):
            nonlocal provider_calls
            provider_calls += 1
            started.set()
            await asyncio.sleep(0.02)
            return "coalesced answer"

        monkeypatch.setattr(bot, "stream_openrouter_reply", generate)
        request = lambda: bot.generate_interactive_reply(
            FakeChannel(),
            "coalesced",
            "same question",
            FakeStatus(),
            user_id=1,
            guild_id=2,
            answer_type=AnswerType.BUILD_META,
        )
        first = asyncio.create_task(request())
        await started.wait()
        second = asyncio.create_task(request())
        assert await first == ("coalesced answer", False)
        assert await second == ("coalesced answer", True)
        assert provider_calls == 1
        await store.close()

    asyncio.run(run())


def test_free_only_cache_hit_and_miss_never_call_provider(monkeypatch, tmp_path):
    async def run():
        bot, store = await _setup_bot(monkeypatch, tmp_path)
        await store.reserve_global_cost(
            bot.POLICY.config.daily_budget_microdollars,
            bot.POLICY.config.daily_budget_microdollars,
        )
        await store.reconcile_global_cost(
            bot.POLICY.config.daily_budget_microdollars,
            bot.POLICY.config.daily_budget_microdollars,
            successful=True,
        )
        calls = []

        async def should_not_call(*args, **kwargs):
            calls.append(1)
            raise AssertionError("provider called in FREE_ONLY")

        monkeypatch.setattr(bot, "stream_openrouter_reply", should_not_call)
        prompt = "free-only cache question"
        key = build_cache_key(prompt, bot.RESPONSE_CACHE_PATCH_VERSION)
        await bot.discord_client.response_cache.put(
            key,
            bot.RESPONSE_CACHE_PATCH_VERSION,
            question_hash(prompt),
            "free cached answer",
            [],
            AnswerType.BUILD_META.value,
            60,
        )
        answer, from_cache = await bot.generate_interactive_reply(
            FakeChannel(),
            "free-cache",
            prompt,
            FakeStatus(),
            user_id=1,
            guild_id=2,
            answer_type=AnswerType.BUILD_META,
        )
        assert (answer, from_cache) == ("free cached answer", True)

        try:
            await bot.generate_interactive_reply(
                FakeChannel(),
                "free-miss",
                "free-only miss",
                FakeStatus(),
                user_id=1,
                guild_id=2,
                answer_type=AnswerType.BUILD_META,
            )
        except bot.BudgetUnavailable:
            pass
        assert calls == []
        await store.close()

    asyncio.run(run())


def test_cache_read_failure_continues_uncached(monkeypatch, tmp_path):
    async def run():
        bot, store = await _setup_bot(monkeypatch, tmp_path)
        cache = bot.discord_client.response_cache
        async def failing_get(*args, **kwargs):
            raise OSError("cache unavailable")

        monkeypatch.setattr(cache, "get", failing_get)

        async def generate(*args, **kwargs):
            return "uncached answer"

        monkeypatch.setattr(bot, "stream_openrouter_reply", generate)
        answer, from_cache = await bot.generate_interactive_reply(
            FakeChannel(),
            "uncached",
            "cache failure question",
            FakeStatus(),
            user_id=1,
            guild_id=2,
            answer_type=AnswerType.BUILD_META,
        )
        assert (answer, from_cache) == ("uncached answer", False)
        await store.close()

    asyncio.run(run())
