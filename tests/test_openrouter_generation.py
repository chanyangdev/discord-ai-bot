import asyncio
import importlib
from collections import deque
from types import SimpleNamespace

import pytest


class FakeStatusMessage:
    def __init__(self):
        self.edits = []

    async def edit(self, *, content):
        self.edits.append(content)


class FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class FakeThread:
    id = "thread-test"

    def __init__(self):
        self.status_message = FakeStatusMessage()

    def typing(self):
        return FakeTyping()

    async def send(self, content):
        self.status_message.edits.append(content)
        return self.status_message


def _load_bot(monkeypatch, **extra_env):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENROUTER_MODELS",
        "test/free-model:free,test/paid-model,test/reliable-model",
    )
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)

    import discord

    monkeypatch.setattr(discord.Client, "run", lambda *args, **kwargs: None)
    import bot

    return importlib.reload(bot)


def test_build_messages_with_zero_prior_turns(monkeypatch):
    bot = _load_bot(monkeypatch)

    messages = bot.build_messages("empty-conversation", "current prompt")

    assert messages[0] == {"role": "system", "content": bot.SYSTEM_PROMPT}
    assert messages[1:] == [{"role": "user", "content": "current prompt"}]


def test_build_messages_keeps_eight_completed_turns_in_order(monkeypatch):
    bot = _load_bot(monkeypatch)
    bot.conversation_history["conversation"] = deque(
        (
            {"user": f"user-{index}", "assistant": f"assistant-{index}"}
            for index in range(8)
        ),
        maxlen=bot.MAX_TURNS,
    )

    messages = bot.build_messages("conversation", "current prompt")

    assert [message["content"] for message in messages[1:-1]] == [
        item
        for index in range(8)
        for item in (f"user-{index}", f"assistant-{index}")
    ]


def test_build_messages_nine_turns_keep_latest_eight(monkeypatch):
    bot = _load_bot(monkeypatch)
    bot.conversation_history["conversation"] = deque(
        (
            {"user": f"user-{index}", "assistant": f"assistant-{index}"}
            for index in range(9)
        ),
        maxlen=bot.MAX_TURNS,
    )

    messages = bot.build_messages("conversation", "current prompt")

    history = [message["content"] for message in messages[1:-1]]
    assert history == [
        item
        for index in range(1, 9)
        for item in (f"user-{index}", f"assistant-{index}")
    ]
    assert "user-0" not in history
    assert "assistant-0" not in history


def test_conversation_ids_remain_isolated(monkeypatch):
    bot = _load_bot(monkeypatch)
    bot.conversation_history["conversation-one"].append(
        {"user": "one-user", "assistant": "one-answer"}
    )
    bot.conversation_history["conversation-two"].append(
        {"user": "two-user", "assistant": "two-answer"}
    )

    first = bot.build_messages("conversation-one", "first current")
    second = bot.build_messages("conversation-two", "second current")

    assert "two-user" not in [message["content"] for message in first]
    assert "one-user" not in [message["content"] for message in second]


@pytest.mark.parametrize("succeeds", [False, True])
def test_answer_in_thread_saves_only_successful_exchange(monkeypatch, succeeds):
    bot = _load_bot(monkeypatch)
    thread = FakeThread()

    async def fake_stream(*args, **kwargs):
        if not succeeds:
            raise RuntimeError("provider failure")
        return "complete answer"

    monkeypatch.setattr(bot, "stream_openrouter_reply", fake_stream)

    asyncio.run(
        bot.answer_in_thread(
            thread,
            "current prompt",
            user_id=1,
            guild_id=2,
        )
    )

    history = list(bot.conversation_history.get(thread.id, ()))
    if succeeds:
        assert history == [{"user": "current prompt", "assistant": "complete answer"}]
    else:
        assert history == []
        assert thread.status_message.edits[-1] == bot.PROVIDER_ERROR_MESSAGE


def test_request_uses_stable_prefix_and_ordered_fallback_models(monkeypatch):
    bot = _load_bot(monkeypatch)
    bot.conversation_history["conversation"].append(
        {"user": "prior user", "assistant": "prior assistant"}
    )
    captured = {}

    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "sent"):
                raise StopAsyncIteration
            self.sent = True
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="complete answer")
                    )
                ],
            )

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeStream()

    class FakeGlobalStore:
        async def get_global_cost_usage(self):
            return bot.GlobalCostUsage("test-date")

        async def reserve_global_cost(self, amount, budget):
            return amount

        async def reconcile_global_cost(self, reserved, actual, *, successful):
            return reserved

        async def release_global_cost(self, reserved):
            return reserved

    monkeypatch.setattr(bot.openrouter.chat.completions, "create", fake_create)
    bot.discord_client.token_usage_store = FakeGlobalStore()
    monkeypatch.setattr(bot, "reserve_user_tokens", _reserve_without_storage)
    monkeypatch.setattr(bot, "update_usage_after_response", _ignore_usage_update)
    monkeypatch.setattr(bot, "release_user_tokens", _ignore_release)

    asyncio.run(
        bot.stream_openrouter_reply(
            FakeThread(),
            "conversation",
            "current prompt",
            FakeStatusMessage(),
            user_id=1,
            guild_id=2,
        )
    )

    messages = captured["messages"]
    assert messages[0] == {"role": "system", "content": bot.SYSTEM_PROMPT}
    assert [message["content"] for message in messages[1:]] == [
        "prior user",
        "prior assistant",
        "current prompt",
    ]
    assert captured["model"] == bot.OPENROUTER_MODELS[0]
    assert captured["extra_body"]["models"] == list(bot.OPENROUTER_MODELS)
    assert 1 <= len(captured["extra_body"]["models"]) <= 3


def test_partial_stream_is_not_retried(monkeypatch):
    bot = _load_bot(monkeypatch)
    calls = []

    class FakeGlobalStore:
        async def get_global_cost_usage(self):
            return bot.GlobalCostUsage("test-date")

        async def reserve_global_cost(self, amount, budget):
            return amount

        async def reconcile_global_cost(self, reserved, actual, *, successful):
            return reserved

        async def release_global_cost(self, reserved):
            return reserved

    class PartialStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, "sent"):
                self.sent = True
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="partial answer")
                        )
                    ]
                )
            raise RuntimeError("stream interrupted")

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return PartialStream()

    bot.discord_client.token_usage_store = FakeGlobalStore()
    monkeypatch.setattr(bot.openrouter.chat.completions, "create", fake_create)
    monkeypatch.setattr(bot, "reserve_user_tokens", _reserve_without_storage)
    monkeypatch.setattr(bot, "release_user_tokens", _ignore_release)

    with pytest.raises(RuntimeError, match="stream interrupted"):
        asyncio.run(
            bot.stream_openrouter_reply(
                FakeThread(),
                "conversation",
                "current prompt",
                FakeStatusMessage(),
                user_id=1,
                guild_id=2,
            )
        )

    assert len(calls) == 1


def test_dm_routes_to_shared_orchestrator_with_guild_zero(monkeypatch):
    bot = _load_bot(monkeypatch)
    captured = {}

    class Message:
        author = SimpleNamespace(bot=False, id=42)
        guild = None
        channel = FakeThread()
        content = "dm prompt"

    async def fake_answer(channel, prompt, *, user_id, guild_id, answer_type=None):
        captured.update(
            prompt=prompt,
            user_id=user_id,
            guild_id=guild_id,
            answer_type=answer_type,
        )

    monkeypatch.setattr(bot, "answer_in_thread", fake_answer)
    asyncio.run(bot.on_message(Message()))

    assert captured == {
        "prompt": "dm prompt",
        "user_id": 42,
        "guild_id": 0,
        "answer_type": None,
    }


def test_owner_only_rejects_dm_interactions(monkeypatch):
    bot = _load_bot(monkeypatch)
    interaction = SimpleNamespace(guild_id=None, user=SimpleNamespace(id=1))
    assert asyncio.run(bot.owner_only(interaction)) is False


async def _reserve_without_storage(*args, **kwargs):
    return 1


async def _ignore_usage_update(*args, **kwargs):
    return None


async def _ignore_release(*args, **kwargs):
    return None


class _FakeGlobalStore:
    def __init__(self, committed=0, reserved=0):
        self.committed = committed
        self.reserved = reserved

    async def get_global_cost_usage(self):
        import bot as bot_module

        return bot_module.GlobalCostUsage(
            "test-date",
            committed_microdollars=self.committed,
            reserved_microdollars=self.reserved,
        )

    async def reserve_global_cost(self, amount, budget):
        return amount

    async def reconcile_global_cost(self, reserved, actual, *, successful):
        return reserved

    async def release_global_cost(self, reserved):
        return reserved


def _capture_openrouter_request(bot, monkeypatch):
    captured = {}

    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "sent"):
                raise StopAsyncIteration
            self.sent = True
            return SimpleNamespace(
                model=captured.get("model"),
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="answer"))
                ],
            )

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeStream()

    monkeypatch.setattr(bot.openrouter.chat.completions, "create", fake_create)
    monkeypatch.setattr(bot, "reserve_user_tokens", _reserve_without_storage)
    monkeypatch.setattr(bot, "update_usage_after_response", _ignore_usage_update)
    monkeypatch.setattr(bot, "release_user_tokens", _ignore_release)
    return captured


def test_live_meta_prompt_routes_to_research_models(monkeypatch):
    bot = _load_bot(monkeypatch)
    captured = _capture_openrouter_request(bot, monkeypatch)
    bot.discord_client.token_usage_store = _FakeGlobalStore()

    asyncio.run(
        bot.stream_openrouter_reply(
            FakeThread(),
            "conversation",
            "current prompt",
            FakeStatusMessage(),
            user_id=1,
            guild_id=2,
            category=bot.RequestCategory.LIVE_META,
        )
    )

    assert captured["model"] == bot.LLM_CONFIG.research_models[0]
    assert captured["extra_body"]["models"] == list(bot.LLM_CONFIG.research_models)


def test_general_chat_prompt_routes_to_general_models(monkeypatch):
    bot = _load_bot(monkeypatch)
    captured = _capture_openrouter_request(bot, monkeypatch)
    bot.discord_client.token_usage_store = _FakeGlobalStore()

    asyncio.run(
        bot.stream_openrouter_reply(
            FakeThread(),
            "conversation",
            "current prompt",
            FakeStatusMessage(),
            user_id=1,
            guild_id=2,
            category=bot.RequestCategory.GENERAL_CHAT,
        )
    )

    assert captured["model"] == bot.LLM_CONFIG.general_models[0]


def test_cheap_budget_threshold_forces_general_route_for_live_meta(monkeypatch):
    bot = _load_bot(monkeypatch)
    captured = _capture_openrouter_request(bot, monkeypatch)
    economy_committed = int(
        bot.POLICY.config.daily_budget_microdollars
        * float(bot.ECONOMY_THRESHOLD)
    )
    bot.discord_client.token_usage_store = _FakeGlobalStore(
        committed=economy_committed
    )

    asyncio.run(
        bot.stream_openrouter_reply(
            FakeThread(),
            "conversation",
            "current prompt",
            FakeStatusMessage(),
            user_id=1,
            guild_id=2,
            category=bot.RequestCategory.LIVE_META,
        )
    )

    assert captured["model"] == bot.LLM_CONFIG.general_models[0]


def test_free_only_env_routes_everything_to_free_model(monkeypatch):
    bot = _load_bot(monkeypatch, LLM_FREE_ONLY="true")
    captured = _capture_openrouter_request(bot, monkeypatch)
    bot.discord_client.token_usage_store = _FakeGlobalStore()

    asyncio.run(
        bot.stream_openrouter_reply(
            FakeThread(),
            "conversation",
            "current prompt",
            FakeStatusMessage(),
            user_id=1,
            guild_id=2,
            category=bot.RequestCategory.LIVE_META,
        )
    )

    assert captured["model"] == bot.LLM_CONFIG.free_model
    assert captured["extra_body"]["models"] == [bot.LLM_CONFIG.free_model]


def test_static_fact_bypasses_llm_when_local_resolver_has_an_answer(monkeypatch):
    bot = _load_bot(monkeypatch)

    async def fake_resolver(question):
        return "local structured-data answer"

    monkeypatch.setattr(bot, "default_static_fact_resolver", fake_resolver)

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("stream_openrouter_reply should not be called")

    monkeypatch.setattr(bot, "stream_openrouter_reply", fail_if_called)
    bot.discord_client.token_usage_store = _FakeGlobalStore()

    status_message = FakeStatusMessage()
    answer, from_cache = asyncio.run(
        bot.generate_interactive_reply(
            FakeThread(),
            "conversation",
            "what is the cooldown of this ability",
            status_message,
            user_id=1,
            guild_id=2,
        )
    )

    assert answer == "local structured-data answer"
    assert from_cache is True


def test_missing_openrouter_api_key_fails_clearly(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    import discord

    monkeypatch.setattr(discord.Client, "run", lambda *args, **kwargs: None)
    import bot

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        importlib.reload(bot)
