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


def _load_bot(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENROUTER_MODELS",
        "test/free-model,test/paid-model,test/reliable-model",
    )

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

    monkeypatch.setattr(bot.openrouter.chat.completions, "create", fake_create)
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
    assert captured["extra_body"]["models"] == bot.OPENROUTER_MODELS
    assert 1 <= len(captured["extra_body"]["models"]) <= 3


async def _reserve_without_storage(*args, **kwargs):
    return 1


async def _ignore_usage_update(*args, **kwargs):
    return None


async def _ignore_release(*args, **kwargs):
    return None
