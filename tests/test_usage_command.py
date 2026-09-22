import asyncio
import importlib

from storage import DailyUsage, RateLimitResult


class FakeUsageStore:
    def __init__(self):
        self.requested_user_ids = []

    async def get_daily_usage(self, user_id):
        self.requested_user_ids.append(user_id)
        return DailyUsage(
            user_id=user_id,
            usage_date="2026-09-18",
            prompt_tokens=120,
            completion_tokens=30,
            reserved_tokens=400,
            successful_requests=2,
            updated_at=123,
        )

    async def get_user_rate_limit_status(self, **kwargs):
        return RateLimitResult(True, 4, 10)


class FakeQuotaService:
    def __init__(self):
        self.requested_user_ids = []

    async def resolve_daily_limit(self, user_id):
        self.requested_user_ids.append(user_id)
        return 1000


class FakeResponse:
    def __init__(self):
        self.content = None
        self.ephemeral = None

    async def send_message(self, content, *, ephemeral):
        self.content = content
        self.ephemeral = ephemeral


class FakeInteraction:
    def __init__(self, user_id):
        self.user = type("User", (), {"id": user_id})()
        self.response = FakeResponse()


def _load_bot(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import discord

    monkeypatch.setattr(discord.Client, "run", lambda *args, **kwargs: None)
    import bot

    return importlib.reload(bot)


def test_usage_command_is_private_and_user_scoped(monkeypatch):
    bot = _load_bot(monkeypatch)
    usage_store = FakeUsageStore()
    quota_service = FakeQuotaService()
    bot.discord_client.token_usage_store = usage_store
    bot.discord_client.quota_service = quota_service
    interaction = FakeInteraction(42)

    asyncio.run(bot.usage_command.callback(interaction))

    assert usage_store.requested_user_ids == [42]
    assert quota_service.requested_user_ids == [42]
    assert interaction.response.ephemeral is True
    assert "AI requests: `4 / 10` in the last hour" in interaction.response.content
    assert "Daily tokens: `150 / 1,000`" in interaction.response.content
    assert "Daily tokens remaining: `850`" in interaction.response.content
    assert "00:00 UTC" in interaction.response.content
    assert "reserved" not in interaction.response.content.lower()
    assert "updated" not in interaction.response.content.lower()
    assert "user_id" not in interaction.response.content.lower()


def test_usage_formatter_never_shows_private_fields(monkeypatch):
    bot = _load_bot(monkeypatch)
    response = bot.format_usage_response(
        DailyUsage(
            user_id=9,
            usage_date="2026-09-18",
            prompt_tokens=10,
            completion_tokens=5,
            reserved_tokens=99,
            successful_requests=3,
            updated_at=456,
        ),
        20,
        RateLimitResult(True, 3, 10),
    )

    assert response == (
        "**AI usage**\n"
        "AI requests: `3 / 10` in the last hour\n"
        "Daily tokens: `15 / 20`\n"
        "Daily tokens remaining: `5`\n"
        "Token reset: `00:00 UTC`"
    )
