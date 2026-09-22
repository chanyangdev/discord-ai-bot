import importlib
import os


def _reload_bot(monkeypatch, **overrides):
    for key, value in overrides.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))

    import discord

    monkeypatch.setattr(discord.Client, "run", lambda *args, **kwargs: None)
    import bot

    return importlib.reload(bot)


def test_response_cache_defaults(monkeypatch):
    monkeypatch.delenv("RESPONSE_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_STATIC_TTL_SECONDS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_META_TTL_SECONDS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_PATCH_NOTES_TTL_SECONDS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_CLEANUP_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_KEY_VERSION", raising=False)
    monkeypatch.delenv("FREE_DAILY_TOKEN_LIMIT", raising=False)
    monkeypatch.delenv("PREMIUM_DAILY_TOKEN_LIMIT", raising=False)
    monkeypatch.delenv("TOKEN_RESERVATION_TTL_SECONDS", raising=False)
    monkeypatch.delenv("USER_RATE_LIMIT_REQUESTS", raising=False)
    monkeypatch.delenv("USER_RATE_LIMIT_WINDOW_SECONDS", raising=False)

    module = _reload_bot(monkeypatch)

    assert module.FREE_DAILY_TOKEN_LIMIT == 200000
    assert module.PREMIUM_DAILY_TOKEN_LIMIT == 1000000
    assert module.TOKEN_QUOTA_RESET_TIMEZONE == "UTC"
    assert module.TOKEN_RESERVATION_TTL_SECONDS == 900
    assert module.USER_RATE_LIMIT_REQUESTS == 10
    assert module.USER_RATE_LIMIT_WINDOW_SECONDS == 3600
    assert module.RESPONSE_CACHE_ENABLED is True
    assert module.RESPONSE_CACHE_STATIC_TTL_SECONDS == 604800
    assert module.RESPONSE_CACHE_META_TTL_SECONDS == 21600
    assert module.RESPONSE_CACHE_PATCH_NOTES_TTL_SECONDS == 86400
    assert module.RESPONSE_CACHE_CLEANUP_INTERVAL_SECONDS == 21600
    assert module.RESPONSE_CACHE_KEY_VERSION == "v1"


def test_response_cache_rejects_invalid_ttls(monkeypatch):
    monkeypatch.setenv("RESPONSE_CACHE_STATIC_TTL_SECONDS", "0")

    try:
        _reload_bot(monkeypatch)
    except RuntimeError as exc:
        assert "RESPONSE_CACHE_STATIC_TTL_SECONDS" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for non-positive TTL")


def test_quota_limits_and_reservation_ttl_reject_non_positive_values(monkeypatch):
    for setting in (
        "FREE_DAILY_TOKEN_LIMIT",
        "PREMIUM_DAILY_TOKEN_LIMIT",
        "TOKEN_RESERVATION_TTL_SECONDS",
        "USER_RATE_LIMIT_REQUESTS",
        "USER_RATE_LIMIT_WINDOW_SECONDS",
    ):
        monkeypatch.setenv(setting, "0")
        try:
            _reload_bot(monkeypatch)
        except RuntimeError as exc:
            assert setting in str(exc)
        else:
            raise AssertionError(f"Expected RuntimeError for {setting}")
        finally:
            monkeypatch.delenv(setting, raising=False)
