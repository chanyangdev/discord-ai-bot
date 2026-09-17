import importlib
import os


def _reload_bot(monkeypatch, **overrides):
    for key, value in overrides.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))

    import bot

    return importlib.reload(bot)


def test_response_cache_defaults(monkeypatch):
    monkeypatch.delenv("RESPONSE_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_STATIC_TTL_SECONDS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_META_TTL_SECONDS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_PATCH_NOTES_TTL_SECONDS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_CLEANUP_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_KEY_VERSION", raising=False)

    module = _reload_bot(monkeypatch)

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
