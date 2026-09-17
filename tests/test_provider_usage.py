from types import SimpleNamespace

from google.genai import types

from provider_usage import (
    ProviderUsage,
    aggregate_provider_usage,
    extract_provider_usage,
    usage_for_commit,
)


def test_stream_response_uses_candidates_token_count():
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=41,
        candidates_token_count=9,
    )
    response = types.GenerateContentResponse(usage_metadata=usage)

    assert extract_provider_usage(response) == ProviderUsage(41, 9, True)


def test_non_stream_response_uses_response_token_count():
    usage = types.UsageMetadata(
        prompt_token_count=31,
        response_token_count=8,
    )
    response = SimpleNamespace(usage_metadata=usage)

    assert extract_provider_usage(response) == ProviderUsage(31, 8, True)


def test_missing_usage_uses_conservative_prompt_and_completion_fallback(caplog):
    result = usage_for_commit(
        SimpleNamespace(usage_metadata=None),
        fallback_prompt_tokens=120,
        fallback_completion_tokens=1024,
    )

    assert result == ProviderUsage(120, 1024, False)
    assert "Provider token usage unavailable" in caplog.text


def test_round_usage_is_aggregated():
    assert aggregate_provider_usage(
        [ProviderUsage(10, 4, True), ProviderUsage(20, 6, True)]
    ) == ProviderUsage(30, 10, True)
