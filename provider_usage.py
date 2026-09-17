"""Provider usage extraction for the installed Google GenAI response shapes."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("jarvis")


@dataclass(frozen=True)
class ProviderUsage:
    prompt_tokens: int
    completion_tokens: int
    authoritative: bool


def extract_provider_usage(response: Any) -> ProviderUsage | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_token_count", None)
    completion_tokens = getattr(usage, "candidates_token_count", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "response_token_count", None)

    if prompt_tokens is None and completion_tokens is None:
        return None

    return ProviderUsage(
        prompt_tokens=max(0, int(prompt_tokens or 0)),
        completion_tokens=(
            max(0, int(completion_tokens))
            if completion_tokens is not None
            else 0
        ),
        authoritative=prompt_tokens is not None and completion_tokens is not None,
    )


def usage_for_commit(
    response: Any,
    *,
    fallback_prompt_tokens: int,
    fallback_completion_tokens: int,
) -> ProviderUsage:
    usage = extract_provider_usage(response)
    if usage is not None and usage.authoritative:
        return usage

    if usage is not None:
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens or fallback_completion_tokens
    else:
        prompt_tokens = fallback_prompt_tokens
        completion_tokens = fallback_completion_tokens

    logger.warning(
        "Provider token usage unavailable; using conservative fallback"
    )
    return ProviderUsage(
        prompt_tokens=max(0, int(prompt_tokens)),
        completion_tokens=max(0, int(completion_tokens)),
        authoritative=False,
    )


def aggregate_provider_usage(usages: list[ProviderUsage]) -> ProviderUsage | None:
    if not usages:
        return None
    return ProviderUsage(
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        authoritative=all(usage.authoritative for usage in usages),
    )
