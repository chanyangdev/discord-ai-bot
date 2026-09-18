"""Centralized OpenRouter task-based model routing.

This module is the single place where OpenRouter model IDs, request
categories, and route selection live. Commands and generation code should
import from here instead of hardcoding model IDs.

Routing strategy (see README.md for the full write-up):

- ``static_fact``: answered from local/structured data when possible; falls
  back to the general route only when an LLM is actually needed.
- ``general_chat``: always uses the general route.
- ``live_meta``: uses the research route, ideally after current patch/search
  context has been retrieved (see :mod:`live_search`).
- ``player_specific``: uses Riot API data when available (see
  :mod:`riot_api`), then the research route to synthesize it. Never invents
  player data when the Riot API is unavailable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable, Mapping, Sequence

logger = logging.getLogger("jarvis.llm_routing")

DEFAULT_GENERAL_MODELS: tuple[str, ...] = (
    "qwen/qwen3.7-flash",
    "deepseek/deepseek-v4-flash-0731",
    "qwen/qwen3.7-plus",
)
DEFAULT_RESEARCH_MODELS: tuple[str, ...] = (
    "google/gemini-3.7-flash",
    "qwen/qwen3.7-plus",
    "z-ai/glm-5.3-flash",
)
DEFAULT_FREE_MODEL = "openrouter/free"


class RequestCategory(StrEnum):
    STATIC_FACT = "static_fact"
    GENERAL_CHAT = "general_chat"
    LIVE_META = "live_meta"
    PLAYER_SPECIFIC = "player_specific"


class ModelRoute(StrEnum):
    GENERAL = "general"
    RESEARCH = "research"
    FREE = "free_only"


class RoutingMode(StrEnum):
    NORMAL = "normal"
    FREE_ONLY = "free_only"


def parse_model_list(
    value: str,
    *,
    min_models: int = 1,
    max_models: int | None = None,
) -> tuple[str, ...]:
    """Parse a comma-separated model list.

    Trims whitespace around each entry and rejects empty model IDs.
    """
    models = tuple(model.strip() for model in value.split(","))
    models = tuple(model for model in models if model)
    if len(models) < min_models:
        raise ValueError("model list must contain at least one non-empty model ID")
    if max_models is not None and len(models) > max_models:
        raise ValueError(f"model list must contain at most {max_models} model IDs")
    return models


def _resolve_env_value(
    getenv: Callable[[str], str | None],
    preferred: str,
    deprecated: str | None = None,
) -> str | None:
    value = getenv(preferred)
    if value is not None:
        return value
    if deprecated is not None:
        value = getenv(deprecated)
        if value is not None:
            logger.warning(
                "%s is deprecated; use %s instead.", deprecated, preferred
            )
            return value
    return None


def _get_bool(getenv: Callable[[str], str | None], name: str, default: bool) -> bool:
    value = getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value like true/false or 1/0.")


def _get_positive_int(
    getenv: Callable[[str], str | None], name: str, default: int
) -> int:
    value = getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return parsed


def _get_positive_float(
    getenv: Callable[[str], str | None], name: str, default: float
) -> float:
    value = getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive number.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive number.")
    return parsed


@dataclass(frozen=True)
class LLMConfig:
    general_models: tuple[str, ...]
    research_models: tuple[str, ...]
    free_model: str
    routing_mode: RoutingMode
    free_only: bool
    max_output_tokens_general: int
    max_output_tokens_research: int
    request_timeout_seconds: float
    max_retries: int
    http_referer: str
    app_name: str

    @property
    def effective_free_only(self) -> bool:
        return self.free_only or self.routing_mode is RoutingMode.FREE_ONLY


def load_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """Load LLM routing configuration from environment variables.

    Backward compatibility: ``OPENROUTER_MODELS`` (this repository's existing
    single list) and the more generic ``OPENROUTER_MODEL``/``LLM_MODEL``
    single-model overrides are still honored. When set, they override the
    general route's primary model and a non-fatal deprecation warning is
    logged. Prefer ``OPENROUTER_GENERAL_MODELS`` going forward.
    """
    source = env if env is not None else os.environ
    getenv = source.get

    general_raw = _resolve_env_value(getenv, "OPENROUTER_GENERAL_MODELS") or ",".join(
        DEFAULT_GENERAL_MODELS
    )
    research_raw = _resolve_env_value(getenv, "OPENROUTER_RESEARCH_MODELS") or ",".join(
        DEFAULT_RESEARCH_MODELS
    )
    free_model = (getenv("OPENROUTER_FREE_MODEL") or DEFAULT_FREE_MODEL).strip()
    if not free_model:
        raise RuntimeError("OPENROUTER_FREE_MODEL must not be empty.")

    try:
        general_models = list(parse_model_list(general_raw))
    except ValueError as exc:
        raise RuntimeError(f"OPENROUTER_GENERAL_MODELS: {exc}") from exc
    try:
        research_models = list(parse_model_list(research_raw))
    except ValueError as exc:
        raise RuntimeError(f"OPENROUTER_RESEARCH_MODELS: {exc}") from exc

    deprecated_list = getenv("OPENROUTER_MODELS")
    if deprecated_list:
        try:
            parsed_list = parse_model_list(deprecated_list, max_models=3)
        except ValueError as exc:
            raise RuntimeError(f"OPENROUTER_MODELS: {exc}") from exc
        logger.warning(
            "OPENROUTER_MODELS is deprecated; use OPENROUTER_GENERAL_MODELS "
            "instead. Using it to override the general model route for this run."
        )
        general_models = list(parsed_list)

    deprecated_single = getenv("OPENROUTER_MODEL") or getenv("LLM_MODEL")
    if deprecated_single:
        override = deprecated_single.strip()
        if override:
            logger.warning(
                "OPENROUTER_MODEL/LLM_MODEL is deprecated; set the general "
                "primary model via OPENROUTER_GENERAL_MODELS instead. Using "
                "%s as the general primary model for this run.",
                override,
            )
            general_models = [override] + [m for m in general_models if m != override]

    routing_mode_raw = (
        _resolve_env_value(getenv, "LLM_ROUTING_MODE") or RoutingMode.NORMAL.value
    ).strip().lower()
    try:
        routing_mode = RoutingMode(routing_mode_raw)
    except ValueError as exc:
        allowed = [mode.value for mode in RoutingMode]
        raise RuntimeError(f"LLM_ROUTING_MODE must be one of {allowed}.") from exc

    return LLMConfig(
        general_models=tuple(general_models),
        research_models=tuple(research_models),
        free_model=free_model,
        routing_mode=routing_mode,
        free_only=_get_bool(getenv, "LLM_FREE_ONLY", False),
        max_output_tokens_general=_get_positive_int(
            getenv, "LLM_MAX_OUTPUT_TOKENS_GENERAL", 500
        ),
        max_output_tokens_research=_get_positive_int(
            getenv, "LLM_MAX_OUTPUT_TOKENS_RESEARCH", 900
        ),
        request_timeout_seconds=_get_positive_float(
            getenv, "LLM_REQUEST_TIMEOUT_SECONDS", 45.0
        ),
        max_retries=_get_positive_int(getenv, "LLM_MAX_RETRIES", 2),
        http_referer=(
            getenv("OPENROUTER_HTTP_REFERER") or getenv("OPENROUTER_SITE_URL") or ""
        ).strip(),
        app_name=(getenv("OPENROUTER_APP_NAME") or "Jarvis Discord Bot").strip(),
    )


_LIVE_META_COMMANDS = {"meta", "patch", "tierlist", "tier-list"}
_PLAYER_COMMANDS = {"profile", "stats", "rank", "summoner"}
_STATIC_FACT_COMMANDS = {"static", "fact", "lookup", "wiki"}

_LIVE_META_KEYWORDS = (
    "patch notes",
    "current patch",
    "latest patch",
    "this patch",
    "tier list",
    "tier-list",
    "meta right now",
    "current meta",
    "buffed",
    "nerfed",
    "win rate",
    "winrate",
    "pick rate",
    "op.gg/champions",
    "u.gg/tier-list",
)
_PLAYER_KEYWORDS = (
    "my account",
    "my summoner",
    "my rank",
    "my stats",
    "my match",
    "my games",
    "riot id",
    "summoner name",
)
_STATIC_FACT_KEYWORDS = (
    "what is",
    "define",
    "base stats of",
    "cooldown of",
    "cost of",
    "how much mana",
    "how much health",
    "base damage of",
)


def classify_request(
    *,
    command: str | None = None,
    text: str = "",
    live_meta_flag: bool = False,
    player_specific_flag: bool = False,
    static_fact_flag: bool = False,
) -> RequestCategory:
    """Classify a request deterministically and cheaply.

    Prefers command type, slash-command name, and explicit flags over
    keyword matching in free text, and never issues an LLM call to classify.
    """
    normalized_command = (command or "").strip().lower()
    normalized_text = text.strip().lower()

    if static_fact_flag or normalized_command in _STATIC_FACT_COMMANDS:
        return RequestCategory.STATIC_FACT
    if player_specific_flag or normalized_command in _PLAYER_COMMANDS:
        return RequestCategory.PLAYER_SPECIFIC
    if live_meta_flag or normalized_command in _LIVE_META_COMMANDS:
        return RequestCategory.LIVE_META

    if any(keyword in normalized_text for keyword in _PLAYER_KEYWORDS):
        return RequestCategory.PLAYER_SPECIFIC
    if any(keyword in normalized_text for keyword in _LIVE_META_KEYWORDS):
        return RequestCategory.LIVE_META
    if any(keyword in normalized_text for keyword in _STATIC_FACT_KEYWORDS):
        return RequestCategory.STATIC_FACT

    return RequestCategory.GENERAL_CHAT


def select_model_route(
    category: RequestCategory,
    config: LLMConfig,
    *,
    force_free_only: bool = False,
    force_general_only: bool = False,
) -> ModelRoute:
    """Map a classified request category to a model route.

    ``force_free_only`` should reflect ``LLM_FREE_ONLY``/
    ``LLM_ROUTING_MODE=free_only`` or an exhausted daily budget.
    ``force_general_only`` should reflect the cheap-budget threshold, which
    routes all eligible traffic through the general route.
    """
    if force_free_only or config.effective_free_only:
        return ModelRoute.FREE
    if force_general_only:
        return ModelRoute.GENERAL
    if category in (RequestCategory.LIVE_META, RequestCategory.PLAYER_SPECIFIC):
        return ModelRoute.RESEARCH
    return ModelRoute.GENERAL


def models_for_route(route: ModelRoute, config: LLMConfig) -> tuple[str, ...]:
    if route is ModelRoute.FREE:
        return (config.free_model,)
    if route is ModelRoute.RESEARCH:
        return config.research_models
    return config.general_models


def max_output_tokens_for_route(route: ModelRoute, config: LLMConfig) -> int:
    if route is ModelRoute.RESEARCH:
        return config.max_output_tokens_research
    return config.max_output_tokens_general


def create_openrouter_request(
    *,
    route: ModelRoute,
    config: LLMConfig,
    messages: Sequence[Mapping[str, str]],
    max_output_tokens: int | None = None,
    stream: bool = True,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build kwargs for ``AsyncOpenAI.chat.completions.create`` on OpenRouter.

    Uses OpenRouter's model-array fallback mechanism (``extra_body.models``)
    instead of a manual retry loop, and keeps provider fallback enabled.
    """
    models = models_for_route(route, config)
    if not models:
        raise RuntimeError(f"no models configured for route {route.value}")

    extra_body: dict[str, Any] = {"models": list(models)}

    if tools:
        # OpenRouter's "Exacto" provider-routing option can improve
        # tool-calling correctness, but its exact request field must be
        # verified against the installed OpenRouter/OpenAI SDK types or
        # current OpenRouter documentation before it is enabled here. We do
        # not guess the field name, so normal provider routing (with
        # fallbacks enabled) is retained.
        # TODO: enable Exacto routing once the field name has been verified.
        pass

    request: dict[str, Any] = {
        "model": models[0],
        "messages": list(messages),
        "max_tokens": max_output_tokens or max_output_tokens_for_route(route, config),
        "stream": stream,
        "extra_body": extra_body,
    }
    if stream:
        request["stream_options"] = {"include_usage": True}
    if tools:
        request["tools"] = list(tools)
    return request


def detect_fallback(
    route: ModelRoute, config: LLMConfig, actual_model: str | None
) -> bool:
    """Return True if OpenRouter appears to have used a fallback model."""
    if not actual_model:
        return False
    models = models_for_route(route, config)
    if not models:
        return False
    return actual_model != models[0]


def record_llm_usage(
    *,
    logger_: logging.Logger,
    route: ModelRoute,
    requested_category: RequestCategory | None,
    actual_model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cost_usd: float | None,
    latency_seconds: float | None,
    cache_status: str,
    fallback_used: bool | None = None,
    error_category: str | None = None,
) -> None:
    """Log observability fields for one OpenRouter response.

    Never pass secrets (API keys, auth headers) or raw prompt content here.
    """
    logger_.info(
        "llm_usage route=%s category=%s model=%s fallback_used=%s "
        "prompt_tokens=%s completion_tokens=%s total_tokens=%s cost_usd=%s "
        "latency_ms=%s cache=%s error=%s",
        route.value,
        requested_category.value if requested_category else "unknown",
        actual_model or "unknown",
        fallback_used,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        f"{cost_usd:.6f}" if cost_usd is not None else None,
        int(latency_seconds * 1000) if latency_seconds is not None else None,
        cache_status,
        error_category or "none",
    )


StaticFactResolver = Callable[[str], Awaitable[str | None]]


async def default_static_fact_resolver(question: str) -> str | None:
    """Placeholder for Data Dragon / local structured-data lookups.

    TODO: integrate a real local data source (e.g. League of Legends Data
    Dragon) so ``static_fact`` requests can be answered without an LLM call.
    Always returns None for now, which defers to the general LLM route.
    """
    return None
