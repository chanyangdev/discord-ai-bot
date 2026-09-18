from decimal import Decimal

import asyncio

import pytest

from llm_routing import (
    DEFAULT_FREE_MODEL,
    DEFAULT_GENERAL_MODELS,
    DEFAULT_RESEARCH_MODELS,
    LLMConfig,
    ModelRoute,
    RequestCategory,
    RoutingMode,
    classify_request,
    create_openrouter_request,
    default_static_fact_resolver,
    detect_fallback,
    load_llm_config,
    max_output_tokens_for_route,
    models_for_route,
    parse_model_list,
    record_llm_usage,
    select_model_route,
)


def make_config(**overrides) -> LLMConfig:
    values = {
        "general_models": DEFAULT_GENERAL_MODELS,
        "research_models": DEFAULT_RESEARCH_MODELS,
        "free_model": DEFAULT_FREE_MODEL,
        "routing_mode": RoutingMode.NORMAL,
        "free_only": False,
        "max_output_tokens_general": 500,
        "max_output_tokens_research": 900,
        "request_timeout_seconds": 45.0,
        "max_retries": 2,
        "http_referer": "",
        "app_name": "Jarvis Discord Bot",
    }
    values.update(overrides)
    return LLMConfig(**values)


# --- Parsing model lists from environment variables ---


def test_parse_model_list_trims_whitespace_and_rejects_empty_ids():
    assert parse_model_list(" a/one , a/two ,a/three") == (
        "a/one",
        "a/two",
        "a/three",
    )
    assert parse_model_list("a/one,,a/two") == ("a/one", "a/two")


def test_parse_model_list_rejects_all_empty():
    with pytest.raises(ValueError):
        parse_model_list("  ,  ,")


def test_parse_model_list_enforces_max():
    with pytest.raises(ValueError):
        parse_model_list("a,b,c,d", max_models=3)


# --- Default model values ---


def test_load_llm_config_defaults_match_spec():
    config = load_llm_config({})

    assert config.general_models == (
        "qwen/qwen3.7-flash",
        "deepseek/deepseek-v4-flash-0731",
        "qwen/qwen3.7-plus",
    )
    assert config.research_models == (
        "google/gemini-3.7-flash",
        "qwen/qwen3.7-plus",
        "z-ai/glm-5.3-flash",
    )
    assert config.free_model == "openrouter/free"
    assert config.routing_mode is RoutingMode.NORMAL
    assert config.free_only is False
    assert config.max_output_tokens_general == 500
    assert config.max_output_tokens_research == 900
    assert config.request_timeout_seconds == 45.0
    assert config.max_retries == 2


def test_load_llm_config_reads_overrides():
    env = {
        "OPENROUTER_GENERAL_MODELS": "custom/a,custom/b",
        "OPENROUTER_RESEARCH_MODELS": "custom/c",
        "OPENROUTER_FREE_MODEL": "custom/free",
        "LLM_ROUTING_MODE": "free_only",
        "LLM_FREE_ONLY": "true",
        "LLM_MAX_OUTPUT_TOKENS_GENERAL": "111",
        "LLM_MAX_OUTPUT_TOKENS_RESEARCH": "222",
        "LLM_REQUEST_TIMEOUT_SECONDS": "10",
        "LLM_MAX_RETRIES": "5",
    }
    config = load_llm_config(env)

    assert config.general_models == ("custom/a", "custom/b")
    assert config.research_models == ("custom/c",)
    assert config.free_model == "custom/free"
    assert config.routing_mode is RoutingMode.FREE_ONLY
    assert config.free_only is True
    assert config.max_output_tokens_general == 111
    assert config.max_output_tokens_research == 222
    assert config.request_timeout_seconds == 10.0
    assert config.max_retries == 5


def test_load_llm_config_supports_deprecated_openrouter_models(caplog):
    with caplog.at_level("WARNING"):
        config = load_llm_config({"OPENROUTER_MODELS": "legacy/a,legacy/b"})

    assert config.general_models == ("legacy/a", "legacy/b")
    assert any("deprecated" in record.message for record in caplog.records)


def test_load_llm_config_supports_deprecated_single_model_override(caplog):
    with caplog.at_level("WARNING"):
        config = load_llm_config({"OPENROUTER_MODEL": "override/model"})

    assert config.general_models[0] == "override/model"
    assert any("deprecated" in record.message for record in caplog.records)


def test_load_llm_config_rejects_invalid_routing_mode():
    with pytest.raises(RuntimeError):
        load_llm_config({"LLM_ROUTING_MODE": "bogus"})


def test_load_llm_config_rejects_non_positive_numeric_settings():
    with pytest.raises(RuntimeError):
        load_llm_config({"LLM_MAX_RETRIES": "0"})
    with pytest.raises(RuntimeError):
        load_llm_config({"LLM_REQUEST_TIMEOUT_SECONDS": "-1"})


def test_load_llm_config_rejects_bad_boolean():
    with pytest.raises(RuntimeError):
        load_llm_config({"LLM_FREE_ONLY": "maybe"})


# --- Classification ---


def test_classify_request_defaults_to_general_chat():
    assert classify_request(text="hello there") == RequestCategory.GENERAL_CHAT


def test_classify_request_uses_keywords_for_live_meta():
    assert (
        classify_request(text="what changed in the current patch notes?")
        == RequestCategory.LIVE_META
    )


def test_classify_request_uses_keywords_for_player_specific():
    assert (
        classify_request(text="can you check my rank please")
        == RequestCategory.PLAYER_SPECIFIC
    )


def test_classify_request_uses_keywords_for_static_fact():
    assert (
        classify_request(text="what is the cooldown of this ability")
        == RequestCategory.STATIC_FACT
    )


def test_classify_request_prefers_explicit_flags_and_commands_over_keywords():
    assert (
        classify_request(command="rank", text="just chatting")
        == RequestCategory.PLAYER_SPECIFIC
    )
    assert (
        classify_request(live_meta_flag=True, text="my rank please")
        == RequestCategory.LIVE_META
    )


# --- General request routing ---


def test_select_model_route_general_chat_uses_general_route():
    config = make_config()
    route = select_model_route(RequestCategory.GENERAL_CHAT, config)
    assert route is ModelRoute.GENERAL
    assert models_for_route(route, config) == config.general_models


def test_select_model_route_static_fact_falls_back_to_general_route():
    config = make_config()
    route = select_model_route(RequestCategory.STATIC_FACT, config)
    assert route is ModelRoute.GENERAL


# --- Live-meta routing ---


def test_select_model_route_live_meta_uses_research_route():
    config = make_config()
    route = select_model_route(RequestCategory.LIVE_META, config)
    assert route is ModelRoute.RESEARCH
    assert models_for_route(route, config) == config.research_models


def test_select_model_route_player_specific_uses_research_route():
    config = make_config()
    route = select_model_route(RequestCategory.PLAYER_SPECIFIC, config)
    assert route is ModelRoute.RESEARCH


# --- Free-only mode ---


def test_select_model_route_free_only_env_forces_free_route():
    config = make_config(free_only=True)
    route = select_model_route(RequestCategory.LIVE_META, config)
    assert route is ModelRoute.FREE
    assert models_for_route(route, config) == (config.free_model,)


def test_select_model_route_routing_mode_free_only_forces_free_route():
    config = make_config(routing_mode=RoutingMode.FREE_ONLY)
    route = select_model_route(RequestCategory.GENERAL_CHAT, config)
    assert route is ModelRoute.FREE


# --- Budget threshold behavior ---


def test_select_model_route_force_free_only_overrides_category():
    config = make_config()
    route = select_model_route(
        RequestCategory.LIVE_META, config, force_free_only=True
    )
    assert route is ModelRoute.FREE


def test_select_model_route_force_general_only_overrides_research_category():
    config = make_config()
    route = select_model_route(
        RequestCategory.LIVE_META, config, force_general_only=True
    )
    assert route is ModelRoute.GENERAL


def test_max_output_tokens_for_route_differs_by_route():
    config = make_config()
    assert max_output_tokens_for_route(ModelRoute.GENERAL, config) == 500
    assert max_output_tokens_for_route(ModelRoute.RESEARCH, config) == 900


# --- Fallback request payload ordering ---


def test_create_openrouter_request_orders_models_and_keeps_fallback_enabled():
    config = make_config()
    request = create_openrouter_request(
        route=ModelRoute.GENERAL,
        config=config,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert request["model"] == config.general_models[0]
    assert request["extra_body"]["models"] == list(config.general_models)
    assert request["max_tokens"] == config.max_output_tokens_general
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}


def test_create_openrouter_request_research_route_uses_research_models():
    config = make_config()
    request = create_openrouter_request(
        route=ModelRoute.RESEARCH,
        config=config,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert request["model"] == config.research_models[0]
    assert request["extra_body"]["models"] == list(config.research_models)
    assert request["max_tokens"] == config.max_output_tokens_research


def test_detect_fallback_true_when_actual_model_differs():
    config = make_config()
    route = ModelRoute.GENERAL
    assert detect_fallback(route, config, config.general_models[1]) is True
    assert detect_fallback(route, config, config.general_models[0]) is False
    assert detect_fallback(route, config, None) is False


# --- Static facts bypassing the LLM when local data is available ---


def test_default_static_fact_resolver_defers_to_llm():
    assert asyncio.run(default_static_fact_resolver("what is x")) is None


# --- No secret values appearing in logs ---


def test_record_llm_usage_never_logs_secrets(caplog):
    with caplog.at_level("INFO"):
        record_llm_usage(
            logger_=__import__("logging").getLogger("test.llm_usage"),
            route=ModelRoute.GENERAL,
            requested_category=RequestCategory.GENERAL_CHAT,
            actual_model="qwen/qwen3.7-flash",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
            latency_seconds=0.5,
            cache_status="miss",
            fallback_used=False,
        )

    output = "\n".join(record.message for record in caplog.records)
    assert "sk-" not in output
    assert "Authorization" not in output
    assert "Bearer" not in output
    assert "qwen/qwen3.7-flash" in output
