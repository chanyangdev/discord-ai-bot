import pytest

from persona import (
    Formality,
    HumorLevel,
    Persona,
    Verbosity,
    build_system_prompt,
    load_persona_config,
)


def test_persona_defaults_match_intended_values():
    config = load_persona_config({})

    assert config.persona is Persona.JARVIS
    assert config.owner_honorific == ""
    assert config.formality is Formality.BALANCED
    assert config.humor_level is HumorLevel.LOW
    assert config.default_verbosity is Verbosity.CONCISE
    assert config.use_marvel_references is False
    assert config.require_patch_for_meta is True


def test_persona_reads_overrides_from_env():
    env = {
        "BOT_PERSONA": "jarvis",
        "JARVIS_OWNER_HONORIFIC": "sir",
        "JARVIS_FORMALITY": "formal",
        "JARVIS_HUMOR_LEVEL": "medium",
        "JARVIS_DEFAULT_VERBOSITY": "detailed",
        "JARVIS_USE_MARVEL_REFERENCES": "true",
        "JARVIS_REQUIRE_PATCH_FOR_META": "false",
    }
    config = load_persona_config(env)

    assert config.owner_honorific == "sir"
    assert config.formality is Formality.FORMAL
    assert config.humor_level is HumorLevel.MEDIUM
    assert config.default_verbosity is Verbosity.DETAILED
    assert config.use_marvel_references is True
    assert config.require_patch_for_meta is False


def test_persona_rejects_invalid_choice_values():
    with pytest.raises(RuntimeError, match="JARVIS_FORMALITY"):
        load_persona_config({"JARVIS_FORMALITY": "sassy"})
    with pytest.raises(RuntimeError, match="JARVIS_HUMOR_LEVEL"):
        load_persona_config({"JARVIS_HUMOR_LEVEL": "extreme"})
    with pytest.raises(RuntimeError, match="JARVIS_DEFAULT_VERBOSITY"):
        load_persona_config({"JARVIS_DEFAULT_VERBOSITY": "novel"})
    with pytest.raises(RuntimeError, match="BOT_PERSONA"):
        load_persona_config({"BOT_PERSONA": "ultron"})


def test_persona_rejects_invalid_boolean():
    with pytest.raises(RuntimeError):
        load_persona_config({"JARVIS_USE_MARVEL_REFERENCES": "maybe"})


def test_build_system_prompt_jarvis_includes_honorific_when_set():
    config = load_persona_config({"JARVIS_OWNER_HONORIFIC": "sir"})
    prompt = build_system_prompt(config)
    assert 'as "sir"' in prompt
    assert "J.A.R.V.I.S." in prompt


def test_build_system_prompt_jarvis_omits_honorific_when_unset():
    config = load_persona_config({})
    prompt = build_system_prompt(config)
    assert "no owner honorific is configured" in prompt


def test_build_system_prompt_respects_marvel_reference_toggle():
    off_prompt = build_system_prompt(load_persona_config({}))
    on_prompt = build_system_prompt(
        load_persona_config({"JARVIS_USE_MARVEL_REFERENCES": "true"})
    )
    assert "Do not make Stark/Avengers/suit references" in off_prompt
    assert "Stark-adjacent inspiration are" in on_prompt


def test_build_system_prompt_respects_patch_requirement_toggle():
    strict_prompt = build_system_prompt(load_persona_config({}))
    relaxed_prompt = build_system_prompt(
        load_persona_config({"JARVIS_REQUIRE_PATCH_FOR_META": "false"})
    )
    assert "never on model memory alone" in strict_prompt
    assert "never on model memory alone" not in relaxed_prompt


def test_build_system_prompt_plain_persona_drops_jarvis_identity():
    config = load_persona_config({"BOT_PERSONA": "plain"})
    prompt = build_system_prompt(config)
    assert "J.A.R.V.I.S." not in prompt
    assert "friendly, practical AI assistant" in prompt


def test_build_system_prompt_always_includes_safety_rules():
    for persona in ("jarvis", "plain"):
        prompt = build_system_prompt(load_persona_config({"BOT_PERSONA": persona}))
        assert "NEVER-GUESS-META RULE" in prompt
        assert "BOUNDARIES" in prompt
        assert "Never reveal or reproduce API keys" in prompt


def test_build_system_prompt_never_claims_to_be_the_literal_character():
    prompt = build_system_prompt(load_persona_config({}))
    assert "You are not Tony Stark, Vision, Ultron" in prompt
