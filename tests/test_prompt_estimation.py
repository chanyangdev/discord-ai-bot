from prompt_estimation import (
    estimate_complete_prompt_tokens,
    estimate_reservation_tokens,
)


def test_history_increases_complete_prompt_estimate():
    base = estimate_complete_prompt_tokens(
        system_text="system",
        contents=[{"role": "user", "parts": [{"text": "question"}]}],
    )
    with_history = estimate_complete_prompt_tokens(
        system_text="system",
        contents=[
            {"role": "user", "parts": [{"text": "previous question"}]},
            {"role": "model", "parts": [{"text": "previous answer"}]},
            {"role": "user", "parts": [{"text": "question"}]},
        ],
    )
    assert with_history > base


def test_retrieved_context_and_tools_increase_estimate():
    base = estimate_complete_prompt_tokens(
        system_text="system",
        contents=[],
    )
    enriched = estimate_complete_prompt_tokens(
        system_text="system",
        contents=[],
        retrieved_context=[{"text": "retrieved context"}],
        tool_definitions=[{"name": "lookup", "description": "tool"}],
    )
    assert enriched > base


def test_max_output_tokens_are_always_included():
    prompt = estimate_complete_prompt_tokens(
        system_text="system",
        contents=[],
    )
    reservation = estimate_reservation_tokens(
        system_text="system",
        contents=[],
        max_output_tokens=1024,
    )
    assert reservation == prompt + 1024


def test_invalid_max_output_tokens_are_rejected():
    try:
        estimate_reservation_tokens(
            system_text="system",
            contents=[],
            max_output_tokens=0,
        )
    except ValueError as error:
        assert "max_output_tokens" in str(error)
    else:
        raise AssertionError("Expected invalid max output tokens to fail")