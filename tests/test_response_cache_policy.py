import pytest

from response_cache import AnswerType, CachePolicyDecision, decide_cache_policy


@pytest.mark.parametrize(
    ("answer_type", "ttl_seconds"),
    [
        (AnswerType.STATIC_FACT, 700),
        (AnswerType.BUILD_META, 70),
        (AnswerType.PATCH_SUMMARY, 7),
    ],
)
def test_allowed_answer_types_are_cacheable(answer_type, ttl_seconds):
    decision = decide_cache_policy(
        answer_type,
        static_ttl_seconds=700,
        meta_ttl_seconds=70,
        patch_notes_ttl_seconds=7,
    )

    assert decision == CachePolicyDecision(
        True, answer_type.value, ttl_seconds, None
    )


@pytest.mark.parametrize(
    ("answer_type", "skip_reason"),
    [
        (AnswerType.PLAYER_SPECIFIC, "player_specific"),
        (AnswerType.ACCOUNT_DATA, "not_cacheable"),
        (AnswerType.PRIVATE_CONTENT, "not_cacheable"),
        (AnswerType.ADMIN_COMMAND, "not_cacheable"),
        (AnswerType.MODERATION_OUTCOME, "not_cacheable"),
        (AnswerType.ERROR, "not_cacheable"),
        (AnswerType.REFUSAL, "not_cacheable"),
        (AnswerType.UNKNOWN, "not_cacheable"),
        ("future_classification", "not_cacheable"),
    ],
)
def test_denied_answer_types_are_not_cacheable(answer_type, skip_reason):
    decision = decide_cache_policy(answer_type)

    assert decision.cacheable is False
    assert decision.ttl_seconds is None
    assert decision.skip_reason == skip_reason


@pytest.mark.parametrize(
    ("kwargs", "skip_reason"),
    [
        ({"successful": False}, "unsuccessful"),
        ({"depends_on_conversation_history": True}, "conversation_history"),
        ({"player_specific": True}, "player_specific"),
        ({"safe_to_share": False}, "unsafe_to_share"),
    ],
)
def test_safety_overrides_deny_allowed_classifications(kwargs, skip_reason):
    decision = decide_cache_policy(AnswerType.STATIC_FACT, **kwargs)

    assert decision.cacheable is False
    assert decision.answer_type == AnswerType.STATIC_FACT.value
    assert decision.ttl_seconds is None
    assert decision.skip_reason == skip_reason