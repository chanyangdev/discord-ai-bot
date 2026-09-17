import hashlib

import pytest

from response_cache import build_cache_key, normalize_question, question_hash


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("  What   Is   This?  ", "what is this?"),
        ("Straße", "strasse"),
        ("ＡＢＣ　１２３", "abc 123"),
    ],
)
def test_normalize_question(question, expected):
    assert normalize_question(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What is this?",
            hashlib.sha256("what is this?".encode("utf-8")).hexdigest(),
        ),
        (
            "  STRASSE  ",
            hashlib.sha256("strasse".encode("utf-8")).hexdigest(),
        ),
    ],
)
def test_question_hashes_normalized_question(question, expected):
    assert question_hash(question) == expected


def test_punctuation_is_preserved():
    assert question_hash("What is this?") != question_hash("What is this")


def test_cache_key_is_deterministic_and_patch_specific():
    question = "  What   is this?  "
    first = build_cache_key(question, "14.1", key_version="v1")
    second = build_cache_key(question, "14.1", key_version="v1")
    other_patch = build_cache_key(question, "14.2", key_version="v1")

    assert first == second
    assert first != other_patch


@pytest.mark.parametrize(
    ("question", "patch_version"),
    [("", "14.1"), ("   \n\t", "14.1"), ("question", "")],
)
def test_empty_inputs_are_rejected(question, patch_version):
    with pytest.raises(ValueError):
        build_cache_key(question, patch_version)