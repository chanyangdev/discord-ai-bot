import hashlib
import re
import unicodedata


RESPONSE_CACHE_KEY_VERSION = "v1"


def normalize_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def question_hash(question: str) -> str:
    normalized_question = normalize_question(question)
    if not normalized_question:
        raise ValueError("question must not be empty after normalization")

    return hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()


def build_cache_key(
    question: str,
    patch_version: str,
    key_version: str = RESPONSE_CACHE_KEY_VERSION,
) -> str:
    normalized_question = normalize_question(question)
    if not normalized_question:
        raise ValueError("question must not be empty after normalization")
    if not patch_version:
        raise ValueError("patch_version must not be empty")

    question_digest = hashlib.sha256(
        normalized_question.encode("utf-8")
    ).hexdigest()
    key_material = "\0".join((key_version, patch_version, question_digest))
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()