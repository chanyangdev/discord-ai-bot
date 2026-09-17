import hashlib
import json
import logging
import re
import time
import unicodedata
from asyncio import Lock
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from storage import connect_sqlite


RESPONSE_CACHE_KEY_VERSION = "v1"
RESPONSE_CACHE_STATIC_TTL_SECONDS = 604800
RESPONSE_CACHE_META_TTL_SECONDS = 21600
RESPONSE_CACHE_PATCH_NOTES_TTL_SECONDS = 86400
PATCH_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+$")
logger = logging.getLogger(__name__)


class AnswerType(str, Enum):
    STATIC_FACT = "static_fact"
    BUILD_META = "build_meta"
    PATCH_SUMMARY = "patch_summary"
    PLAYER_SPECIFIC = "player_specific"
    ACCOUNT_DATA = "account_data"
    PRIVATE_CONTENT = "private_content"
    ADMIN_COMMAND = "admin_command"
    MODERATION_OUTCOME = "moderation_outcome"
    ERROR = "error"
    REFUSAL = "refusal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CachePolicyDecision:
    cacheable: bool
    answer_type: str
    ttl_seconds: int | None
    skip_reason: str | None


@dataclass(frozen=True)
class CachedResponse:
    answer: str
    sources: list[Any]
    answer_type: str
    patch_version: str


def parse_canonical_patch_version(patch_version: str) -> str:
    canonical = patch_version.strip()
    if not PATCH_VERSION_PATTERN.fullmatch(canonical):
        raise ValueError("patch version must use the exact major.minor format")
    return canonical


@dataclass
class _KeyLockEntry:
    lock: Lock
    references: int = 0


def decide_cache_policy(
    answer_type: AnswerType | str,
    *,
    successful: bool = True,
    depends_on_conversation_history: bool = False,
    player_specific: bool = False,
    safe_to_share: bool = True,
    static_ttl_seconds: int = RESPONSE_CACHE_STATIC_TTL_SECONDS,
    meta_ttl_seconds: int = RESPONSE_CACHE_META_TTL_SECONDS,
    patch_notes_ttl_seconds: int = RESPONSE_CACHE_PATCH_NOTES_TTL_SECONDS,
) -> CachePolicyDecision:
    try:
        classification = AnswerType(answer_type)
    except ValueError:
        classification = AnswerType.UNKNOWN

    if not successful:
        return CachePolicyDecision(False, classification.value, None, "unsuccessful")
    if depends_on_conversation_history:
        return CachePolicyDecision(
            False, classification.value, None, "conversation_history"
        )
    if player_specific or classification is AnswerType.PLAYER_SPECIFIC:
        return CachePolicyDecision(False, classification.value, None, "player_specific")
    if not safe_to_share:
        return CachePolicyDecision(False, classification.value, None, "unsafe_to_share")

    ttl_by_type = {
        AnswerType.STATIC_FACT: static_ttl_seconds,
        AnswerType.BUILD_META: meta_ttl_seconds,
        AnswerType.PATCH_SUMMARY: patch_notes_ttl_seconds,
    }
    ttl_seconds = ttl_by_type.get(classification)
    if ttl_seconds is None:
        return CachePolicyDecision(False, classification.value, None, "not_cacheable")
    return CachePolicyDecision(True, classification.value, ttl_seconds, None)


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


class ResponseCache:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # This registry coordinates requests within one bot process only.
        self._key_locks: dict[str, _KeyLockEntry] = {}
        self._key_locks_guard = Lock()

    @asynccontextmanager
    async def _lock_for_key(self, cache_key: str):
        async with self._key_locks_guard:
            entry = self._key_locks.get(cache_key)
            if entry is None:
                entry = _KeyLockEntry(lock=Lock())
                self._key_locks[cache_key] = entry
            entry.references += 1

        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            async with self._key_locks_guard:
                entry.references -= 1
                if entry.references == 0:
                    self._key_locks.pop(cache_key, None)

    async def get_or_load(
        self,
        cache_key: str,
        loader: Callable[[], Awaitable[CachedResponse | None]],
        *,
        cacheable: bool = True,
    ) -> CachedResponse | None:
        cached = await self.get(cache_key) if cacheable else None
        if cached is not None or not cacheable:
            return cached if cached is not None else await loader()

        async with self._lock_for_key(cache_key):
            cached = await self.get(cache_key)
            if cached is not None:
                return cached

            loaded = await loader()
            if loaded is None:
                return None

            cached = await self.get(cache_key)
            return cached if cached is not None else loaded

    async def get(self, cache_key: str) -> CachedResponse | None:
        now = int(time.time())

        async with connect_sqlite(self.db_path) as connection:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                """
                SELECT answer, sources_json, answer_type, patch_version
                FROM response_cache
                WHERE cache_key = ? AND expires_at > ?
                """,
                (cache_key, now),
            )
            row = await cursor.fetchone()
            if row is None:
                await connection.rollback()
                return None

            try:
                sources = json.loads(row[1])
            except (TypeError, json.JSONDecodeError):
                logger.warning("response cache malformed row: invalid sources JSON")
                await connection.rollback()
                return None

            if not isinstance(sources, list):
                logger.warning("response cache malformed row: sources is not a list")
                await connection.rollback()
                return None

            await connection.execute(
                """
                UPDATE response_cache
                SET hit_count = hit_count + 1, last_accessed_at = ?
                WHERE cache_key = ? AND expires_at > ?
                """,
                (now, cache_key, now),
            )
            await connection.commit()

        return CachedResponse(
            answer=row[0],
            sources=sources,
            answer_type=row[2],
            patch_version=row[3],
        )

    async def put(
        self,
        cache_key: str,
        patch_version: str,
        question_hash: str,
        answer: str,
        sources: list[Any],
        answer_type: str,
        ttl_seconds: int,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not isinstance(sources, list):
            raise ValueError("sources must be a list")

        created_at = int(time.time())
        expires_at = created_at + ttl_seconds
        sources_json = json.dumps(sources, ensure_ascii=False)

        async with connect_sqlite(self.db_path) as connection:
            await connection.execute(
                """
                INSERT INTO response_cache (
                    cache_key, patch_version, question_hash, answer,
                    sources_json, answer_type, created_at, expires_at,
                    last_accessed_at, hit_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    patch_version = excluded.patch_version,
                    question_hash = excluded.question_hash,
                    answer = excluded.answer,
                    sources_json = excluded.sources_json,
                    answer_type = excluded.answer_type,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    last_accessed_at = excluded.last_accessed_at,
                    hit_count = 0
                """,
                (
                    cache_key,
                    patch_version,
                    question_hash,
                    answer,
                    sources_json,
                    answer_type,
                    created_at,
                    expires_at,
                    created_at,
                ),
            )
            await connection.commit()

    async def delete_expired(self) -> int:
        async with connect_sqlite(self.db_path) as connection:
            cursor = await connection.execute(
                "DELETE FROM response_cache WHERE expires_at <= ?",
                (int(time.time()),),
            )
            await connection.commit()
            return cursor.rowcount

    async def invalidate_patch(self, patch_version: str) -> int:
        async with connect_sqlite(self.db_path) as connection:
            cursor = await connection.execute(
                "DELETE FROM response_cache WHERE patch_version = ?",
                (patch_version,),
            )
            await connection.commit()
            return cursor.rowcount