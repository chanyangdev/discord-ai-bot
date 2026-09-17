import hashlib
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from storage import connect_sqlite


RESPONSE_CACHE_KEY_VERSION = "v1"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedResponse:
    answer: str
    sources: list[Any]
    answer_type: str
    patch_version: str


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