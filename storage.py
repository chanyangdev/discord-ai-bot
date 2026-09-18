import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite


SQLITE_BUSY_TIMEOUT_SECONDS = 5.0


def connect_sqlite(db_path: str | os.PathLike[str]) -> aiosqlite.Connection:
    return aiosqlite.connect(db_path, timeout=SQLITE_BUSY_TIMEOUT_SECONDS)


class QuotaExceeded(RuntimeError):
    """Raised when a user exceeds the per-day quota."""


@dataclass(frozen=True)
class DailyUsage:
    user_id: int
    usage_date: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reserved_tokens: int = 0
    successful_requests: int = 0
    updated_at: int = 0


@dataclass(frozen=True)
class GlobalCostUsage:
    usage_date: str
    committed_microdollars: int = 0
    reserved_microdollars: int = 0
    successful_paid_requests: int = 0
    updated_at: int = 0


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _utc_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp())


async def _prepare_connection(connection: aiosqlite.Connection) -> None:
    await connection.execute(
        f"PRAGMA busy_timeout = {int(SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}"
    )


class TokenUsageStore:
    def __init__(self, db_path: str | os.PathLike[str], daily_limit: int) -> None:
        self.db_path = str(Path(db_path))
        self.daily_limit = int(daily_limit)
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        async with connect_sqlite(self.db_path) as connection:
            await _prepare_connection(connection)
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    user_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    utc_date TEXT NOT NULL,
                    used_tokens INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, guild_id, utc_date)
                )
                """
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_token_usage_lookup "
                "ON token_usage (user_id, guild_id, utc_date)"
            )
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_user_token_usage (
                    user_id INTEGER NOT NULL,
                    usage_date TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0
                        CHECK (prompt_tokens >= 0),
                    completion_tokens INTEGER NOT NULL DEFAULT 0
                        CHECK (completion_tokens >= 0),
                    reserved_tokens INTEGER NOT NULL DEFAULT 0
                        CHECK (reserved_tokens >= 0),
                    successful_requests INTEGER NOT NULL DEFAULT 0
                        CHECK (successful_requests >= 0),
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (user_id, usage_date)
                )
                """
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_daily_user_token_usage_date "
                "ON daily_user_token_usage (usage_date)"
            )
            await connection.execute(
                """
                INSERT INTO daily_user_token_usage (
                    user_id, usage_date, prompt_tokens, updated_at
                )
                SELECT user_id, utc_date, SUM(used_tokens), CAST(strftime('%s', 'now') AS INTEGER)
                FROM token_usage
                GROUP BY user_id, utc_date
                ON CONFLICT(user_id, usage_date) DO NOTHING
                """
            )
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_token_quota_overrides (
                    user_id INTEGER PRIMARY KEY,
                    daily_limit INTEGER NOT NULL CHECK (daily_limit > 0)
                )
                """
            )
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    patch_version TEXT NOT NULL,
                    question_hash TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    answer_type TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    last_accessed_at INTEGER NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                        CHECK (hit_count >= 0)
                )
                """
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_response_cache_expires_at "
                "ON response_cache (expires_at)"
            )
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS global_provider_cost_usage (
                    usage_date TEXT PRIMARY KEY,
                    committed_microdollars INTEGER NOT NULL DEFAULT 0
                        CHECK (committed_microdollars >= 0),
                    reserved_microdollars INTEGER NOT NULL DEFAULT 0
                        CHECK (reserved_microdollars >= 0),
                    successful_paid_requests INTEGER NOT NULL DEFAULT 0
                        CHECK (successful_paid_requests >= 0),
                    updated_at INTEGER NOT NULL
                )
                """
            )
            await connection.execute(
                "UPDATE daily_user_token_usage SET reserved_tokens = 0 "
                "WHERE reserved_tokens > 0"
            )
            await connection.execute(
                """
                UPDATE global_provider_cost_usage
                SET committed_microdollars =
                        committed_microdollars + reserved_microdollars,
                    reserved_microdollars = 0,
                    updated_at = ?
                WHERE reserved_microdollars > 0
                """,
                (_utc_timestamp(),),
            )
            await connection.commit()
        self._initialized = True

    async def close(self) -> None:
        return None

    async def get_daily_usage(self, user_id: int) -> DailyUsage:
        _validate_non_negative_int("user_id", user_id)
        usage_date = _utc_date()
        async with connect_sqlite(self.db_path) as connection:
            await _prepare_connection(connection)
            cursor = await connection.execute(
                """
                SELECT prompt_tokens, completion_tokens, reserved_tokens,
                       successful_requests, updated_at
                FROM daily_user_token_usage
                WHERE user_id = ? AND usage_date = ?
                """,
                (user_id, usage_date),
            )
            row = await cursor.fetchone()
        if row is None:
            return DailyUsage(user_id=user_id, usage_date=usage_date)
        return DailyUsage(user_id, usage_date, *map(int, row))

    async def get_quota_override(self, user_id: int) -> int | None:
        _validate_non_negative_int("user_id", user_id)
        async with connect_sqlite(self.db_path) as connection:
            await _prepare_connection(connection)
            cursor = await connection.execute(
                "SELECT daily_limit FROM user_token_quota_overrides WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
        return None if row is None else int(row[0])

    async def get_global_cost_usage(self) -> GlobalCostUsage:
        usage_date = _utc_date()
        async with connect_sqlite(self.db_path) as connection:
            await _prepare_connection(connection)
            await connection.execute(
                """
                INSERT INTO global_provider_cost_usage (usage_date, updated_at)
                VALUES (?, ?)
                ON CONFLICT(usage_date) DO NOTHING
                """,
                (usage_date, _utc_timestamp()),
            )
            await connection.commit()
            cursor = await connection.execute(
                """
                SELECT committed_microdollars, reserved_microdollars,
                       successful_paid_requests, updated_at
                FROM global_provider_cost_usage
                WHERE usage_date = ?
                """,
                (usage_date,),
            )
            row = await cursor.fetchone()
        return GlobalCostUsage(usage_date, *map(int, row))

    async def reserve_global_cost(
        self,
        amount_microdollars: int,
        daily_budget_microdollars: int,
    ) -> int:
        amount_microdollars = _validate_positive_int(
            "amount_microdollars", amount_microdollars
        )
        daily_budget_microdollars = _validate_positive_int(
            "daily_budget_microdollars", daily_budget_microdollars
        )
        usage_date = _utc_date()
        async with connect_sqlite(self.db_path) as connection:
            try:
                await _prepare_connection(connection)
                await connection.execute("BEGIN IMMEDIATE")
                timestamp = _utc_timestamp()
                await connection.execute(
                    """
                    INSERT INTO global_provider_cost_usage (usage_date, updated_at)
                    VALUES (?, ?)
                    ON CONFLICT(usage_date) DO NOTHING
                    """,
                    (usage_date, timestamp),
                )
                cursor = await connection.execute(
                    """
                    UPDATE global_provider_cost_usage
                    SET reserved_microdollars = reserved_microdollars + ?,
                        updated_at = ?
                    WHERE usage_date = ?
                      AND committed_microdollars + reserved_microdollars + ? <= ?
                    """,
                    (
                        amount_microdollars,
                        timestamp,
                        usage_date,
                        amount_microdollars,
                        daily_budget_microdollars,
                    ),
                )
                if cursor.rowcount != 1:
                    raise QuotaExceeded("Global daily AI budget reached.")
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return amount_microdollars

    async def reconcile_global_cost(
        self,
        reserved_microdollars: int,
        actual_microdollars: int | None,
        *,
        successful: bool,
    ) -> int:
        reserved_microdollars = _validate_non_negative_int(
            "reserved_microdollars", reserved_microdollars
        )
        if actual_microdollars is not None:
            actual_microdollars = _validate_non_negative_int(
                "actual_microdollars", actual_microdollars
            )
        committed = (
            actual_microdollars
            if actual_microdollars is not None
            else reserved_microdollars
        )
        usage_date = _utc_date()
        async with connect_sqlite(self.db_path) as connection:
            try:
                await _prepare_connection(connection)
                await connection.execute("BEGIN IMMEDIATE")
                cursor = await connection.execute(
                    """
                    UPDATE global_provider_cost_usage
                    SET committed_microdollars = committed_microdollars + ?,
                        reserved_microdollars = reserved_microdollars - ?,
                        successful_paid_requests = successful_paid_requests + ?,
                        updated_at = ?
                    WHERE usage_date = ? AND reserved_microdollars >= ?
                    """,
                    (
                        committed,
                        reserved_microdollars,
                        1 if successful else 0,
                        _utc_timestamp(),
                        usage_date,
                        reserved_microdollars,
                    ),
                )
                if cursor.rowcount != 1:
                    raise LookupError("Expected global cost row is missing")
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return committed

    async def release_global_cost(self, reserved_microdollars: int) -> int:
        reserved_microdollars = _validate_non_negative_int(
            "reserved_microdollars", reserved_microdollars
        )
        usage_date = _utc_date()
        async with connect_sqlite(self.db_path) as connection:
            try:
                await _prepare_connection(connection)
                await connection.execute("BEGIN IMMEDIATE")
                cursor = await connection.execute(
                    """
                    UPDATE global_provider_cost_usage
                    SET reserved_microdollars = reserved_microdollars - ?,
                        updated_at = ?
                    WHERE usage_date = ? AND reserved_microdollars >= ?
                    """,
                    (
                        reserved_microdollars,
                        _utc_timestamp(),
                        usage_date,
                        reserved_microdollars,
                    ),
                )
                if cursor.rowcount != 1:
                    raise LookupError("Expected global cost row is missing")
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return reserved_microdollars

    async def reserve_tokens(
        self,
        user_id: int,
        estimated_tokens: int,
        daily_limit: int,
        *legacy_args: object,
    ) -> int:
        if legacy_args:
            estimated_tokens = int(legacy_args[-1])
            daily_limit = self.daily_limit
        _validate_non_negative_int("user_id", user_id)
        estimated_tokens = _validate_positive_int("estimated_tokens", estimated_tokens)
        daily_limit = _validate_positive_int("daily_limit", daily_limit)
        usage_date = _utc_date()
        async with connect_sqlite(self.db_path) as connection:
            try:
                await _prepare_connection(connection)
                await connection.execute("BEGIN IMMEDIATE")
                cursor = await connection.execute(
                    "SELECT daily_limit FROM user_token_quota_overrides WHERE user_id = ?",
                    (user_id,),
                )
                override_row = await cursor.fetchone()
                effective_limit = (
                    int(override_row[0]) if override_row is not None else daily_limit
                )
                timestamp = _utc_timestamp()
                await connection.execute(
                    """
                    INSERT INTO daily_user_token_usage (user_id, usage_date, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id, usage_date) DO NOTHING
                    """,
                    (user_id, usage_date, timestamp),
                )
                cursor = await connection.execute(
                    """
                    UPDATE daily_user_token_usage
                    SET reserved_tokens = reserved_tokens + ?, updated_at = ?
                    WHERE user_id = ? AND usage_date = ?
                      AND prompt_tokens + completion_tokens + reserved_tokens + ? <= ?
                    """,
                    (
                        estimated_tokens,
                        timestamp,
                        user_id,
                        usage_date,
                        estimated_tokens,
                        effective_limit,
                    ),
                )
                if cursor.rowcount != 1:
                    raise QuotaExceeded(f"Daily token limit reached for user {user_id}.")
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return estimated_tokens

    async def commit_tokens(
        self,
        user_id: int,
        reserved_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> int:
        _validate_non_negative_int("user_id", user_id)
        reserved_tokens = _validate_non_negative_int("reserved_tokens", reserved_tokens)
        prompt_tokens = _validate_non_negative_int("prompt_tokens", prompt_tokens)
        completion_tokens = _validate_non_negative_int("completion_tokens", completion_tokens)
        usage_date = _utc_date()
        async with connect_sqlite(self.db_path) as connection:
            try:
                await _prepare_connection(connection)
                await connection.execute("BEGIN IMMEDIATE")
                cursor = await connection.execute(
                    """
                    UPDATE daily_user_token_usage
                    SET prompt_tokens = prompt_tokens + ?,
                        completion_tokens = completion_tokens + ?,
                        reserved_tokens = reserved_tokens - ?,
                        successful_requests = successful_requests + 1,
                        updated_at = ?
                    WHERE user_id = ? AND usage_date = ?
                      AND reserved_tokens >= ?
                    """,
                    (
                        prompt_tokens,
                        completion_tokens,
                        reserved_tokens,
                        _utc_timestamp(),
                        user_id,
                        usage_date,
                        reserved_tokens,
                    ),
                )
                if cursor.rowcount != 1:
                    raise LookupError("Expected daily token usage row is missing")
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return prompt_tokens + completion_tokens

    async def release_tokens(
        self,
        user_id: int,
        reserved_tokens: int,
        *legacy_args: object,
    ) -> int:
        if len(legacy_args) == 2:
            reserved_tokens = int(legacy_args[-1])
        _validate_non_negative_int("user_id", user_id)
        reserved_tokens = _validate_non_negative_int("reserved_tokens", reserved_tokens)
        usage_date = _utc_date()
        async with connect_sqlite(self.db_path) as connection:
            try:
                await _prepare_connection(connection)
                await connection.execute("BEGIN IMMEDIATE")
                cursor = await connection.execute(
                    """
                    UPDATE daily_user_token_usage
                    SET reserved_tokens = reserved_tokens - ?, updated_at = ?
                    WHERE user_id = ? AND usage_date = ? AND reserved_tokens >= ?
                    """,
                    (
                        reserved_tokens,
                        _utc_timestamp(),
                        user_id,
                        usage_date,
                        reserved_tokens,
                    ),
                )
                if cursor.rowcount != 1:
                    raise LookupError("Expected daily token usage row is missing")
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return reserved_tokens

    async def usage_for_day(self, user_id: int, guild_id: int, utc_date: str) -> int:
        async with connect_sqlite(self.db_path) as connection:
            await _prepare_connection(connection)
            cursor = await connection.execute(
                """
                SELECT prompt_tokens + completion_tokens + reserved_tokens
                FROM daily_user_token_usage WHERE user_id = ? AND usage_date = ?
                """,
                (user_id, utc_date),
            )
            row = await cursor.fetchone()
        return 0 if row is None else int(row[0])

    async def apply_usage(self, user_id: int, guild_id: int, utc_date: str,
                          reserved_tokens: int, prompt_tokens: int,
                          completion_tokens: int) -> int:
        if utc_date != _utc_date():
            raise ValueError("usage_date must be the current UTC date")
        return await self.commit_tokens(
            user_id, reserved_tokens, prompt_tokens, completion_tokens
        )

    async def __aenter__(self) -> "TokenUsageStore":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
