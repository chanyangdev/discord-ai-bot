import os
from pathlib import Path
from typing import Any

import aiosqlite


class QuotaExceeded(RuntimeError):
    """Raised when a user exceeds the per-day quota."""


class TokenUsageStore:
    def __init__(self, db_path: str | os.PathLike[str], daily_limit: int) -> None:
        self.db_path = str(Path(db_path))
        self.daily_limit = int(daily_limit)

    async def initialize(self) -> None:
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as connection:
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
            await connection.commit()

    async def close(self) -> None:
        return None

    async def reserve_tokens(
        self,
        user_id: int,
        guild_id: int,
        utc_date: str,
        amount: int,
    ) -> int:
        if amount <= 0:
            return 0

        async with aiosqlite.connect(self.db_path) as connection:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                """
                INSERT INTO token_usage (user_id, guild_id, utc_date, used_tokens)
                VALUES (?, ?, ?, 0)
                ON CONFLICT(user_id, guild_id, utc_date) DO NOTHING
                """,
                (user_id, guild_id, utc_date),
            )
            cursor = await connection.execute(
                """
                UPDATE token_usage
                SET used_tokens = used_tokens + ?
                WHERE user_id = ?
                  AND guild_id = ?
                  AND utc_date = ?
                  AND used_tokens + ? <= ?
                """,
                (amount, user_id, guild_id, utc_date, amount, self.daily_limit),
            )
            if cursor.rowcount == 0:
                await connection.rollback()
                raise QuotaExceeded(
                    f"Daily token limit reached for user {user_id} in guild {guild_id}."
                )
            await connection.commit()

        return amount

    async def release_tokens(
        self,
        user_id: int,
        guild_id: int,
        utc_date: str,
        amount: int,
    ) -> int:
        if amount <= 0:
            return 0

        async with aiosqlite.connect(self.db_path) as connection:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                """
                UPDATE token_usage
                SET used_tokens = CASE
                    WHEN used_tokens >= ? THEN used_tokens - ?
                    ELSE 0
                END
                WHERE user_id = ?
                  AND guild_id = ?
                  AND utc_date = ?
                """,
                (amount, amount, user_id, guild_id, utc_date),
            )
            await connection.commit()

        return amount

    async def apply_usage(
        self,
        user_id: int,
        guild_id: int,
        utc_date: str,
        reserved_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> int:
        actual_total = max(0, int(prompt_tokens) + int(completion_tokens))

        async with aiosqlite.connect(self.db_path) as connection:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                """
                SELECT used_tokens
                FROM token_usage
                WHERE user_id = ?
                  AND guild_id = ?
                  AND utc_date = ?
                """,
                (user_id, guild_id, utc_date),
            )
            row = await cursor.fetchone()

            if row is None:
                await connection.execute(
                    """
                    INSERT INTO token_usage (user_id, guild_id, utc_date, used_tokens)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, guild_id, utc_date, actual_total),
                )
            else:
                current_total = int(row[0])
                new_total = max(0, current_total - int(reserved_tokens) + actual_total)
                await connection.execute(
                    """
                    UPDATE token_usage
                    SET used_tokens = ?
                    WHERE user_id = ?
                      AND guild_id = ?
                      AND utc_date = ?
                    """,
                    (new_total, user_id, guild_id, utc_date),
                )
            await connection.commit()

        return actual_total

    async def usage_for_day(
        self,
        user_id: int,
        guild_id: int,
        utc_date: str,
    ) -> int:
        async with aiosqlite.connect(self.db_path) as connection:
            cursor = await connection.execute(
                """
                SELECT COALESCE(used_tokens, 0)
                FROM token_usage
                WHERE user_id = ?
                  AND guild_id = ?
                  AND utc_date = ?
                """,
                (user_id, guild_id, utc_date),
            )
            row = await cursor.fetchone()

        if row is None:
            return 0
        return int(row[0])

    async def __aenter__(self) -> "TokenUsageStore":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
