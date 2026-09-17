import asyncio

import aiosqlite

from storage import TokenUsageStore


def test_response_cache_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "migration.db"

    async def run() -> None:
        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        await store.initialize()
        await store.close()

        async with aiosqlite.connect(db_path) as connection:
            cursor = await connection.execute("PRAGMA table_info(response_cache)")
            columns = await cursor.fetchall()
            assert [(row[1], row[2], row[3], row[4]) for row in columns] == [
                ("cache_key", "TEXT", 0, None),
                ("patch_version", "TEXT", 1, None),
                ("question_hash", "TEXT", 1, None),
                ("answer", "TEXT", 1, None),
                ("sources_json", "TEXT", 1, "'[]'"),
                ("answer_type", "TEXT", 1, None),
                ("created_at", "INTEGER", 1, None),
                ("expires_at", "INTEGER", 1, None),
                ("last_accessed_at", "INTEGER", 1, None),
                ("hit_count", "INTEGER", 1, "0"),
            ]

            cursor = await connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'response_cache'"
            )
            table_sql = (await cursor.fetchone())[0]
            assert "CHECK (hit_count >= 0)" in table_sql

            cursor = await connection.execute("PRAGMA index_list(response_cache)")
            indexes = {row[1] for row in await cursor.fetchall()}
            assert "idx_response_cache_expires_at" in indexes

    asyncio.run(run())


def test_user_global_quota_migration_is_idempotent_and_preserves_legacy_data(
    tmp_path,
):
    db_path = tmp_path / "quota-migration.db"

    async def run() -> None:
        async with aiosqlite.connect(db_path) as connection:
            await connection.execute(
                """
                CREATE TABLE token_usage (
                    user_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    utc_date TEXT NOT NULL,
                    used_tokens INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, guild_id, utc_date)
                )
                """
            )
            await connection.execute(
                "INSERT INTO token_usage VALUES (?, ?, ?, ?)",
                (7, 101, "2026-09-18", 300),
            )
            await connection.execute(
                "INSERT INTO token_usage VALUES (?, ?, ?, ?)",
                (7, 202, "2026-09-18", 200),
            )
            await connection.commit()

        store = TokenUsageStore(str(db_path), daily_limit=1000)
        await store.initialize()
        await store.initialize()

        async with aiosqlite.connect(db_path) as connection:
            cursor = await connection.execute(
                "PRAGMA table_info(daily_user_token_usage)"
            )
            columns = {row[1]: row for row in await cursor.fetchall()}
            assert set(columns) == {
                "user_id",
                "usage_date",
                "prompt_tokens",
                "completion_tokens",
                "reserved_tokens",
                "successful_requests",
                "updated_at",
            }
            assert columns["prompt_tokens"][4] == "0"
            assert columns["completion_tokens"][4] == "0"
            assert columns["reserved_tokens"][4] == "0"
            assert columns["successful_requests"][4] == "0"

            cursor = await connection.execute(
                "SELECT prompt_tokens, completion_tokens, reserved_tokens "
                "FROM daily_user_token_usage WHERE user_id = ? AND usage_date = ?",
                (7, "2026-09-18"),
            )
            assert await cursor.fetchone() == (500, 0, 0)

            cursor = await connection.execute(
                "PRAGMA index_list(daily_user_token_usage)"
            )
            indexes = {row[1] for row in await cursor.fetchall()}
            assert "idx_daily_user_token_usage_date" in indexes

            cursor = await connection.execute(
                "PRAGMA table_info(user_token_quota_overrides)"
            )
            assert {row[1] for row in await cursor.fetchall()} == {
                "user_id",
                "daily_limit",
            }

            cursor = await connection.execute(
                "SELECT used_tokens FROM token_usage "
                "WHERE user_id = ? AND guild_id = ? AND utc_date = ?",
                (7, 101, "2026-09-18"),
            )
            assert await cursor.fetchone() == (300,)

        await store.close()

    asyncio.run(run())