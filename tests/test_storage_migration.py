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