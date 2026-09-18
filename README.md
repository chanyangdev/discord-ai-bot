# Discord bot for personal server

## Quota deployment model

The bot is deployed as one `discord.Client` process using one SQLite file.
During startup database initialization, aggregate token reservations are reset
to zero. This recovers reservations left by a process crash before new requests
are accepted. This recovery is not a distributed lock or multi-replica safety
mechanism; deployments sharing one SQLite file across multiple bot processes
must use request-level reservation IDs and expiration handling instead.

## Railway deployment

1. Deploy this repository as a GitHub-connected Railway service. Railway should
	detect the root [Dockerfile](Dockerfile).
2. Create and attach a Railway volume mounted at `/data`, then set
	`SQLITE_PATH=/data/jarvis.db`. Keep exactly one replica while SQLite is used.
3. Configure secrets through Railway's Variables tab, never through Git. Required
	variables are `DISCORD_BOT_TOKEN`, `OPENROUTER_API_KEY`, and
	`OPENROUTER_MODELS`. `OPENROUTER_MODELS` must contain one to three valid,
	comma-separated model IDs. Verify the example OpenRouter model IDs in
	[.env.example](.env.example) before using them; do not assume they remain valid.
	Other settings may be copied from [.env.example](.env.example).
4. No public domain is required for this Discord worker. Deploy and verify that
	logs show SQLite initialization and a successful Discord login. Restart the
	service and confirm persisted usage/cache data survives.
5. Paid Railway plans can use Always restart; free/trial plans may be limited to
	On Failure with 10 retries.

