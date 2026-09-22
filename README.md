# Discord bot for personal server

## J.A.R.V.I.S. persona configuration

The system prompt is built at startup by [persona.py](persona.py) from
environment variables instead of being hardcoded, so it stays testable and
adjustable without editing code:

| Variable | Effect |
|---|---|
| `BOT_PERSONA` (`jarvis` default, or `plain`) | Selects the J.A.R.V.I.S. system prompt or a neutral generic-assistant fallback. |
| `JARVIS_OWNER_HONORIFIC` | If set, the prompt instructs the bot to address the server owner with this honorific; if unset, no honorific is used. |
| `JARVIS_FORMALITY` (`casual`/`balanced`/`formal`) | Adjusts the tone guidance in the prompt. |
| `JARVIS_HUMOR_LEVEL` (`none`/`low`/`medium`) | Adjusts how much dry wit the prompt permits. |
| `JARVIS_DEFAULT_VERBOSITY` (`concise`/`detailed`) | Adjusts the default reply-length guidance. |
| `JARVIS_USE_MARVEL_REFERENCES` (bool) | Permits occasional light Stark-adjacent references when `true`; otherwise the prompt forbids them. |
| `JARVIS_REQUIRE_PATCH_FOR_META` (bool) | When `true`, live-meta answers must cite patch-grounded sources, never model memory alone. |

These only change wording in the system prompt; they never remove the
non-negotiable safety sections (never-guess-meta rule, boundaries, and the
"not the literal copyrighted character" disclaimer), which are identical for
every persona. See [tests/test_persona.py](tests/test_persona.py) for the
exact behavior each setting produces.

## Model routing strategy

OpenRouter requests are routed by task instead of using one model for every
request. The routing logic lives in [llm_routing.py](llm_routing.py) so model
IDs are defined in one place instead of being scattered across commands.

Requests are classified deterministically and cheaply (no extra LLM call) via
`classify_request()` in [llm_routing.py](llm_routing.py), using command type,
explicit flags, and keyword rules:

- `static_fact` — answered from local/structured data (e.g. Data Dragon) when
  possible; only falls back to the general route when an LLM is actually
  needed. Local lookups are currently a stub (`default_static_fact_resolver`)
  pending a real Data Dragon integration; see the `TODO` in
  [llm_routing.py](llm_routing.py).
- `general_chat` — always uses the **general route**.
- `live_meta` — uses the **research route**, ideally after current
  patch/search context has been retrieved. Live search is currently a stub
  (`live_search.py`); until it is implemented, responses disclose that
  current-patch data could not be verified rather than guessing.
- `player_specific` — uses Riot API data when available (`riot_api.py`,
  currently a stub that always reports "no data"), then the research route to
  synthesize it. The bot never invents player data when the Riot API is
  unavailable.

`selectModelRoute()` maps a category to one of three routes:

| Route | Used for | Models (env var) |
|---|---|---|
| General | `general_chat`, and `static_fact` when an LLM is needed | `OPENROUTER_GENERAL_MODELS` |
| Research | `live_meta`, `player_specific` | `OPENROUTER_RESEARCH_MODELS` |
| Free-only | forced by budget exhaustion or `LLM_FREE_ONLY`/`LLM_ROUTING_MODE=free_only` | `OPENROUTER_FREE_MODEL` |

Each route sends OpenRouter an ordered model list (primary + fallbacks) via
`extra_body.models`, using OpenRouter's own model-fallback mechanism instead of
manual retry loops. Provider fallback stays enabled; the request is not
restricted to a single provider.

**Warning:** OpenRouter pricing, model availability, promotions, and
free-model membership can change at any time. Review current model pricing
and data policies on OpenRouter before a production launch, and re-verify the
example model IDs in [.env.example](.env.example) — they may no longer exist
or may no longer be free/cheap.

## Environment setup

Copy [.env.example](.env.example) to `.env` and fill in secrets. Key routing
variables:

```
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_GENERAL_MODELS=qwen/qwen3.7-flash,deepseek/deepseek-v4-flash-0731,qwen/qwen3.7-plus
OPENROUTER_RESEARCH_MODELS=google/gemini-3.7-flash,qwen/qwen3.7-plus,z-ai/glm-5.3-flash
OPENROUTER_FREE_MODEL=openrouter/free
LLM_ROUTING_MODE=normal
LLM_FREE_ONLY=false
LLM_MAX_OUTPUT_TOKENS_GENERAL=500
LLM_MAX_OUTPUT_TOKENS_RESEARCH=900
LLM_REQUEST_TIMEOUT_SECONDS=45
LLM_MAX_RETRIES=2
```

Model lists are comma-separated; whitespace around each ID is trimmed and
empty IDs are rejected. If `OPENROUTER_API_KEY` is missing, the bot fails
clearly at startup instead of silently attempting an LLM request. Never commit
a real API key, print it, or log it — only model IDs, token counts, and
estimated cost are logged.

### Overriding model lists

Set `OPENROUTER_GENERAL_MODELS` / `OPENROUTER_RESEARCH_MODELS` /
`OPENROUTER_FREE_MODEL` to change the routed model lists. For backward
compatibility, this repository's original `OPENROUTER_MODELS` variable (and
the more generic `OPENROUTER_MODEL`/`LLM_MODEL` single-model overrides) are
still honored: if set, they override the general route's primary model and
log a non-fatal deprecation warning. Migrate by moving those values into
`OPENROUTER_GENERAL_MODELS` and removing the deprecated variable.

### Free-only mode

Set `LLM_FREE_ONLY=true` or `LLM_ROUTING_MODE=free_only` to force every LLM
request through `OPENROUTER_FREE_MODEL`, regardless of category or budget.
Data Dragon/local static-fact lookups and other no-LLM features keep working
in this mode.

### Budget degradation

The daily budget ladder (`DAILY_LLM_BUDGET_USD`, `DAILY_BUDGET_CACHE_THRESHOLD`,
`DAILY_BUDGET_CHEAP_THRESHOLD`) still works as before, now combined with
task-based routing:

1. **Normal** — general route for ordinary chat; research route only for
   live-meta and player-specific synthesis.
2. **At/above `DAILY_BUDGET_CACHE_THRESHOLD`** — prefer cached/precomputed
   responses and avoid duplicate searches.
3. **At/above `DAILY_BUDGET_CHEAP_THRESHOLD`** — route all eligible LLM
   traffic through the general route, reduce output-token limits, and skip
   optional research calls.
4. **Free-only** (`LLM_FREE_ONLY=true` or `LLM_ROUTING_MODE=free_only`) — use
   `OPENROUTER_FREE_MODEL`, keep caching on, and apply strict output limits.
   Data Dragon and other no-LLM features keep working.
5. **Budget exhausted** — stop paid LLM requests, serve cached answers when
   available, keep Data Dragon/static lookups working, and return a friendly
   degraded-mode message if no cached answer exists.

`DAILY_AI_BUDGET_USD`, `CACHE_FIRST_THRESHOLD`, `ECONOMY_THRESHOLD`, and
`RESPONSE_CACHE_ENABLED` are still honored as deprecated aliases for
`DAILY_LLM_BUDGET_USD`, `DAILY_BUDGET_CACHE_THRESHOLD`,
`DAILY_BUDGET_CHEAP_THRESHOLD`, and `ENABLE_RESPONSE_CACHE` respectively.

## Quota deployment model

The bot is deployed as one `discord.Client` process using one SQLite file.
During startup database initialization, aggregate token reservations are reset
to zero. This recovers reservations left by a process crash before new requests
are accepted. This recovery is not a distributed lock or multi-replica safety
mechanism; deployments sharing one SQLite file across multiple bot processes
must use request-level reservation IDs and expiration handling instead.

## Request and budget limits

AI-triggering requests share a persistent per-user rolling limit across servers
and DMs. The default is `USER_RATE_LIMIT_REQUESTS=10` within
`USER_RATE_LIMIT_WINDOW_SECONDS=3600`; cache hits count toward this limit, while
help, health, status, usage, and owner operations do not. Daily token quotas and
the global `DAILY_AI_BUDGET_USD` reset at `00:00 UTC`. The global cap uses
integer microdollars and will serve only eligible cache or local results when it
is exhausted. Configure a hard provider spending limit as a separate backstop.

## Railway deployment

1. Deploy this repository as a GitHub-connected Railway service. Railway should
	detect the root [Dockerfile](Dockerfile).
2. Create and attach a Railway volume mounted at `/data`, then set
	`SQLITE_PATH=/data/jarvis.db`. Keep exactly one replica while SQLite is used.
3. Configure secrets through Railway's Variables tab, never through Git. Required
<<<<<<< HEAD
	variables are `DISCORD_BOT_TOKEN`, `OPENROUTER_API_KEY`, and
	`OPENROUTER_MODELS`. `OPENROUTER_MODELS` must contain one to three valid,
	comma-separated model IDs. Verify the example OpenRouter model IDs in
	[.env.example](.env.example) before using them; do not assume they remain valid.
	Other settings may be copied from [.env.example](.env.example).
	Changing Railway Variables triggers a redeploy.
=======
	variables are `DISCORD_BOT_TOKEN` and `OPENROUTER_API_KEY`. Model routing
	defaults to the built-in model lists documented above; set
	`OPENROUTER_GENERAL_MODELS` / `OPENROUTER_RESEARCH_MODELS` /
	`OPENROUTER_FREE_MODEL` to override them. Verify the example OpenRouter
	model IDs in [.env.example](.env.example) before using them; do not assume
	they remain valid, free, or available. Other settings may be copied from
	[.env.example](.env.example).
>>>>>>> d1f9252e92e69f3db908e3d99e2a59a14652d3fe
4. No public domain is required for this Discord worker. Deploy and verify that
	logs show SQLite initialization and a successful Discord login. Restart the
	service and confirm persisted usage/cache data survives.
5. Paid Railway plans can use Always restart; free/trial plans may be limited to
	On Failure with 10 retries.

