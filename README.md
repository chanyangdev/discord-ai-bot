# Discord bot for personal server

## Quota deployment model

The bot is deployed as one `discord.Client` process using one SQLite file.
During startup database initialization, aggregate token reservations are reset
to zero. This recovers reservations left by a process crash before new requests
are accepted. This recovery is not a distributed lock or multi-replica safety
mechanism; deployments sharing one SQLite file across multiple bot processes
must use request-level reservation IDs and expiration handling instead.

