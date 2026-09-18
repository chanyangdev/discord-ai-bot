"""Minimal Riot API client interface.

TODO: implement real Riot API calls (account/summoner lookups, ranked stats,
match history) once API keys, rate limiting, and regional routing are
configured. This stub always reports "no data" (rather than raising) so
``player_specific`` requests degrade gracefully instead of inventing player
data when the Riot API is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RiotApiUnavailable(RuntimeError):
    """Raised when Riot API configuration is required but missing."""


@dataclass(frozen=True)
class RiotPlayerData:
    riot_id: str
    payload: dict[str, Any]


async def fetch_player_data(riot_id: str) -> RiotPlayerData | None:
    """Return structured Riot API data for a player, or None if unavailable.

    Returning ``None`` (instead of raising) lets callers treat "no data" as
    an expected, user-safe degraded case: they must not invent player data
    when this returns None.
    """
    return None
