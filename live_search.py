"""Stub interface for live-meta search (current patch notes, external sources).

TODO: implement a real retrieval step (patch notes source, search API, or
cached game-data feed) so ``live_meta`` requests can be grounded in current
data before the research route is called. Currently always returns no
results so the bot degrades gracefully (discloses uncertainty) instead of
fabricating current-meta claims.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveMetaSearchResult:
    patch_version: str | None
    source_links: tuple[str, ...]
    context_text: str


async def fetch_live_meta_context(query: str) -> LiveMetaSearchResult | None:
    return None
