"""Typed local fallback boundary for degraded AI mode."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LocalFallbackResult:
    answer: str
    data_version: str | None = None


class LocalFallbackProvider(Protocol):
    async def lookup(
        self,
        prompt: str,
        *,
        patch_version: str,
    ) -> LocalFallbackResult | None: ...


class UnavailableLocalFallback:
    async def lookup(
        self,
        prompt: str,
        *,
        patch_version: str,
    ) -> LocalFallbackResult | None:
        return None
