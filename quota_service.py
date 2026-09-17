from collections.abc import Awaitable, Callable
from typing import Any

from storage import TokenUsageStore


EntitlementLookup = Callable[[int], Awaitable[bool]]


class QuotaService:
    def __init__(
        self,
        usage_store: TokenUsageStore,
        *,
        free_daily_limit: int,
        premium_daily_limit: int,
        paid_entitlement_lookup: EntitlementLookup,
    ) -> None:
        self.usage_store = usage_store
        self.free_daily_limit = self._positive_limit(
            "free_daily_limit", free_daily_limit
        )
        self.premium_daily_limit = self._positive_limit(
            "premium_daily_limit", premium_daily_limit
        )
        self.paid_entitlement_lookup = paid_entitlement_lookup

    async def resolve_daily_limit(self, user_id: int) -> int:
        override = await self.usage_store.get_quota_override(user_id)
        if self._is_valid_limit(override):
            return override

        try:
            is_paid = await self.paid_entitlement_lookup(user_id)
        except Exception:
            is_paid = False

        return self.premium_daily_limit if is_paid else self.free_daily_limit

    @staticmethod
    def _is_valid_limit(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @classmethod
    def _positive_limit(cls, name: str, value: int) -> int:
        if not cls._is_valid_limit(value):
            raise ValueError(f"{name} must be a positive integer")
        return value