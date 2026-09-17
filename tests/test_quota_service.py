import asyncio

import pytest

from quota_service import QuotaService


class FakeUsageStore:
    def __init__(self, override):
        self.override = override
        self.looked_up_users = []

    async def get_quota_override(self, user_id):
        self.looked_up_users.append(user_id)
        return self.override


@pytest.mark.parametrize("override", [123, 1])
def test_database_override_wins(override):
    async def run() -> None:
        store = FakeUsageStore(override)
        service = QuotaService(
            store,
            free_daily_limit=100,
            premium_daily_limit=1000,
            paid_entitlement_lookup=lambda user_id: _paid(True),
        )
        assert await service.resolve_daily_limit(7) == override
        assert store.looked_up_users == [7]

    asyncio.run(run())


def test_paid_user_gets_premium_limit():
    async def run() -> None:
        service = QuotaService(
            FakeUsageStore(None),
            free_daily_limit=100,
            premium_daily_limit=1000,
            paid_entitlement_lookup=lambda user_id: _paid(True),
        )
        assert await service.resolve_daily_limit(7) == 1000

    asyncio.run(run())


def test_free_user_gets_free_limit():
    async def run() -> None:
        service = QuotaService(
            FakeUsageStore(None),
            free_daily_limit=100,
            premium_daily_limit=1000,
            paid_entitlement_lookup=lambda user_id: _paid(False),
        )
        assert await service.resolve_daily_limit(7) == 100

    asyncio.run(run())


@pytest.mark.parametrize("override", [None, 0, -1, True, "100"])
def test_invalid_override_continues_to_entitlement_policy(override):
    async def run() -> None:
        service = QuotaService(
            FakeUsageStore(override),
            free_daily_limit=100,
            premium_daily_limit=1000,
            paid_entitlement_lookup=lambda user_id: _paid(True),
        )
        assert await service.resolve_daily_limit(7) == 1000

    asyncio.run(run())


def test_entitlement_lookup_failure_defaults_to_free():
    async def lookup(user_id):
        raise RuntimeError("provider unavailable")

    async def run() -> None:
        service = QuotaService(
            FakeUsageStore(None),
            free_daily_limit=100,
            premium_daily_limit=1000,
            paid_entitlement_lookup=lookup,
        )
        assert await service.resolve_daily_limit(7) == 100

    asyncio.run(run())


def test_service_limits_must_be_positive():
    with pytest.raises(ValueError):
        QuotaService(
            FakeUsageStore(None),
            free_daily_limit=0,
            premium_daily_limit=1000,
            paid_entitlement_lookup=lambda user_id: _paid(False),
        )


async def _paid(value: bool) -> bool:
    return value