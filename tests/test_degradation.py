import asyncio
from decimal import Decimal

import pytest

from degradation import (
    APPROVED_ECONOMY_MODELS,
    AdmissionController,
    AdmissionError,
    AdmissionShutdown,
    BudgetMode,
    DegradationPolicy,
    DegradationPolicyConfig,
    parse_economy_models,
)
from storage import GlobalCostUsage, QuotaExceeded, TokenUsageStore


NORMAL_MODELS = ("qwen/qwen3.8-27b:free", "paid/cheap", "paid/reliable")


def make_policy(**overrides):
    values = {
        "daily_budget_microdollars": 1_000_000,
        "cache_first_threshold": Decimal("0.60"),
        "economy_threshold": Decimal("0.80"),
        "free_only_threshold": Decimal("1.00"),
        "max_request_cost_microdollars": 10_000,
        "normal_max_output_tokens": 1024,
        "economy_max_output_tokens": 512,
        "economy_models": ("qwen/qwen3.8-27b:free",),
        "paid_llm_enabled": True,
    }
    values.update(overrides)
    return DegradationPolicy(DegradationPolicyConfig(**values))


@pytest.mark.parametrize(
    ("used", "mode"),
    [
        (0, BudgetMode.NORMAL),
        (599_999, BudgetMode.NORMAL),
        (600_000, BudgetMode.CACHE_FIRST),
        (799_999, BudgetMode.CACHE_FIRST),
        (800_000, BudgetMode.ECONOMY),
        (999_999, BudgetMode.ECONOMY),
        (1_000_000, BudgetMode.FREE_ONLY),
        (1_000_001, BudgetMode.FREE_ONLY),
    ],
)
def test_policy_boundaries_are_deterministic(used, mode):
    decision = make_policy().decide(
        committed_microdollars=used,
        reserved_microdollars=0,
        normal_models=NORMAL_MODELS,
    )

    assert decision.mode is mode


def test_invalid_thresholds_fail_validation():
    with pytest.raises(ValueError):
        make_policy(economy_threshold=Decimal("0.50"))

    with pytest.raises(ValueError):
        make_policy(free_only_threshold=Decimal("1.10"))


def test_economy_allowlist_is_ordered_and_rejects_unapproved_values():
    ordered = parse_economy_models(
        "google/gemini-3.5-flash-lite,qwen/qwen3.8-27b:free"
    )
    assert ordered == (
        "google/gemini-3.5-flash-lite",
        "qwen/qwen3.8-27b:free",
    )
    assert set(ordered) <= APPROVED_ECONOMY_MODELS
    for value in (
        "anthropic/claude-opus-5",
        "arbitrary/model",
        "qwen/qwen3.8-27b:free,qwen/qwen3.8-27b:free",
        "",
        "a,b,c,d",
    ):
        with pytest.raises(ValueError):
            parse_economy_models(value)


def test_free_only_filters_paid_models_and_manual_disable():
    decision = make_policy().decide(
        committed_microdollars=0,
        reserved_microdollars=0,
        normal_models=NORMAL_MODELS,
        paid_path_disabled=True,
    )

    assert decision.mode is BudgetMode.FREE_ONLY
    assert decision.allowed_models == ("qwen/qwen3.8-27b:free",)
    assert decision.paid_provider_allowed is False


def test_free_only_uses_explicit_free_models_when_configured():
    decision = make_policy(free_models=("openrouter/free",)).decide(
        committed_microdollars=1_000_000,
        reserved_microdollars=0,
        normal_models=NORMAL_MODELS,
    )

    assert decision.mode is BudgetMode.FREE_ONLY
    assert decision.allowed_models == ("openrouter/free",)


def test_free_models_config_rejects_empty_list():
    with pytest.raises(ValueError):
        make_policy(free_models=())


async def _run_controller(controller, active, started, release):
    async with controller.admit():
        active.append(1)
        started.set()
        await release.wait()
        active.pop()


def test_admission_never_exceeds_concurrency():
    async def run():
        controller = AdmissionController(2, 4, 1)
        active = []
        started = asyncio.Event()
        release = asyncio.Event()
        tasks = [
            asyncio.create_task(_run_controller(controller, active, started, release))
            for _ in range(6)
        ]
        await started.wait()
        await asyncio.sleep(0)
        assert controller.active <= 2
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(run())


def test_admission_rejects_full_queue():
    async def run():
        controller = AdmissionController(1, 0, 0.1)
        release = asyncio.Event()

        async def hold():
            async with controller.admit():
                await release.wait()

        holder = asyncio.create_task(hold())
        await asyncio.sleep(0)
        with pytest.raises(AdmissionError, match="queue_full"):
            async with controller.admit():
                pass
        release.set()
        await holder

    asyncio.run(run())


def test_global_cost_reservations_are_atomic(tmp_path):
    async def run():
        store = TokenUsageStore(str(tmp_path / "budget.db"), daily_limit=1000)
        await store.initialize()

        async def reserve():
            try:
                await store.reserve_global_cost(600_000, 1_000_000)
                return True
            except QuotaExceeded:
                return False

        results = await asyncio.gather(reserve(), reserve())
        assert results.count(True) == 1
        usage = await store.get_global_cost_usage()
        assert usage.reserved_microdollars == 600_000
        await store.close()

    asyncio.run(run())


def test_global_cost_reconciles_authoritative_cost(tmp_path):
    async def run():
        store = TokenUsageStore(str(tmp_path / "budget.db"), daily_limit=1000)
        await store.initialize()
        await store.reserve_global_cost(600_000, 1_000_000)
        await store.reconcile_global_cost(600_000, 250_000, successful=True)
        usage = await store.get_global_cost_usage()
        assert usage == GlobalCostUsage(
            usage.usage_date, 250_000, 0, 1, usage.updated_at
        )
        await store.close()

    asyncio.run(run())


def test_global_cost_unknown_failure_keeps_conservative_reservation(tmp_path):
    async def run():
        store = TokenUsageStore(str(tmp_path / "budget.db"), daily_limit=1000)
        await store.initialize()
        await store.reserve_global_cost(600_000, 1_000_000)
        await store.reconcile_global_cost(600_000, None, successful=False)
        usage = await store.get_global_cost_usage()
        assert usage.committed_microdollars == 600_000
        assert usage.reserved_microdollars == 0
        assert usage.successful_paid_requests == 0
        await store.close()

    asyncio.run(run())


def test_global_cost_startup_recovers_crashed_reservation(tmp_path):
    async def run():
        store = TokenUsageStore(str(tmp_path / "budget.db"), daily_limit=1000)
        await store.initialize()
        await store.reserve_global_cost(600_000, 1_000_000)
        await store.close()

        recovered = TokenUsageStore(str(tmp_path / "budget.db"), daily_limit=1000)
        await recovered.initialize()
        usage = await recovered.get_global_cost_usage()
        assert usage.committed_microdollars == 600_000
        assert usage.reserved_microdollars == 0
        await recovered.close()

    asyncio.run(run())


def test_queue_timeout_releases_waiter():
    async def run():
        controller = AdmissionController(1, 1, 0.01)
        release = asyncio.Event()

        async def hold():
            async with controller.admit():
                await release.wait()

        holder = asyncio.create_task(hold())
        await asyncio.sleep(0)
        with pytest.raises(AdmissionError, match="queue_timeout"):
            async with controller.admit():
                pass
        assert controller.waiting == 0
        release.set()
        await holder

    asyncio.run(run())


def test_admission_shutdown_rejects_new_work_and_drains():
    async def run():
        controller = AdmissionController(1, 2, 1, 0.05)
        release = asyncio.Event()

        async def hold():
            async with controller.admit():
                await release.wait()

        holder = asyncio.create_task(hold())
        await asyncio.sleep(0)
        async def wait_for_shutdown():
            async with controller.admit():
                pass

        waiter = asyncio.create_task(wait_for_shutdown())
        await asyncio.sleep(0)
        await controller.close()
        release.set()
        await asyncio.gather(holder, waiter, return_exceptions=True)
        with pytest.raises(AdmissionShutdown):
            await controller.admit().__aenter__()
        assert controller.active == 0
        assert controller.waiting == 0
        await controller.close()

    asyncio.run(run())
