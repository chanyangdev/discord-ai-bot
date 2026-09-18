"""Budget degradation policy and bounded external-work admission."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import AsyncIterator


class BudgetMode(StrEnum):
    NORMAL = "normal"
    CACHE_FIRST = "cache_first"
    ECONOMY = "economy"
    FREE_ONLY = "free_only"


APPROVED_ECONOMY_MODELS = frozenset(
    {
        "qwen/qwen3.8-27b:free",
        "google/gemini-3.5-flash-lite",
    }
)


def parse_economy_models(value: str) -> tuple[str, ...]:
    models = tuple(model.strip() for model in value.split(","))
    if not 1 <= len(models) <= 3 or any(not model for model in models):
        raise ValueError("economy models must contain one to three non-empty models")
    if len(set(models)) != len(models):
        raise ValueError("economy models must not contain duplicates")
    if any(model not in APPROVED_ECONOMY_MODELS for model in models):
        raise ValueError("economy model is not approved")
    return models


@dataclass(frozen=True)
class DegradationPolicyConfig:
    daily_budget_microdollars: int
    cache_first_threshold: Decimal
    economy_threshold: Decimal
    free_only_threshold: Decimal
    max_request_cost_microdollars: int
    normal_max_output_tokens: int
    economy_max_output_tokens: int
    economy_models: tuple[str, ...]
    paid_llm_enabled: bool

    def validate(self) -> None:
        if self.daily_budget_microdollars <= 0:
            raise ValueError("daily budget must be positive")
        if self.max_request_cost_microdollars <= 0:
            raise ValueError("maximum request cost must be positive")
        if self.max_request_cost_microdollars > self.daily_budget_microdollars:
            raise ValueError("maximum request cost cannot exceed daily budget")
        if self.normal_max_output_tokens <= 0:
            raise ValueError("normal output token limit must be positive")
        if self.economy_max_output_tokens <= 0:
            raise ValueError("economy output token limit must be positive")
        thresholds = (
            self.cache_first_threshold,
            self.economy_threshold,
            self.free_only_threshold,
        )
        if any(
            not threshold.is_finite() or threshold < 0 or threshold > 1
            for threshold in thresholds
        ):
            raise ValueError("degradation thresholds must be between 0 and 1")
        if not (
            self.cache_first_threshold
            < self.economy_threshold
            < self.free_only_threshold
        ):
            raise ValueError("degradation thresholds must be strictly ordered")
        if not 1 <= len(self.economy_models) <= 3:
            raise ValueError("economy models must contain one to three models")
        if any(not model.strip() for model in self.economy_models):
            raise ValueError("economy models must be non-empty")


@dataclass(frozen=True)
class DegradationDecision:
    mode: BudgetMode
    utilization_ratio: Decimal
    cache_first: bool
    live_search_allowed: bool
    allowed_models: tuple[str, ...]
    max_output_tokens: int
    paid_provider_allowed: bool
    reason: str


def is_free_model(model: str) -> bool:
    return model.strip().endswith(":free")


class DegradationPolicy:
    def __init__(self, config: DegradationPolicyConfig) -> None:
        config.validate()
        self.config = config

    def decide(
        self,
        *,
        committed_microdollars: int,
        reserved_microdollars: int,
        normal_models: tuple[str, ...],
        paid_path_disabled: bool = False,
    ) -> DegradationDecision:
        if committed_microdollars < 0 or reserved_microdollars < 0:
            raise ValueError("budget usage cannot be negative")
        utilization = Decimal(committed_microdollars + reserved_microdollars) / Decimal(
            self.config.daily_budget_microdollars
        )
        if paid_path_disabled or utilization >= self.config.free_only_threshold:
            mode = BudgetMode.FREE_ONLY
            models = tuple(model for model in normal_models if is_free_model(model))
            reason = "paid_path_disabled" if paid_path_disabled else "budget_exhausted"
            return DegradationDecision(
                mode,
                utilization,
                True,
                False,
                models,
                self.config.economy_max_output_tokens,
                False,
                reason,
            )
        if utilization >= self.config.economy_threshold:
            return DegradationDecision(
                BudgetMode.ECONOMY,
                utilization,
                True,
                False,
                self.config.economy_models,
                self.config.economy_max_output_tokens,
                any(not is_free_model(model) for model in self.config.economy_models),
                "economy_threshold",
            )
        if utilization >= self.config.cache_first_threshold:
            return DegradationDecision(
                BudgetMode.CACHE_FIRST,
                utilization,
                True,
                True,
                normal_models,
                self.config.normal_max_output_tokens,
                any(not is_free_model(model) for model in normal_models),
                "cache_first_threshold",
            )
        return DegradationDecision(
            BudgetMode.NORMAL,
            utilization,
            False,
            True,
            normal_models,
            self.config.normal_max_output_tokens,
            any(not is_free_model(model) for model in normal_models),
            "normal_budget",
        )


class AdmissionError(RuntimeError):
    """Raised when external-work admission cannot be granted."""


class AdmissionShutdown(AdmissionError):
    """Raised when admission is closing or closed."""


class AdmissionState(StrEnum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class AdmissionController:
    def __init__(
        self,
        max_concurrent: int,
        max_waiting: int,
        timeout_seconds: float,
        shutdown_timeout_seconds: float = 5.0,
    ) -> None:
        if max_concurrent <= 0 or max_waiting < 0 or timeout_seconds <= 0:
            raise ValueError("invalid admission controller limits")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("invalid shutdown timeout")
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self.timeout_seconds = timeout_seconds
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._waiting = 0
        self._active = 0
        self._active_tasks: set[asyncio.Task[object]] = set()
        self._state = AdmissionState.OPEN
        self.shutdown_timeout_seconds = shutdown_timeout_seconds

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def active(self) -> int:
        return self._active

    @property
    def state(self) -> AdmissionState:
        return self._state

    async def _set_status(
        self,
        status_message: object | None,
        content: str,
    ) -> None:
        if status_message is None:
            return
        edit = getattr(status_message, "edit", None)
        if edit is not None:
            await edit(content=content)

    @asynccontextmanager
    async def admit(self, status_message: object | None = None) -> AsyncIterator[None]:
        task = asyncio.current_task()
        queued = False
        acquired = False
        try:
            async with self._condition:
                if self._state is not AdmissionState.OPEN:
                    raise AdmissionShutdown("admission_shutdown")
                queued = self._active >= self.max_concurrent
                if queued:
                    if self._waiting >= self.max_waiting:
                        raise AdmissionError("queue_full")
                    self._waiting += 1

            if queued:
                await self._set_status(
                    status_message,
                    "High traffic - your request is queued. Please try again later.",
                )

            async with self._condition:
                try:
                    await asyncio.wait_for(
                        self._wait_for_slot(), timeout=self.timeout_seconds
                    )
                except asyncio.TimeoutError as exc:
                    raise AdmissionError("queue_timeout") from exc
                finally:
                    if queued:
                        self._waiting -= 1
                        self._condition.notify_all()
                if self._state is not AdmissionState.OPEN:
                    raise AdmissionShutdown("admission_shutdown")
                self._active += 1
                acquired = True
                if task is not None:
                    self._active_tasks.add(task)
            yield
        finally:
            async with self._condition:
                if task is not None:
                    self._active_tasks.discard(task)
                if acquired:
                    self._active -= 1
                self._condition.notify_all()

    async def _wait_for_slot(self) -> None:
        while self._active >= self.max_concurrent:
            await self._condition.wait()

    async def close(self) -> None:
        current = asyncio.current_task()
        async with self._condition:
            if self._state is AdmissionState.CLOSED:
                return
            self._state = AdmissionState.CLOSING
            self._condition.notify_all()

        async def drained() -> None:
            async with self._condition:
                while self._active or self._waiting:
                    await self._condition.wait()

        try:
            await asyncio.wait_for(drained(), self.shutdown_timeout_seconds)
        except asyncio.TimeoutError:
            async with self._condition:
                tasks = [task for task in self._active_tasks if task is not current]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            async with self._condition:
                self._waiting = 0
                self._active = 0
                self._active_tasks.clear()
                self._state = AdmissionState.CLOSED
                self._condition.notify_all()


class BudgetUnavailable(RuntimeError):
    """Raised when paid work is unavailable under the current budget policy."""
