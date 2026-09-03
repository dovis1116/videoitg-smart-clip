"""Explicit, deterministic latency-budget decisions for offline replay.

This module does not execute inference or inspect global GPU state.  Callers
provide queue/deadline/uncertainty signals, and the returned decision is fully
serializable for tracing and rollback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetDecision:
    policy_version: str
    mode: str
    max_frames: int
    reason: str


@dataclass(frozen=True)
class BudgetPolicy:
    """A conservative policy whose thresholds must be set by profiling."""

    policy_version: str = "phase5-preflight-v0"
    default_max_frames: int = 16
    reduced_max_frames: int = 32
    full_max_frames: int = 64
    full_uncertainty_threshold: float | None = None
    reduced_queue_threshold: int | None = None
    bypass_queue_threshold: int | None = None
    min_deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if self.default_max_frames <= 0 or self.reduced_max_frames <= 0 or self.full_max_frames <= 0:
            raise ValueError("frame budgets must be positive")
        if not 0.0 <= (self.full_uncertainty_threshold if self.full_uncertainty_threshold is not None else 0.0) <= 1.0:
            raise ValueError("full_uncertainty_threshold must be in [0, 1]")
        for value in (self.reduced_queue_threshold, self.bypass_queue_threshold):
            if value is not None and value < 0:
                raise ValueError("queue thresholds must be non-negative")
        if self.min_deadline_ms is not None and self.min_deadline_ms <= 0:
            raise ValueError("min_deadline_ms must be positive")

    def choose(
        self,
        *,
        queue_length: int,
        uncertainty: float | None = None,
        deadline_ms: float | None = None,
    ) -> BudgetDecision:
        if queue_length < 0:
            raise ValueError("queue_length must be non-negative")
        if uncertainty is not None and not 0.0 <= uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        if deadline_ms is not None and deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if self.bypass_queue_threshold is not None and queue_length >= self.bypass_queue_threshold:
            return BudgetDecision(self.policy_version, "retrieval_only", 0, "queue_bypass_threshold")
        if self.min_deadline_ms is not None and deadline_ms is not None and deadline_ms < self.min_deadline_ms:
            return BudgetDecision(self.policy_version, "retrieval_only", 0, "deadline_below_minimum")
        if (
            self.full_uncertainty_threshold is not None
            and uncertainty is not None
            and uncertainty >= self.full_uncertainty_threshold
            and (self.reduced_queue_threshold is None or queue_length <= self.reduced_queue_threshold)
        ):
            return BudgetDecision(self.policy_version, "full", self.full_max_frames, "high_uncertainty")
        if self.reduced_queue_threshold is not None and queue_length <= self.reduced_queue_threshold:
            return BudgetDecision(self.policy_version, "reduced", self.reduced_max_frames, "within_reduced_queue_budget")
        return BudgetDecision(self.policy_version, "fixed", self.default_max_frames, "default_budget")
