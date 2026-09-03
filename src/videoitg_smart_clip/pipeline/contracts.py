"""Stable contracts for the coarse-to-fine temporal grounding pipeline.

Business code depends on these protocols and records, rather than on a
particular grounding model.  Implementations may be replaced independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class CandidateWindow:
    start: float
    end: float
    score: float
    candidate_id: str = ""

    def __post_init__(self) -> None:
        if any(not math.isfinite(float(value)) for value in (self.start, self.end, self.score)):
            raise ValueError("candidate window values must be finite")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("candidate window must satisfy 0 <= start < end")


@dataclass(frozen=True)
class GroundingPrediction:
    candidate_id: str
    raw_start: float
    raw_end: float
    grounding_score: float
    inference_latency_ms: float = 0.0
    model_version: str = "unknown"
    boundary_confidence: float = 0.0
    completeness_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not math.isfinite(float(value)) for value in (self.raw_start, self.raw_end, self.grounding_score, self.inference_latency_ms, self.boundary_confidence, self.completeness_score)):
            raise ValueError("grounding prediction values must be finite")
        if self.raw_start < 0 or self.raw_end <= self.raw_start:
            raise ValueError("grounding prediction must satisfy 0 <= raw_start < raw_end")


class TemporalRetriever(Protocol):
    def index(self, video_path: str | Path, video_id: str) -> Any:
        ...

    def retrieve(self, video_id: str, query: str, top_n: int) -> list[CandidateWindow]:
        ...


class TemporalGrounder(Protocol):
    model_version: str

    def predict(
        self,
        video_path: str | Path,
        query: str,
        candidate_windows: Sequence[CandidateWindow],
    ) -> list[GroundingPrediction]:
        ...
