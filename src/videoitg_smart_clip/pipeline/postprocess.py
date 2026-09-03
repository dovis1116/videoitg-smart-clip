"""Independent boundary, ranking, deduplication and no-match stages."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Iterable, Sequence

from .contracts import CandidateWindow, GroundingPrediction


class BoundaryRefiner:
    version = "boundary-refinement-v1"

    def __init__(self, *, enabled: bool = True, expansion_seconds: float = 0.0, start_offset_seconds: float = 0.0, end_offset_seconds: float = 0.0, start_padding_seconds: float = 0.0, end_padding_seconds: float = 0.0) -> None:
        self.enabled = enabled
        numeric = {
            "expansion_seconds": expansion_seconds,
            "start_offset_seconds": start_offset_seconds,
            "end_offset_seconds": end_offset_seconds,
            "start_padding_seconds": start_padding_seconds,
            "end_padding_seconds": end_padding_seconds,
        }
        if any(not math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("boundary parameters must be finite")
        if float(expansion_seconds) < 0 or float(start_padding_seconds) < 0 or float(end_padding_seconds) < 0:
            raise ValueError("boundary expansion and padding must be non-negative")
        self.expansion_seconds = float(expansion_seconds)
        self.start_offset_seconds = float(start_offset_seconds)
        self.end_offset_seconds = float(end_offset_seconds)
        self.start_padding_seconds = float(start_padding_seconds)
        self.end_padding_seconds = float(end_padding_seconds)

    def refine(self, prediction: GroundingPrediction, *, duration: float | None = None) -> dict[str, float | bool]:
        start, end = prediction.raw_start, prediction.raw_end
        if self.enabled:
            start = start + self.start_offset_seconds - self.start_padding_seconds - self.expansion_seconds
            end = end + self.end_offset_seconds + self.end_padding_seconds + self.expansion_seconds
            start = max(0.0, start)
            if duration is not None:
                end = min(duration, end)
            if end <= start:
                raise ValueError("refined boundary must satisfy start < end")
        return {"refined_start": start, "refined_end": end, "boundary_refined": self.enabled}


class CandidateRanker:
    version = "ranking-v1"

    def __init__(self, *, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {"retrieval": 0.30, "grounding": 0.40, "boundary": 0.10, "completeness": 0.20, "duplication": 0.0}
        required = ("retrieval", "grounding", "completeness")
        if any(key not in self.weights for key in required):
            raise ValueError("ranking weights must include retrieval, grounding and completeness")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in self.weights.values()):
            raise ValueError("ranking weights must be finite and non-negative")

    def score(self, retrieval_score: float, prediction: GroundingPrediction, *, duplication_penalty: float = 0.0) -> float:
        return (
            self.weights["retrieval"] * retrieval_score
            + self.weights["grounding"] * prediction.grounding_score
            + self.weights.get("boundary", 0.0) * prediction.boundary_confidence
            + self.weights["completeness"] * prediction.completeness_score
            - self.weights.get("duplication", 0.0) * duplication_penalty
        )


def temporal_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


class TemporalDeduplicator:
    version = "temporal-dedup-v1"

    def __init__(self, *, temporal_iou_threshold: float = 0.7) -> None:
        if not 0 <= temporal_iou_threshold <= 1:
            raise ValueError("temporal_iou_threshold must be between 0 and 1")
        self.temporal_iou_threshold = temporal_iou_threshold

    def apply(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: item.get("final_score", 0.0), reverse=True):
            interval = (float(row["refined_start"]), float(row["refined_end"]))
            if any(temporal_iou(interval, (float(old["refined_start"]), float(old["refined_end"]))) > self.temporal_iou_threshold for old in kept):
                continue
            kept.append(dict(row))
        for rank, row in enumerate(kept, 1):
            row["rank"] = rank
        return kept


class NoMatchDecider:
    version = "no-match-v1"

    def __init__(self, *, retrieval_threshold: float | None = None, grounding_threshold: float | None = None, margin_threshold: float | None = None) -> None:
        self.retrieval_threshold = retrieval_threshold
        self.grounding_threshold = grounding_threshold
        self.margin_threshold = margin_threshold
        for name, value in (("retrieval_threshold", retrieval_threshold), ("grounding_threshold", grounding_threshold)):
            if value is not None and (not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if margin_threshold is not None and (not math.isfinite(float(margin_threshold)) or float(margin_threshold) < 0.0):
            raise ValueError("margin_threshold must be finite and non-negative")

    def decide(self, rows: Sequence[dict[str, Any]]) -> str:
        if not rows:
            return "NO_MATCH"
        ordered = sorted(rows, key=lambda row: row.get("final_score", 0.0), reverse=True)
        top = float(ordered[0].get("final_score", 0.0))
        second = float(ordered[1].get("final_score", 0.0)) if len(ordered) > 1 else 0.0
        if self.retrieval_threshold is None or self.grounding_threshold is None or self.margin_threshold is None:
            return "POSSIBLE"
        retrieval = float(ordered[0].get("retrieval_score", top))
        if retrieval < self.retrieval_threshold or float(ordered[0].get("grounding_score", 0.0)) < self.grounding_threshold:
            return "NO_MATCH"
        if top - second < self.margin_threshold:
            return "POSSIBLE"
        return "CONFIDENT"


def build_candidate_record(window: CandidateWindow, prediction: GroundingPrediction, refined: dict[str, float | bool], *, final_score: float, retriever_version: str, postprocess_version: str, retrieval_latency_ms: float = 0.0, postprocess_latency_ms: float = 0.0, duplication_penalty: float = 0.0, degraded: bool = False, degrade_level: int = 0, degrade_reason: str | None = None) -> dict[str, Any]:
    """Create the lossless candidate schema; raw model bounds are never overwritten."""
    return {
        "candidate_id": prediction.candidate_id or window.candidate_id,
        "coarse_start": window.start,
        "coarse_end": window.end,
        "retrieval_score": window.score,
        "raw_start": prediction.raw_start,
        "raw_end": prediction.raw_end,
        "grounding_score": prediction.grounding_score,
        "refined_start": refined["refined_start"],
        "refined_end": refined["refined_end"],
        "final_score": final_score,
        "duplication_penalty": duplication_penalty,
        "rank": None,
        "pre_dedup_rank": None,
        "retriever_version": retriever_version,
        "grounder_version": prediction.model_version,
        "postprocess_version": postprocess_version,
        "retrieval_latency_ms": retrieval_latency_ms,
        "grounding_latency_ms": prediction.inference_latency_ms,
        "postprocess_latency_ms": postprocess_latency_ms,
        "degraded": degraded,
        "degrade_level": degrade_level,
        "degrade_reason": degrade_reason,
        "deduplicated": False,
        "boundary_confidence": prediction.boundary_confidence,
        "completeness_score": prediction.completeness_score,
    }
