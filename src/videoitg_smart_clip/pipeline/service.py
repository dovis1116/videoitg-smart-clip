"""Orchestration for query-aware coarse-to-fine temporal grounding."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .contracts import CandidateWindow, TemporalGrounder, TemporalRetriever
from .postprocess import BoundaryRefiner, CandidateRanker, NoMatchDecider, TemporalDeduplicator, build_candidate_record


class CoarseToFinePipeline:
    def __init__(self, retriever: TemporalRetriever, grounder: TemporalGrounder, *, boundary_refiner: BoundaryRefiner | None = None, ranker: CandidateRanker | None = None, deduplicator: TemporalDeduplicator | None = None, no_match: NoMatchDecider | None = None, top_n: int = 20, top_k: int = 3) -> None:
        self.retriever = retriever
        self.grounder = grounder
        self.boundary_refiner = boundary_refiner or BoundaryRefiner()
        self.ranker = ranker or CandidateRanker()
        self.deduplicator = deduplicator or TemporalDeduplicator()
        self.no_match = no_match or NoMatchDecider()
        self.top_n = top_n
        self.top_k = top_k

    def search(self, video_path: str | Path, video_id: str, query: str, *, duration: float | None = None, stage_callback: Callable[[str, float], None] | None = None) -> dict[str, Any]:
        if stage_callback:
            stage_callback("RETRIEVING", 0.40)
        retrieval_started = time.perf_counter()
        windows = self._normalize_windows(video_id, self.retriever.retrieve(video_id, query, self.top_n))
        retrieval_latency = (time.perf_counter() - retrieval_started) * 1000.0
        if stage_callback:
            stage_callback("GROUNDING", 0.62)
        if not windows:
            # An empty coarse recall is already a conclusive no-match for this
            # query.  Do not initialize or invoke an 8B Grounder with an empty
            # batch; this keeps no-match latency bounded and avoids turning a
            # legitimate empty retrieval into a model-loading failure.
            if stage_callback:
                stage_callback("POSTPROCESSING", 0.86)
            return {
                "status": "NO_MATCH",
                "predictions": [],
                "candidates": [],
                "degraded": False,
                "degrade_level": 0,
                "degrade_reason": None,
            }
        try:
            predictions = self.grounder.predict(video_path, query, windows)
        except Exception as exc:
            coarse_rows = [
                {"candidate_id": w.candidate_id, "coarse_start": w.start, "coarse_end": w.end, "retrieval_score": w.score, "raw_start": w.start, "raw_end": w.end, "grounding_score": 0.0, "refined_start": w.start, "refined_end": w.end, "final_score": w.score, "duplication_penalty": 0.0, "rank": index + 1, "retriever_version": getattr(self.retriever, "version", "unknown"), "grounder_version": getattr(self.grounder, "model_version", "unknown"), "postprocess_version": "degraded-coarse-v1", "retrieval_latency_ms": retrieval_latency, "grounding_latency_ms": 0.0, "postprocess_latency_ms": 0.0, "degraded": True, "degrade_level": 2, "degrade_reason": f"grounding_error:{type(exc).__name__}", "deduplicated": False}
                for index, w in enumerate(windows)
            ]
            return {
                "status": "POSSIBLE" if windows else "NO_MATCH",
                "predictions": coarse_rows[: self.top_k],
                "candidates": coarse_rows,
                "degraded": True,
                "degrade_level": 2,
                "degrade_reason": f"grounding_error:{type(exc).__name__}",
            }
        by_id = {window.candidate_id: window for window in windows}
        rows = []
        post_started = time.perf_counter()
        for prediction in predictions:
            window = by_id.get(prediction.candidate_id)
            if window is None:
                continue
            degraded = False
            degrade_level = 0
            degrade_reason = None
            try:
                refined = self.boundary_refiner.refine(prediction, duration=duration)
            except Exception as exc:
                refined = {"refined_start": prediction.raw_start, "refined_end": prediction.raw_end, "boundary_refined": False}
                degraded = True
                degrade_level = 1
                degrade_reason = f"boundary_error:{type(exc).__name__}"
            # Ranking receives a duplication signal but remains independent of
            # the subsequent TemporalDeduplicator.  The default weight is 0
            # until a validation set selects a non-zero penalty.
            duplication_penalty = max(
                (self._temporal_iou((window.start, window.end), (other.start, other.end)) for other in windows if other.candidate_id != window.candidate_id),
                default=0.0,
            )
            final_score = self.ranker.score(window.score, prediction, duplication_penalty=duplication_penalty)
            rows.append(build_candidate_record(window, prediction, refined, final_score=final_score, retriever_version=getattr(self.retriever, "version", "unknown"), postprocess_version=f"{self.boundary_refiner.version}+{self.ranker.version}+{self.deduplicator.version}+{self.no_match.version}", retrieval_latency_ms=retrieval_latency, duplication_penalty=duplication_penalty, degraded=degraded, degrade_level=degrade_level, degrade_reason=degrade_reason))
        if stage_callback:
            stage_callback("POSTPROCESSING", 0.86)
        deduped = self.deduplicator.apply(rows)
        post_latency = (time.perf_counter() - post_started) * 1000.0
        kept_ids = {row.get("candidate_id") for row in deduped}
        pre_rank_by_id = {row.get("candidate_id"): rank for rank, row in enumerate(sorted(rows, key=lambda item: item.get("final_score", 0.0), reverse=True), 1)}
        final_rank_by_id = {row.get("candidate_id"): row.get("rank") for row in deduped}
        for row in rows:
            row["postprocess_latency_ms"] = post_latency
            row["deduplicated"] = row.get("candidate_id") not in kept_ids
            row["pre_dedup_rank"] = pre_rank_by_id.get(row.get("candidate_id"))
            row["rank"] = final_rank_by_id.get(row.get("candidate_id"), row["pre_dedup_rank"])
        status = self.no_match.decide(deduped)
        degraded_rows = [row for row in deduped if row.get("degraded")]
        max_level = max((int(row.get("degrade_level", 0)) for row in degraded_rows), default=0)
        reasons = sorted({str(row.get("degrade_reason")) for row in degraded_rows if row.get("degrade_reason")})
        return {
            "status": status,
            # A NO_MATCH decision must not force candidates onto the user
            # surface.  Keep the full lossless rows in ``candidates`` for
            # diagnostics, threshold calibration and badcase analysis.
            "predictions": [] if status == "NO_MATCH" else deduped[: self.top_k],
            # Preserve every ranked row, including candidates removed by
            # TemporalDeduplicator, for lossless diagnostics and Duplicate
            # Rate computation.  The user-facing predictions stay deduped.
            "candidates": rows,
            "degraded": bool(degraded_rows),
            "degrade_level": max_level,
            "degrade_reason": ";".join(reasons) if reasons else None,
        }

    @staticmethod
    def _normalize_windows(video_id: str, windows) -> list[CandidateWindow]:
        """Ensure every coarse window has a stable, unique diagnostic ID."""

        normalized: list[CandidateWindow] = []
        seen: set[str] = set()
        for index, window in enumerate(windows):
            candidate_id = str(window.candidate_id or f"{video_id}:r{index}")
            if candidate_id in seen:
                candidate_id = f"{candidate_id}~{index}"
            seen.add(candidate_id)
            normalized.append(CandidateWindow(float(window.start), float(window.end), float(window.score), candidate_id))
        return normalized

    @staticmethod
    def _temporal_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
        intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
        union = max(left[1], right[1]) - min(left[0], right[0])
        return intersection / union if union > 0 else 0.0
