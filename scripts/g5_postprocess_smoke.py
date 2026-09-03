#!/usr/bin/env python3
"""Contract smoke for independent G5 post-processing stages.

This fixture verifies stage wiring and lossless fields only.  It is not a
quality claim: No-Match thresholds must still be calibrated on a labeled
validation set containing the required negative categories.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from videoitg_smart_clip.evaluation.metrics import no_match_metrics
from videoitg_smart_clip.pipeline import CandidateWindow, GroundingPrediction
from videoitg_smart_clip.pipeline.postprocess import (
    BoundaryRefiner,
    CandidateRanker,
    NoMatchDecider,
    TemporalDeduplicator,
    build_candidate_record,
)


def run(boundary_enabled: bool) -> dict:
    windows = [
        CandidateWindow(10.0, 20.0, 0.90, "a"),
        CandidateWindow(11.0, 20.5, 0.85, "b"),
        CandidateWindow(40.0, 50.0, 0.35, "c"),
    ]
    predictions = [
        GroundingPrediction("a", 12.0, 18.0, 0.92, 12.0, "TimeLens-8B", 0.90, 0.95),
        GroundingPrediction("b", 12.2, 18.1, 0.80, 12.0, "TimeLens-8B", 0.70, 0.90),
        GroundingPrediction("c", 42.0, 43.0, 0.20, 12.0, "TimeLens-8B", 0.30, 0.40),
    ]
    refiner = BoundaryRefiner(enabled=boundary_enabled, expansion_seconds=0.25)
    ranker = CandidateRanker()
    dedup = TemporalDeduplicator(temporal_iou_threshold=0.7)
    started = time.perf_counter()
    rows = []
    for window, prediction in zip(windows, predictions):
        refined = refiner.refine(prediction)
        rows.append(
            build_candidate_record(
                window,
                prediction,
                refined,
                final_score=ranker.score(window.score, prediction),
                retriever_version="cached-cosine-v1",
                postprocess_version=f"{refiner.version}+{ranker.version}+{dedup.version}",
            )
        )
    kept = dedup.apply(rows)
    dedup_rate = (len(rows) - len(kept)) / len(rows)
    no_match = NoMatchDecider(retrieval_threshold=0.5, grounding_threshold=0.5, margin_threshold=0.2)
    statuses = [
        no_match.decide([{"final_score": 0.82, "retrieval_score": 0.82, "grounding_score": 0.9}, {"final_score": 0.3, "retrieval_score": 0.3, "grounding_score": 0.7}]),
        no_match.decide([{"final_score": 0.65, "retrieval_score": 0.65, "grounding_score": 0.8}, {"final_score": 0.60, "retrieval_score": 0.60, "grounding_score": 0.8}]),
        no_match.decide([{"final_score": 0.2, "retrieval_score": 0.2, "grounding_score": 0.2}]),
    ]
    return {
        "boundary_enabled": boundary_enabled,
        "candidate_count_before_dedup": len(rows),
        "candidate_count_after_dedup": len(kept),
        "duplicate_rate": dedup_rate,
        "raw_fields_present": all("raw_start" in row and "raw_end" in row for row in rows),
        "boundary_changed_when_enabled": any(row["raw_start"] != row["refined_start"] for row in rows) if boundary_enabled else False,
        "statuses": statuses,
        "no_match_metrics": no_match_metrics(statuses, [True, True, False]),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "records": kept,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "records/phase_g5/g5_postprocess_smoke.json")
    args = parser.parse_args()
    result = {"run_id": f"g5_postprocess_smoke_{time.strftime('%Y%m%d_%H%M%S')}", "enabled": run(True), "disabled": run(False), "scope": "synthetic_contract_smoke_not_quality_evidence"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
