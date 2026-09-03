#!/usr/bin/env python3
"""Run lossless post-processing ablations on existing prediction JSONL.

This tool evaluates behavior already represented in prediction artifacts. It
does not run a grounder and does not infer missing raw/refined boundaries.
When a legacy artifact lacks a field, the report marks the corresponding
boundary as a preview-bound proxy.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from videoitg_smart_clip.evaluation.metrics import aggregate_metrics, duplicate_rate, evaluate_sample, segment_iou


def _bounds(prediction: dict, source: str) -> tuple[float, float] | None:
    if source == "raw":
        start = prediction.get("raw_start")
        end = prediction.get("raw_end")
    else:
        start = prediction.get("refined_start")
        end = prediction.get("refined_end")
    if start is None or end is None:
        start = prediction.get("start_s")
        end = prediction.get("end_s")
    if start is None or end is None:
        return None
    return float(start), float(end)


def _with_boundary(prediction: dict, source: str) -> dict:
    item = copy.deepcopy(prediction)
    bounds = _bounds(item, source)
    if bounds is not None:
        item["start_s"], item["end_s"] = bounds
    return item


def _rank(predictions: list[dict], key: str) -> list[dict]:
    if key == "recorded":
        score = lambda item: float(item.get("final_score", item.get("grounding_score", 0.0)))
    elif key == "retrieval":
        score = lambda item: float(item.get("retrieval_score", item.get("score", 0.0)))
    elif key == "grounding":
        score = lambda item: float(item.get("grounding_score", 0.0))
    else:
        raise ValueError(f"unknown ranking key: {key}")
    return sorted(predictions, key=lambda item: (-score(item), str(item.get("candidate_id", ""))))


def _dedup(predictions: list[dict], threshold: float) -> list[dict]:
    kept: list[dict] = []
    for item in _rank(predictions, "recorded"):
        bounds = _bounds(item, "refined")
        if bounds is None:
            kept.append(item)
            continue
        if any(
            (old_bounds := _bounds(old, "refined")) is not None
            and segment_iou(bounds, old_bounds) > threshold
            for old in kept
        ):
            continue
        kept.append(item)
    return kept


def _evaluate(rows: list[dict], predictions_by_row: list[list[dict]], *, variant: str, metadata: dict, dedup_threshold: float) -> dict:
    evaluated = []
    duplicate_values = []
    for row, predictions in zip(rows, predictions_by_row):
        evaluated.append(evaluate_sample(predictions, row.get("ground_truth", []), int(row.get("output_top_k", 3))))
        duplicate_values.append(duplicate_rate(predictions, dedup_threshold))
    result = aggregate_metrics(evaluated) if evaluated else {"sample_count": 0}
    result.update({
        "variant": variant,
        "duplicate_rate_mean": sum(duplicate_values) / len(duplicate_values) if duplicate_values else None,
        "metadata": metadata,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dedup-threshold", type=float, default=0.7)
    args = parser.parse_args()
    if not 0.0 <= args.dedup_threshold <= 1.0:
        raise SystemExit("--dedup-threshold must be in [0, 1]")
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    exact_raw = sum(all(p.get("raw_start") is not None and p.get("raw_end") is not None for p in row.get("predictions", [])) for row in rows)
    exact_refined = sum(all(p.get("refined_start") is not None and p.get("refined_end") is not None for p in row.get("predictions", [])) for row in rows)
    variants: list[dict] = []

    for source in ("raw", "refined"):
        predictions = [[_with_boundary(p, source) for p in row.get("predictions", [])] for row in rows]
        variants.append(_evaluate(rows, predictions, variant=f"boundary_{source}", dedup_threshold=args.dedup_threshold, metadata={
            "boundary_source": source,
            "exact_field_rows": exact_raw if source == "raw" else exact_refined,
            "fallback_to_preview_bounds": True,
        }))

    for key in ("recorded", "retrieval", "grounding"):
        predictions = [_rank([_with_boundary(p, "refined") for p in row.get("predictions", [])], key) for row in rows]
        variants.append(_evaluate(rows, predictions, variant=f"ranking_{key}", dedup_threshold=args.dedup_threshold, metadata={
            "ranking_key": key,
            "boundary_source": "refined_or_preview_proxy",
        }))

    no_dedup = [[_with_boundary(p, "refined") for p in row.get("predictions", [])] for row in rows]
    with_dedup = [[_with_boundary(p, "refined") for p in _dedup(row.get("predictions", []), args.dedup_threshold)] for row in rows]
    variants.append(_evaluate(rows, no_dedup, variant="dedup_off", dedup_threshold=args.dedup_threshold, metadata={"threshold": args.dedup_threshold}))
    variants.append(_evaluate(rows, with_dedup, variant="dedup_on", dedup_threshold=args.dedup_threshold, metadata={"threshold": args.dedup_threshold}))

    report = {
        "run_id": f"postprocess_ablation_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}",
        "input": str(args.input),
        "sample_count": len(rows),
        "prediction_count": sum(len(row.get("predictions", [])) for row in rows),
        "scope": "existing_timelens_predictions_diagnostic_not_quality_gate",
        "raw_boundary_exact_rows": exact_raw,
        "refined_boundary_exact_rows": exact_refined,
        "dedup_threshold": args.dedup_threshold,
        "variants": variants,
        "limitations": [
            "No grounder inference is run; this is a post-processing-only replay.",
            "Legacy pilot rows without raw_start/raw_end use start_s/end_s as an explicit preview-bound proxy.",
            "The input contains target-present diagnostic pilot rows only and cannot calibrate No-Match thresholds.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "run_id": report["run_id"],
        "sample_count": report["sample_count"],
        "prediction_count": report["prediction_count"],
        "raw_boundary_exact_rows": report["raw_boundary_exact_rows"],
        "refined_boundary_exact_rows": report["refined_boundary_exact_rows"],
        "variants": [{key: item[key] for key in ("variant", "recall_at_1_iou_0.3", "recall_at_1_iou_0.5", "max_iou_topk", "miou", "boundary_error_s_topk", "duplicate_rate_mean")} for item in variants],
    }, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
