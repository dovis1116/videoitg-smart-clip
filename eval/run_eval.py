#!/usr/bin/env python3
"""Small, deterministic evaluation entry point for the migrated pipeline.

It consumes JSONL rows containing ``ground_truth`` and ``predictions``. Model
execution stays outside evaluation so Retriever/Grounder experiments remain
reproducible and independently comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoitg_smart_clip.evaluation.metrics import calibrate_no_match_thresholds, evaluate_sample, aggregate_metrics, duplicate_rate, feedback_product_metrics, no_match_metrics, retrieval_recall, top_k_useful_rate
from videoitg_smart_clip.badcase.taxonomy import normalize_badcase_type


def _load_eval_config(path: Path) -> dict:
    """Load and validate the evaluation subsection from the YAML contract."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("PyYAML is required to read the evaluation config") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must be a YAML mapping")
    config = payload.get("evaluation", payload)
    if not isinstance(config, dict):
        raise ValueError("evaluation config section must be a YAML mapping")
    output_top_k = int(config.get("output_top_k", 3))
    duplicate_iou_threshold = float(config.get("duplicate_iou_threshold", 0.7))
    retriever_top_n = tuple(int(value) for value in config.get("retriever_top_n", (5, 10, 20)))
    if output_top_k <= 0:
        raise ValueError("evaluation.output_top_k must be positive")
    if not 0.0 <= duplicate_iou_threshold <= 1.0:
        raise ValueError("evaluation.duplicate_iou_threshold must be in [0, 1]")
    if not retriever_top_n or any(value <= 0 for value in retriever_top_n):
        raise ValueError("evaluation.retriever_top_n must contain positive integers")
    return {
        "output_top_k": output_top_k,
        "duplicate_iou_threshold": duplicate_iou_threshold,
        "retriever_top_n": list(retriever_top_n),
        "data_version": config.get("data_version"),
        "split_version": config.get("split_version"),
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
    return values[index]


def _calibration_samples(rows: list[dict]) -> list[dict]:
    """Build threshold-calibration rows without dropping explicit NO_MATCH outputs.

    A correct no-match result commonly has ``predictions=[]`` while retaining
    coarse/lossless candidates for diagnostics.  Such rows still carry useful
    retrieval evidence; omitting them would bias the validation threshold
    search toward positive samples.
    """

    samples = []
    for row in rows:
        if row.get("actual_match") is None:
            continue
        predictions = row.get("predictions") or []
        candidates = row.get("candidates") or row.get("coarse_windows") or []
        first = predictions[0] if predictions else (candidates[0] if candidates else {})
        retrieval_score = first.get("retrieval_score", first.get("score"))
        if retrieval_score is None:
            retrieval_score = 0.0
        grounding_score = first.get("grounding_score", 0.0) if predictions else 0.0
        samples.append({
            "retrieval_score": float(retrieval_score),
            "grounding_score": float(grounding_score),
            "actual_match": bool(row["actual_match"]),
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None, help="JSONL predictions; omit for a pending empty report")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    args = parser.parse_args()
    eval_config = _load_eval_config(args.config)
    rows = [] if args.input is None else [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluated = []
    retrieval = {n: [] for n in eval_config["retriever_top_n"]}
    for row in rows:
        predictions = row.get("predictions", [])
        gt = row.get("ground_truth", [])
        evaluated.append(evaluate_sample(predictions, gt, int(row.get("output_top_k", eval_config["output_top_k"]))))
        for n in retrieval:
            retrieval[n].append(retrieval_recall(row.get("coarse_windows", []), gt, n))
    if evaluated:
        metrics = aggregate_metrics(evaluated)
    else:
        # Preserve the complete metric schema for an empty/pending run.  A
        # missing key would be ambiguous to downstream dashboards; null means
        # the metric was not measured yet.
        metrics = {
            key: None
            for key in (
                "recall_at_1_iou_0.3", "recall_at_1_iou_0.5", "recall_at_1_iou_0.7",
                "topk_hit_iou_0.3", "topk_hit_iou_0.5", "max_iou_topk", "miou",
                "boundary_error_s_topk", "mean_boundary_error_s",
            )
        }
        metrics["sample_count"] = 0
    metrics["evaluation_config"] = eval_config
    metrics["retrieval_recall"] = {
        f"recall_at_{n}": (sum(values) / len(values) if values else None)
        for n, values in retrieval.items()
    }
    metrics["duplicate_rate"] = (
        sum(duplicate_rate(row.get("candidates", row.get("predictions", [])), eval_config["duplicate_iou_threshold"]) for row in rows) / len(rows)
        if rows else None
    )
    useful_rows = []
    for row, score in zip(rows, evaluated):
        merged = dict(row)
        merged.update(score)
        useful_rows.append(merged)
    useful = top_k_useful_rate(useful_rows)
    metrics["top_k_useful_rate"] = useful
    feedback_metrics = feedback_product_metrics(rows)
    metrics.update(feedback_metrics)
    # No-match metrics are emitted only when the input carries validation
    # labels.  ``actual_match`` follows the metrics contract: True means the
    # event exists, False means a genuine negative sample.
    labeled = [(row.get("status"), row.get("actual_match")) for row in rows if row.get("actual_match") is not None and row.get("status")]
    # Accuracy/FPR/FNR are only defined for a validation input containing
    # both match-presence classes.  A target-present-only pilot must not look
    # like a calibrated No-Match evaluation merely because all of its
    # decisions accepted an event.
    if labeled and {bool(actual) for _, actual in labeled} == {True, False}:
        metrics.update(no_match_metrics([status for status, _ in labeled], [bool(actual) for _, actual in labeled]))
        calibration_samples = _calibration_samples(rows)
        if calibration_samples and {bool(sample["actual_match"]) for sample in calibration_samples} == {True, False}:
            metrics["no_match_calibration"] = calibrate_no_match_thresholds(calibration_samples)
    for key in ("no_match_accuracy", "false_positive_rate", "false_negative_rate"):
        metrics.setdefault(key, None)
    # Optional engineering measurements are aggregated when present; missing
    # fields remain explicit nulls instead of being reported as zero.
    for key in ("decode_latency_ms", "feature_extraction_latency_ms", "retrieval_latency_ms", "grounding_latency_ms", "postprocess_latency_ms", "end_to_end_latency_ms", "gpu_memory_gib"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            metrics[f"{key}_mean"] = sum(values) / len(values)
            metrics[f"{key}_p50"] = _percentile(values, 0.50)
            metrics[f"{key}_p95"] = _percentile(values, 0.95)
        else:
            metrics[f"{key}_mean"] = None
            metrics[f"{key}_p50"] = None
            metrics[f"{key}_p95"] = None
    for key in ("timeout", "failed", "degraded"):
        values = [bool(row[key]) for row in rows if row.get(key) is not None]
        if values:
            metrics[f"{key}_rate"] = sum(values) / len(values)
        else:
            metrics[f"{key}_rate"] = None
    metrics["failure_rate"] = metrics["failed_rate"]
    metrics["degrade_rate"] = metrics["degraded_rate"]
    metrics.setdefault("user_adoption_rate", None)
    metrics.setdefault("mean_manual_boundary_adjustment_seconds", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    badcases = []
    for row, score in zip(rows, evaluated):
        if score.get("topk_hit_iou_0.5", 0) == 0:
            item = dict(row)
            item["badcase_type"] = normalize_badcase_type(row.get("badcase_type"))
            badcases.append(item)
    (args.output_dir / "badcases.json").write_text(json.dumps(badcases, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text("# Evaluation report\n\n```json\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
