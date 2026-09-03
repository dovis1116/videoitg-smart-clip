"""Metrics for target-present temporal grounding pilot runs."""

from __future__ import annotations

from typing import Iterable, Sequence


def merge_segments(segments: Iterable[Sequence[float]]) -> list[list[float]]:
    values = sorted([[float(a), float(b)] for a, b in segments if b > a])
    merged: list[list[float]] = []
    for start, end in values:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def gt_segments_from_clip_num(clip_num: Sequence[int], clip_seconds: float = 5.0) -> list[list[float]]:
    return merge_segments((int(c) * clip_seconds, (int(c) + 1) * clip_seconds) for c in clip_num)


def segment_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    right = min(float(a[1]), float(b[1]))
    intersection = max(0.0, right - left)
    union = max(float(a[1]), float(b[1])) - min(float(a[0]), float(b[0]))
    return intersection / union if union > 0 else 0.0


def _prediction_bounds(prediction: dict) -> tuple[float, float] | None:
    """Read either API preview bounds or the lossless pipeline bounds."""
    start = prediction.get("start_s", prediction.get("refined_start", prediction.get("raw_start")))
    end = prediction.get("end_s", prediction.get("refined_end", prediction.get("raw_end")))
    if start is None or end is None:
        return None
    return float(start), float(end)


def best_iou(predictions: Sequence[dict], ground_truth: Sequence[Sequence[float]], top_k: int) -> float:
    values = [segment_iou(bounds, gt) for p in predictions[:top_k] if (bounds := _prediction_bounds(p)) is not None for gt in ground_truth]
    return max(values, default=0.0)


def boundary_error(predictions: Sequence[dict], ground_truth: Sequence[Sequence[float]], top_k: int) -> float | None:
    values = []
    for p in predictions[:top_k]:
        bounds = _prediction_bounds(p)
        if bounds is None:
            continue
        for gt in ground_truth:
            values.append((abs(bounds[0] - gt[0]) + abs(bounds[1] - gt[1])) / 2.0)
    return min(values) if values else None


def evaluate_sample(predictions: Sequence[dict], ground_truth: Sequence[Sequence[float]], output_top_k: int = 3) -> dict:
    # A single best-IoU calculation is thresholded separately below.
    first_bounds = _prediction_bounds(predictions[0]) if predictions else None
    all_top1 = [segment_iou(first_bounds, gt) for gt in ground_truth] if first_bounds is not None else []
    top1_iou = max(all_top1, default=0.0)
    topk_iou = best_iou(predictions, ground_truth, output_top_k)
    per_prediction_iou = [max((segment_iou(bounds, gt) for gt in ground_truth), default=0.0) for p in predictions[:output_top_k] if (bounds := _prediction_bounds(p)) is not None]
    per_prediction_boundary = [min(((abs(bounds[0] - gt[0]) + abs(bounds[1] - gt[1])) / 2.0 for gt in ground_truth), default=0.0) for p in predictions[:output_top_k] if (bounds := _prediction_bounds(p)) is not None]
    return {
        "recall_at_1_iou_0.3": int(top1_iou >= 0.3),
        "recall_at_1_iou_0.5": int(top1_iou >= 0.5),
        "recall_at_1_iou_0.7": int(top1_iou >= 0.7),
        "topk_hit_iou_0.3": int(topk_iou >= 0.3),
        "topk_hit_iou_0.5": int(topk_iou >= 0.5),
        "max_iou_topk": topk_iou,
        "miou": sum(per_prediction_iou) / len(per_prediction_iou) if per_prediction_iou else 0.0,
        "boundary_error_s_topk": boundary_error(predictions, ground_truth, output_top_k),
        "mean_boundary_error_s": sum(per_prediction_boundary) / len(per_prediction_boundary) if per_prediction_boundary else None,
    }


def aggregate_metrics(rows: Sequence[dict]) -> dict:
    keys = [
        "recall_at_1_iou_0.3",
        "recall_at_1_iou_0.5",
        "recall_at_1_iou_0.7",
        "topk_hit_iou_0.3",
        "topk_hit_iou_0.5",
        "max_iou_topk",
        "miou",
        "boundary_error_s_topk",
        "mean_boundary_error_s",
    ]
    result = {key: sum(float(row[key]) for row in rows if row[key] is not None) / max(1, sum(row[key] is not None for row in rows)) for key in keys}
    result["sample_count"] = len(rows)
    return result


def retrieval_recall(windows: Sequence[dict], ground_truth: Sequence[Sequence[float]], top_n: int, iou_threshold: float = 0.3) -> int:
    """Whether any of the first ``top_n`` coarse windows covers the target."""
    return int(any(segment_iou([float(w["start"]), float(w["end"])], gt) >= iou_threshold for w in windows[:top_n] for gt in ground_truth))


def duplicate_rate(rows: Sequence[dict], iou_threshold: float = 0.7) -> float:
    """Fraction of candidates removed by temporal overlap deduplication."""
    if not rows:
        return 0.0
    if any("deduplicated" in row for row in rows):
        return sum(bool(row.get("deduplicated")) for row in rows) / len(rows)
    duplicates = 0
    for index, row in enumerate(rows):
        interval = [row.get("refined_start", row.get("start_s")), row.get("refined_end", row.get("end_s"))]
        if any(segment_iou(interval, [old.get("refined_start", old.get("start_s")), old.get("refined_end", old.get("end_s"))]) > iou_threshold for old in rows[:index]):
            duplicates += 1
    return duplicates / len(rows)


def top_k_useful_rate(rows: Sequence[dict], *, threshold: float = 0.5) -> float | None:
    """Fraction of rows whose Top-K result is useful.

    A row may provide an explicit boolean ``useful`` label (for user review);
    otherwise the metric uses the evaluated Top-K IoU threshold.  Missing rows
    are not converted to zero.
    """

    if not rows:
        return None
    values = []
    for row in rows:
        if row.get("useful") is not None:
            values.append(bool(row["useful"]))
        elif row.get("topk_hit_iou_0.5") is not None:
            values.append(float(row["topk_hit_iou_0.5"]) > 0)
        elif row.get("predictions") is not None and row.get("ground_truth") is not None:
            values.append(best_iou(row["predictions"], row["ground_truth"], int(row.get("output_top_k", 3))) >= threshold)
    return sum(values) / len(values) if values else None


def feedback_product_metrics(rows: Sequence[dict]) -> dict:
    """Compute adoption and manual-boundary adjustment from feedback events."""

    if not rows:
        return {}
    labels = [str(row.get("label", "")).upper() for row in rows if row.get("label") is not None]
    result = {"user_adoption_rate": labels.count("ACCEPT") / len(labels)} if labels else {}
    adjustments = []
    for row in rows:
        bounds = (row.get("model_start"), row.get("model_end"), row.get("user_start"), row.get("user_end"))
        if all(value is not None for value in bounds):
            model_start, model_end, user_start, user_end = (float(value) for value in bounds)
            adjustments.append((abs(user_start - model_start) + abs(user_end - model_end)) / 2.0)
    if adjustments:
        result["mean_manual_boundary_adjustment_seconds"] = sum(adjustments) / len(adjustments)
    return result


def no_match_metrics(predicted: Sequence[str], actual: Sequence[bool]) -> dict:
    """Accuracy/FPR/FNR for explicit no-match decisions.

    ``actual`` is a match-presence label: ``True`` means the query event is
    present and ``False`` means the sample is a genuine no-match.  This keeps
    the reported rates unambiguous: FPR is rejecting a present event, while FNR
    is accepting a no-match sample.
    """
    if len(predicted) != len(actual) or not actual:
        raise ValueError("predicted and actual must have equal non-zero length")
    predicted_no_match = [value == "NO_MATCH" for value in predicted]
    present = sum(actual)
    absent = len(actual) - present
    false_positive = sum(pred and truth for pred, truth in zip(predicted_no_match, actual))
    false_negative = sum((not pred) and (not truth) for pred, truth in zip(predicted_no_match, actual))
    return {
        "no_match_accuracy": sum(pred == (not truth) for pred, truth in zip(predicted_no_match, actual)) / len(actual),
        "false_positive_rate": false_positive / max(1, present),
        "false_negative_rate": false_negative / max(1, absent),
    }


def calibrate_no_match_thresholds(samples: Sequence[dict]) -> dict:
    """Choose retrieval/grounding rejection thresholds on a validation set.

    Each sample must contain ``retrieval_score``, ``grounding_score`` and
    ``actual_match``.  The returned thresholds are validation artifacts only;
    callers must persist the data/config version before using them in service.
    """

    if not samples or {bool(row.get("actual_match")) for row in samples} != {True, False}:
        raise ValueError("validation samples must contain both match and no-match labels")
    retrieval_values = sorted({0.0, 1.0, *[float(row["retrieval_score"]) for row in samples]})
    grounding_values = sorted({0.0, 1.0, *[float(row["grounding_score"]) for row in samples]})
    best = None
    for retrieval_threshold in retrieval_values:
        for grounding_threshold in grounding_values:
            predicted = [
                "NO_MATCH" if float(row["retrieval_score"]) < retrieval_threshold or float(row["grounding_score"]) < grounding_threshold else "CONFIDENT"
                for row in samples
            ]
            metrics = no_match_metrics(predicted, [bool(row["actual_match"]) for row in samples])
            key = (-metrics["no_match_accuracy"], metrics["false_positive_rate"] + metrics["false_negative_rate"], retrieval_threshold, grounding_threshold)
            if best is None or key < best[0]:
                best = (key, retrieval_threshold, grounding_threshold, metrics)
    assert best is not None
    _, retrieval_threshold, grounding_threshold, metrics = best
    return {"retrieval_threshold": retrieval_threshold, "grounding_threshold": grounding_threshold, "metrics": metrics, "validation_count": len(samples)}
