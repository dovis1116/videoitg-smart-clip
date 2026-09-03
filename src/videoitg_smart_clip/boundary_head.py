"""Small, deterministic boundary-aware score calibrator.

The calibrator is intentionally a CPU-side post-processor.  It is fit only
from rows whose ground-truth intervals are explicitly supplied, then applied
to frozen VideoITG frame scores.  It does not change the VideoITG checkpoint or
the number of model calls.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from videoitg_smart_clip.evaluation.metrics import segment_iou


FEATURE_NAMES = (
    "raw_score",
    "row_z_score",
    "left_score",
    "right_score",
    "neighbor_sum",
    "left_minus_right",
    "local_max",
    "local_mean",
    "rank_fraction",
    "position_fraction",
    "position_centered",
)


def sorted_frames(row: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(row["frame_scores"], key=lambda item: int(item["frame_index"]))


def frame_feature_matrix(row: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return one feature vector per sampled frame in temporal order."""

    frames = sorted_frames(row)
    scores = np.asarray([float(item["score"]) for item in frames], dtype=np.float64)
    if scores.size == 0:
        raise ValueError("frame-score row is empty")
    n = scores.size
    row_z = (scores - scores.mean()) / (scores.std() + 1e-6)
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    rank_fraction = 1.0 - ranks / max(1, n - 1)
    position = np.arange(n, dtype=np.float64) / max(1, n - 1)
    values: list[list[float]] = []
    for i, score in enumerate(scores):
        left = scores[i - 1] if i > 0 else scores[0]
        right = scores[i + 1] if i + 1 < n else scores[-1]
        window = scores[max(0, i - 1) : min(n, i + 2)]
        values.append(
            [
                float(score),
                float(row_z[i]),
                float(left),
                float(right),
                float(left + right),
                float(left - right),
                float(window.max()),
                float(window.mean()),
                float(rank_fraction[i]),
                float(position[i]),
                float(position[i] * (1.0 - position[i])),
            ]
        )
    return np.asarray(values, dtype=np.float64), frames


def candidate_interval(row: dict[str, Any], frame: dict[str, Any], segment_seconds: float = 5.0) -> list[float]:
    fps = float(row.get("fps", 0.0))
    duration = float(row.get("duration_s", 0.0))
    if fps <= 0 or duration <= 0:
        raise ValueError("boundary-head input must contain positive fps and duration_s")
    center = int(frame["frame_index"]) / fps
    start = max(0.0, center - segment_seconds / 2.0)
    end = min(duration, start + segment_seconds)
    return [start, end]


def fit_boundary_head(
    rows: Sequence[dict[str, Any]],
    ridge: float = 0.1,
    positive_iou: float = 0.5,
    segment_seconds: float = 5.0,
) -> dict[str, Any]:
    """Fit a ridge linear scorer against candidate-level IoU labels."""

    if ridge <= 0:
        raise ValueError("ridge must be positive")
    matrices = []
    labels: list[float] = []
    positive_count = 0
    for row in rows:
        matrix, frames = frame_feature_matrix(row)
        matrices.append(matrix)
        for frame in frames:
            candidate = candidate_interval(row, frame, segment_seconds)
            label = float(
                max(
                    (segment_iou(candidate, gt) for gt in row.get("ground_truth_segments", [])),
                    default=0.0,
                )
                >= positive_iou
            )
            labels.append(label)
            positive_count += int(label)
    if not matrices:
        raise ValueError("cannot fit boundary head on zero rows")
    raw = np.vstack(matrices)
    y = np.asarray(labels, dtype=np.float64)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (raw - mean) / scale
    gram = normalized.T @ normalized + ridge * np.eye(normalized.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, normalized.T @ y)
    return {
        "format": "videoitg_boundary_head_v1",
        "feature_names": list(FEATURE_NAMES),
        "segment_seconds": float(segment_seconds),
        "positive_iou": float(positive_iou),
        "ridge": float(ridge),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": float(y.mean()),
        "fit_rows": len(rows),
        "fit_frame_count": int(raw.shape[0]),
        "positive_frame_count": positive_count,
    }


def calibrated_scores(row: dict[str, Any], model: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if model.get("format") != "videoitg_boundary_head_v1":
        raise ValueError("unsupported boundary-head model format")
    matrix, frames = frame_feature_matrix(row)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if matrix.shape[1] != len(weights) or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("boundary-head feature dimension mismatch")
    values = ((matrix - mean) / scale) @ weights + float(model.get("bias", 0.0))
    return values, frames


def calibrated_predictions(row: dict[str, Any], model: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    values, frames = calibrated_scores(row, model)
    order = sorted(range(len(frames)), key=lambda i: (-float(values[i]), int(frames[i]["frame_index"])))
    selected: list[dict[str, Any]] = []
    segment_seconds = float(model.get("segment_seconds", 5.0))
    for index in order:
        start, end = candidate_interval(row, frames[index], segment_seconds)
        if end <= start:
            continue
        if any(max(start, item["start_s"]) < min(end, item["end_s"]) for item in selected):
            continue
        selected.append(
            {
                "candidate_id": f"frame_{int(frames[index]['frame_index'])}",
                "start_s": start,
                "end_s": end,
                "score": float(values[index]),
            }
        )
        if len(selected) >= top_k:
            break
    return selected
