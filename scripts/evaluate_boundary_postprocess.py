#!/usr/bin/env python
"""Evaluate deterministic temporal score post-processors on train/dev only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from videoitg_smart_clip.evaluation.metrics import evaluate_sample


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transformed(scores: list[float], method: str, alpha: float = 0.0, beta: float | None = None) -> list[float]:
    n = len(scores)
    out = []
    for i, score in enumerate(scores):
        left = scores[max(0, i - 1) : i]
        right = scores[i + 1 : min(n, i + 2)]
        neighbors = left + right
        if method == "raw":
            value = score
        elif method == "mean_r1":
            value = sum(scores[max(0, i - 1) : min(n, i + 2)]) / len(scores[max(0, i - 1) : min(n, i + 2)])
        elif method == "mean_r2":
            window = scores[max(0, i - 2) : min(n, i + 3)]
            value = sum(window) / len(window)
        elif method == "neighbor_sum":
            value = score + alpha * sum(neighbors)
        elif method == "neighbor_max":
            value = score + alpha * (max(neighbors) if neighbors else 0.0)
        elif method == "asymmetric_sum":
            right_alpha = alpha if beta is None else beta
            value = score + alpha * (scores[i - 1] if i > 0 else 0.0) + right_alpha * (scores[i + 1] if i + 1 < n else 0.0)
        else:
            raise ValueError(f"unknown method: {method}")
        out.append(value)
    return out


def predictions(row: dict, method: str, alpha: float, margin_threshold: float | None = None, beta: float | None = None) -> list[dict]:
    frames = sorted(row["frame_scores"], key=lambda item: int(item["frame_index"]))
    scores = [float(item["score"]) for item in frames]
    if margin_threshold is not None:
        top_scores = sorted(scores, reverse=True)
        margin = top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 1.0
        adjusted = transformed(scores, method, alpha, beta) if margin < margin_threshold else scores
    else:
        adjusted = transformed(scores, method, alpha, beta)
    order = sorted(range(len(frames)), key=lambda i: (-adjusted[i], int(frames[i]["frame_index"])))
    fps = float(row.get("fps", 0.0))
    # Baseline runs do not persist fps; infer the frame-to-time mapping from the
    # candidate intervals when possible, otherwise use the manifest's 5-second
    # protocol only for score ranking. The caller supplies fps in the dump.
    if fps <= 0:
        raise ValueError("frame-score dump must contain fps")
    duration = float(row["duration_s"])
    segment_seconds = 5.0
    selected = []
    for i in order:
        center = int(frames[i]["frame_index"]) / fps
        start = max(0.0, center - segment_seconds / 2)
        end = min(duration, start + segment_seconds)
        if end <= start:
            continue
        if any(max(start, x["start_s"]) < min(end, x["end_s"]) for x in selected):
            continue
        selected.append({"candidate_id": f"frame_{int(frames[i]['frame_index'])}", "start_s": start, "end_s": end, "score": float(adjusted[i])})
        if len(selected) >= 3:
            break
    return selected


def summary(items: list[dict]) -> dict:
    if not items:
        return {"sample_count": 0}
    keys = ("recall_at_1_iou_0.3", "recall_at_1_iou_0.5", "topk_hit_iou_0.3", "topk_hit_iou_0.5", "max_iou_topk", "boundary_error_s_topk")
    return {"sample_count": len(items), **{key: sum(item[key] for item in items) / len(items) for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.predictions)
    methods = [("raw", 0.0), ("mean_r1", 0.0), ("mean_r2", 0.0)]
    methods.extend(("neighbor_sum", alpha) for alpha in (0.05, 0.1, 0.2, 0.3, 0.5))
    methods.extend(("neighbor_max", alpha) for alpha in (0.05, 0.1, 0.2, 0.3, 0.5))
    reports = []
    for method, alpha in methods:
        by_split: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            pred = predictions(row, method, alpha)
            metrics = evaluate_sample(pred, row["ground_truth_segments"], output_top_k=3)
            by_split[row.get("split", "unknown")].append(metrics)
        reports.append({"method": method, "alpha": alpha, "by_split": {split: summary(items) for split, items in sorted(by_split.items())}})
    report = {"scope": "train_dev_only", "input": str(args.predictions), "candidate_count": len(reports), "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "candidates": len(reports), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
