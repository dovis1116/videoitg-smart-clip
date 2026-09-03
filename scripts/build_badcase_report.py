#!/usr/bin/env python
"""Build an evidence-only badcase distribution from baseline prediction dumps."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def top1_iou(row: dict) -> float:
    predictions = row.get("predictions", [])[:1]
    ground_truth = row.get("ground_truth_segments", [])
    values = []
    for prediction in predictions:
        for gt in ground_truth:
            left = max(float(prediction["start_s"]), float(gt[0]))
            right = min(float(prediction["end_s"]), float(gt[1]))
            intersection = max(0.0, right - left)
            union = max(float(prediction["end_s"]), float(gt[1])) - min(float(prediction["start_s"]), float(gt[0]))
            values.append(intersection / union if union > 0 else 0.0)
    return max(values, default=0.0)


def max_iou(row: dict) -> float:
    return float(row["metrics"].get("max_iou_topk", 0.0))


def classify(row: dict, b2r: dict | None, b2: dict | None) -> str:
    value = top1_iou(row)
    if value >= 0.5:
        category = "correct_iou_ge_0_5"
    elif value >= 0.3:
        category = "near_miss_iou_0_3_to_0_5"
    else:
        category = "weak_overlap_iou_lt_0_3"
    if b2r and b2:
        retrieval_hit = max_iou(b2r) >= 0.3
        rerank_hit = max_iou(b2) >= 0.3
        if retrieval_hit and not rerank_hit:
            category += "+rerank_regression"
    return category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--b2r", type=Path)
    parser.add_argument("--b2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b1 = load_rows(args.b1)
    b2r = load_rows(args.b2r) if args.b2r else {}
    b2 = load_rows(args.b2) if args.b2 else {}
    rows = []
    counts = Counter()
    for sample_id, row in sorted(b1.items()):
        category = classify(row, b2r.get(sample_id), b2.get(sample_id))
        counts[category] += 1
        rows.append({
            "sample_id": sample_id,
            "video_id": row.get("video_id"),
            "split": row.get("split"),
            "query": row.get("query"),
            "b1_top1_iou": top1_iou(row),
            "b1_max_iou_topk": max_iou(row),
            "category": category,
            "b2r_max_iou_topk": max_iou(b2r[sample_id]) if sample_id in b2r else None,
            "b2_max_iou_topk": max_iou(b2[sample_id]) if sample_id in b2 else None,
        })
    report = {
        "scope": {
            "source": str(args.b1),
            "sample_count": len(rows),
            "semantic_labels": "not assigned; categories are metric-derived only",
        },
        "counts": dict(sorted(counts.items())),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": report["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
