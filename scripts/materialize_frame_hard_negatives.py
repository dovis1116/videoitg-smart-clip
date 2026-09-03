#!/usr/bin/env python
"""Materialize reviewed B1 wrong-peak predictions as frame-level diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    output = []
    for row in rows:
        gt = row["ground_truth_segments"]
        candidate = next(
            (prediction for prediction in row["predictions"] if max(iou([prediction["start_s"], prediction["end_s"]], target) for target in gt) < 0.3),
            None,
        )
        if candidate is None:
            continue
        output.append({
            "candidate_id": f"{row['sample_id']}:b1_wrong_peak",
            "sample_id": row["sample_id"],
            "split": row.get("split", "train"),
            "video_path": row["video_path"],
            "query": row["query"],
            "ground_truth_segments": gt,
            "candidate_segment": [candidate["start_s"], candidate["end_s"]],
            "candidate_score": candidate["score"],
            "candidate_source": "B1 full-video frame peak",
            "source_badcase": row.get("primary_code"),
            "review_status": "manual_confirmed_sampled_2s_or_provisional",
            "not_for_training": True,
            "purpose": "frame-level wrong-peak diagnostic; not a semantic no-target label",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + ("\n" if output else ""), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(output), "not_for_training": all(item["not_for_training"] for item in output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
