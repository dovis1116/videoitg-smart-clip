#!/usr/bin/env python
"""Build review-only same-video temporal hard-negative candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def overlap(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    args = parser.parse_args()

    pool_rows = [json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    pool = {row["sample_id"]: row for row in pool_rows}
    review = json.loads(args.review.read_text(encoding="utf-8"))
    rows = []
    for item in review["samples"]:
        if item.get("promotion_status") != "confirmed_sampled_full_video":
            continue
        source = pool[item["sample_id"]]
        gt_segments = [[float(x[0]), float(x[1])] for x in source["ground_truth_segments"]]
        duration = max(float(p["end_s"]) for p in source.get("predictions", []))
        # The pool does not carry duration; extend from the video only when needed.
        try:
            from decord import VideoReader, cpu

            reader = VideoReader(source["video_path"], ctx=cpu(0), num_threads=1)
            duration = len(reader) / float(reader.get_avg_fps())
        except Exception:
            pass
        for label, start, end in (
            ("before_gt", gt_segments[0][0] - args.window_seconds, gt_segments[0][0]),
            ("after_gt", gt_segments[0][1], gt_segments[0][1] + args.window_seconds),
        ):
            start, end = max(0.0, start), min(duration, end)
            if end - start < args.window_seconds * 0.8:
                continue
            candidate = [start, end]
            if any(overlap(candidate, gt) > 0 for gt in gt_segments):
                continue
            rows.append({
                "candidate_id": f"{source['sample_id']}:{label}",
                "sample_id": source["sample_id"],
                "split": source.get("split", "train"),
                "video_path": source["video_path"],
                "query": source["query"],
                "ground_truth_segments": gt_segments,
                "candidate_segment": candidate,
                "candidate_role": "same_video_temporal_hard_negative",
                "source_badcase": item["primary_code"],
                "label_status": "pending_manual",
                "not_for_training": True,
                "rationale": "Adjacent interval generated from a confirmed train-side badcase; must be visually reviewed before any training use.",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "not_for_training": all(r["not_for_training"] for r in rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
