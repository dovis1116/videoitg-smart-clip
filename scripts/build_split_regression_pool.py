#!/usr/bin/env python
"""Build a split-level regression pool with badcase rows and metric controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_jsonl(args.manifest)
    predictions = load_jsonl(args.predictions)
    review = json.loads(args.review.read_text(encoding="utf-8"))
    reviewed = {row["sample_id"]: row for row in review["samples"]}
    rows = []
    for sample_id, item in manifest.items():
        if item.get("split") != args.split or sample_id not in predictions:
            continue
        prediction = predictions[sample_id]
        manual = reviewed.get(sample_id)
        if manual:
            code = manual["primary_code"]
            review_status = "manual_confirmed_sampled_2s" if manual.get("promotion_status") == "confirmed_sampled_full_video" else "provisional_manual"
            rationale = manual["rationale"]
        else:
            code = "control_correct" if prediction["metrics"]["recall_at_1_iou_0.3"] else "metric_unreviewed"
            review_status = "metric_control" if code == "control_correct" else "metric_unreviewed"
            rationale = "Included as a split-level control based on the frozen baseline metric; no semantic error label assigned."
        rows.append({
            "sample_id": sample_id,
            "video_id": item["video_id"],
            "split": args.split,
            "video_path": item["video_path"],
            "query": item["query"],
            "answer": item.get("answer"),
            "ground_truth_segments": prediction["ground_truth_segments"],
            "predictions": prediction["predictions"],
            "primary_code": code,
            "review_status": review_status,
            "rationale": rationale,
            "not_for_training": True,
            "evaluation_role": "heldout_split_regression",
            "source_predictions": str(args.predictions),
            "full_video_replay_status": "sampled_2s_full_video_scan_complete" if manual else "control_no_semantic_review",
        })
    rows.sort(key=lambda row: row["sample_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "codes": {code: sum(r['primary_code'] == code for r in rows) for code in sorted({r['primary_code'] for r in rows})}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
