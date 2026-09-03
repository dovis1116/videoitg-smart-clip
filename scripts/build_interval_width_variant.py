#!/usr/bin/env python
"""Offline remap of frozen frame-peak predictions to another interval width."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from videoitg_smart_clip.evaluation.metrics import aggregate_metrics, evaluate_sample


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    from decord import VideoReader, cpu

    predictions = load_jsonl(args.predictions)
    manifest = load_jsonl(args.manifest)
    results = []
    for sample_id, row in predictions.items():
        item = manifest[sample_id]
        reader = VideoReader(item["video_path"], ctx=cpu(0), num_threads=1)
        duration = len(reader) / float(reader.get_avg_fps())
        remapped = []
        for prediction in row["predictions"]:
            center = (float(prediction["start_s"]) + float(prediction["end_s"])) / 2.0
            start = max(0.0, center - args.width / 2.0)
            end = min(duration, start + args.width)
            if end <= start:
                continue
            remapped.append({**prediction, "start_s": start, "end_s": end, "interval_width_s": args.width})
        metrics = evaluate_sample(remapped, row["ground_truth_segments"], output_top_k=3)
        results.append({**row, "baseline": f"{row.get('baseline', 'B1')}_width{args.width:g}", "predictions": remapped, "metrics": metrics, "interval_variant": {"width_s": args.width, "source_predictions": str(args.predictions)}})
    results.sort(key=lambda row: row["sample_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")
    summary = {"source_predictions": str(args.predictions), "manifest": str(args.manifest), "width_s": args.width, "sample_count": len(results), "metrics": aggregate_metrics([row["metrics"] for row in results]), "output": str(args.output)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
