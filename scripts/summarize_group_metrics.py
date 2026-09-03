#!/usr/bin/env python
"""Summarize baseline metrics by source group and video-level split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from videoitg_smart_clip.evaluation.metrics import evaluate_sample


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"sample_count": 0}
    keys = ["recall_at_1_iou_0.3", "recall_at_1_iou_0.5", "topk_hit_iou_0.3", "topk_hit_iou_0.5", "max_iou_topk", "boundary_error_s_topk"]
    return {"sample_count": len(rows), **{key: sum(float(row[key]) for row in rows if row[key] is not None) / max(1, sum(row[key] is not None for row in rows)) for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--b2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_jsonl(args.manifest)
    sources = {"B0": load_jsonl(args.b0), "B1": load_jsonl(args.b1), "B2": load_jsonl(args.b2)}
    groups = {"source_group": defaultdict(lambda: defaultdict(list)), "split": defaultdict(lambda: defaultdict(list))}
    for sample_id, meta in manifest.items():
        for baseline, rows in sources.items():
            row = rows.get(sample_id)
            if not row:
                continue
            metrics = evaluate_sample(row["predictions"], row["ground_truth_segments"], output_top_k=3)
            for axis, value in (("source_group", meta.get("source_group", "unknown")), ("split", meta.get("split", "unknown"))):
                groups[axis][value][baseline].append(metrics)
    report = {axis: {value: {baseline: aggregate(metrics) for baseline, metrics in sorted(by_baseline.items())} for value, by_baseline in sorted(values.items())} for axis, values in groups.items()}
    report["scope"] = {"manifest": str(args.manifest), "sample_count": len(manifest), "interpretation": "Descriptive strata only; small groups are not evidence of generalization."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source_groups": list(report["source_group"]), "splits": list(report["split"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
