#!/usr/bin/env python
"""Evaluate a prediction dump on the frozen pilot regression pool."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from videoitg_smart_clip.evaluation.metrics import evaluate_sample


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = load_jsonl(args.pool)
    predictions = load_jsonl(args.predictions)
    per_sample = []
    for sample_id, item in pool.items():
        prediction = predictions.get(sample_id)
        if prediction is None:
            per_sample.append({"sample_id": sample_id, "primary_code": item["primary_code"], "status": "missing_prediction"})
            continue
        metrics = evaluate_sample(prediction["predictions"], item["ground_truth_segments"], output_top_k=3)
        per_sample.append({
            "sample_id": sample_id,
            "primary_code": item["primary_code"],
            "split": item["split"],
            "status": "evaluated",
            "metrics": metrics,
        })

    evaluated = [row for row in per_sample if row["status"] == "evaluated"]
    by_code = defaultdict(list)
    for row in evaluated:
        by_code[row["primary_code"]].append(row["metrics"])

    def summary(rows: list[dict]) -> dict:
        if not rows:
            return {"sample_count": 0}
        return {
            "sample_count": len(rows),
            "recall_at_1_iou_0.3": sum(x["recall_at_1_iou_0.3"] for x in rows) / len(rows),
            "recall_at_1_iou_0.5": sum(x["recall_at_1_iou_0.5"] for x in rows) / len(rows),
            "topk_hit_iou_0.3": sum(x["topk_hit_iou_0.3"] for x in rows) / len(rows),
            "topk_hit_iou_0.5": sum(x["topk_hit_iou_0.5"] for x in rows) / len(rows),
            "max_iou_topk": sum(x["max_iou_topk"] for x in rows) / len(rows),
        }

    report = {
        "pool": str(args.pool),
        "predictions": str(args.predictions),
        "pool_not_for_training": all(item.get("not_for_training") is True for item in pool.values()),
        "status_counts": {"pool_count": len(pool), "evaluated": len(evaluated), "missing_prediction": len(pool) - len(evaluated)},
        "overall": summary([row["metrics"] for row in evaluated]),
        "by_primary_code": {code: summary(metrics) for code, metrics in sorted(by_code.items())},
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status_counts", "overall", "by_primary_code")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
