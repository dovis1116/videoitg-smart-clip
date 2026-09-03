#!/usr/bin/env python
"""Materialize confirmed provisional badcases into a frozen pilot regression pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--evidence-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion-status", default="confirmed_keyframe")
    parser.add_argument("--evaluation-role", default="pilot_regression_only")
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence_index.read_text(encoding="utf-8"))
    evidence_by_id = {row["sample_id"]: row for row in evidence["rows"]}
    rows = []
    for item in review["samples"]:
        if item.get("promotion_status") != args.promotion_status:
            continue
        source = evidence_by_id[item["sample_id"]]
        rows.append({
            "sample_id": item["sample_id"],
            "video_id": source["video_id"],
            "split": source["split"],
            "video_path": source["video_path"],
            "query": source["query"],
            "answer": source.get("answer"),
            "ground_truth_segments": source["ground_truth_segments"],
            "predictions": source["predictions"],
            "primary_code": item["primary_code"],
            "severity": item["severity"],
            "confidence": item["confidence"],
            "rationale": item["rationale"],
            "source_evidence": review["source_evidence"],
            "evaluation_role": args.evaluation_role,
            "not_for_training": True,
            "full_video_replay_status": "pending",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "codes": sorted({r['primary_code'] for r in rows})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
