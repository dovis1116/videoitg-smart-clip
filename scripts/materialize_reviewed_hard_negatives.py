#!/usr/bin/env python
"""Materialize only reviewed temporal hard negatives; keep them out of training by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_rows = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = {row["candidate_id"]: row for row in candidate_rows}
    review = {row["candidate_id"]: row for row in json.loads(args.review.read_text(encoding="utf-8"))["rows"]}
    allowed = {"confirmed_adjacent_non_target", "confirmed_boundary_hard_negative"}
    rows = []
    for candidate_id, item in candidates.items():
        verdict = review.get(candidate_id)
        if not verdict or verdict["status"] not in allowed:
            continue
        rows.append({**item, "review_status": verdict["status"], "review_rationale": verdict["rationale"], "materialized_for": "phase4_prototype_only", "not_for_training": True})
    rows.sort(key=lambda row: row["candidate_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "not_for_training": all(row["not_for_training"] for row in rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
