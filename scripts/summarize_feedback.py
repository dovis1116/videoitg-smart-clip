"""Summarize versioned feedback events without exposing query/video content."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = Counter(row["label"] for row in rows)
    adjusted = [
        row for row in rows
        if row.get("adjusted_start_s") is not None or row.get("adjusted_end_s") is not None
    ]
    return {
        "source": str(path),
        "events": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "adjusted_boundary_events": len(adjusted),
        "feedback_versions": sorted({row.get("feedback_version") for row in rows}),
        "model_versions": sorted({row.get("model_version") for row in rows}),
        "policy_versions": sorted({row.get("policy_version") for row in rows}),
        "raw_query_or_video_path_in_schema": any("query" in row or "video_path" in row for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"feedback file does not exist: {args.input}")
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
