#!/usr/bin/env python
"""Remove media paths flagged by the duplicate-screen artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--duplicate-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = json.loads(args.duplicate_screen.read_text(encoding="utf-8"))
    excluded = {x for pair in screen.get("first_frame_near_duplicate_pairs", []) for x in (pair["video_a"], pair["video_b"])}
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in rows if row["video_id"] not in excluded]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in kept) + "\n", encoding="utf-8")
    print(json.dumps({"input": len(rows), "excluded": len(rows) - len(kept), "output": len(kept), "excluded_video_count": len(excluded)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
