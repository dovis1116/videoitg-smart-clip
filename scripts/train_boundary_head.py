#!/usr/bin/env python
"""Fit the CPU-side boundary head on a train-only frame-score dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from videoitg_smart_clip.boundary_head import fit_boundary_head


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--ridge", type=float, default=0.1)
    args = parser.parse_args()
    rows = [row for row in load(args.input) if row.get("split") == args.split]
    if not rows:
        raise SystemExit(f"no rows for split={args.split!r}")
    model = fit_boundary_head(rows, ridge=args.ridge)
    model.update({"input": str(args.input), "fit_split": args.split})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "model": str(args.output), "positive_frames": model["positive_frame_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
