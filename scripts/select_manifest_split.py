#!/usr/bin/env python
"""Select one split from an existing JSONL manifest without changing rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row.get("split") in set(args.split)]
    if not selected:
        raise RuntimeError(f"no rows for split={args.split}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "rows": len(selected), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
