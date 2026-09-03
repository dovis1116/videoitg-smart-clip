#!/usr/bin/env python3
"""Validate the labeled G8 grounding/no-match JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoitg_smart_clip.evaluation.validation import ValidationManifestError, validate_manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="JSONL validation manifest")
    parser.add_argument("--require-complete", action="store_true", help="reject pending labels")
    parser.add_argument("--require-negative-categories", action="store_true", help="require all four negative families and both classes")
    parser.add_argument("--require-files", action="store_true", help="require every video_path to exist on the local filesystem")
    args = parser.parse_args()
    try:
        rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = validate_manifest_rows(
            rows,
            require_complete=args.require_complete,
            require_negative_categories=args.require_negative_categories,
            require_files=args.require_files,
        )
    except (OSError, json.JSONDecodeError, ValidationManifestError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": "valid", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
