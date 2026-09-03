#!/usr/bin/env python
"""Check that pilot manifest videos exist, decode, and contain valid labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from decord import VideoReader, cpu

    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        path = Path(item["video_path"])
        result = {
            "sample_id": item["sample_id"],
            "video_id": item["video_id"],
            "path": str(path),
            "exists": path.is_file(),
            "decoded": False,
            "frame_count": None,
            "fps": None,
            "duration_s": None,
            "frame_num_in_1fps_bounds": False,
            "clip_num_in_5s_bounds": False,
            "feature_frame_count_1fps": None,
            "clip_count_5s": None,
            "max_frame_num": max(item["frame_num"]),
            "max_clip_num": max(item["clip_num"]),
            "error": None,
        }
        try:
            reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
            result["decoded"] = True
            result["frame_count"] = len(reader)
            result["fps"] = float(reader.get_avg_fps())
            result["duration_s"] = len(reader) / result["fps"]
            sample_stride = max(1, round(result["fps"] / 1.0))
            result["feature_frame_count_1fps"] = len(range(0, len(reader), sample_stride))
            result["clip_count_5s"] = max(1, math.ceil(result["duration_s"] / 5.0))
            result["frame_num_in_1fps_bounds"] = result["max_frame_num"] < result["feature_frame_count_1fps"]
            result["clip_num_in_5s_bounds"] = result["max_clip_num"] < result["clip_count_5s"]
        except Exception as exc:  # Keep all bad rows in the audit artifact.
            result["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(result)
    summary = {
        "manifest": str(args.manifest),
        "row_count": len(rows),
        "exists_count": sum(x["exists"] for x in rows),
        "decoded_count": sum(x["decoded"] for x in rows),
        "valid_frame_label_count": sum(x["frame_num_in_1fps_bounds"] for x in rows),
        "valid_clip_label_count": sum(x["clip_num_in_5s_bounds"] for x in rows),
        "errors": [x for x in rows if x["error"]],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
