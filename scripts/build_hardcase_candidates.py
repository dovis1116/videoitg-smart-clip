#!/usr/bin/env python
"""Create review-only hardcase candidates without pretending to create labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        tags = []
        if len(item["clip_num"]) > 1:
            tags.append("boundary_shift_candidate")
        if len(item["clip_num"]) == 1:
            tags.append("short_event_candidate")
        # This is intentionally a review request, not a no-match label.
        tags.append("no_match_requires_manual_review")
        out.append(
            {
                "sample_id": item["sample_id"],
                "video_id": item["video_id"],
                "video_path": item["video_path"],
                "query": item["query"],
                "answer": item["answer"],
                "source_group": item["source_group"],
                "candidate_tags": tags,
                "label_status": "pending_manual",
                "evidence": {
                    "clip_num": item["clip_num"],
                    "frame_num": item["frame_num"],
                    "existence_field": item.get("existence"),
                    "motion_field": item.get("motion"),
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(out), "label_status": "pending_manual"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
