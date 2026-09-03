#!/usr/bin/env python
"""Select unused, media-backed records for a deterministic held-out audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def split_for_video(video: str) -> str:
    bucket = int(hashlib.sha256(video.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "dev" if bucket < 20 else "train"


def source_group(video: str) -> str:
    parts = video.split("/")
    return parts[1] if len(parts) > 1 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    excluded: set[str] = set()
    for manifest in args.exclude_manifest:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                excluded.add(json.loads(line)["video_id"])

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    first_by_video: dict[str, dict] = {}
    for item in records:
        first_by_video.setdefault(item["video"], item)

    selected = []
    for video in sorted(first_by_video):
        item = first_by_video[video]
        path = args.raw_root / video
        if video in excluded or split_for_video(video) != args.split or not path.is_file():
            continue
        selected.append(
            {
                "sample_id": f"videoitg40k:heldout_extra:{item['id']}",
                "video_id": video,
                "video_path": str(path),
                "source_group": source_group(video),
                "split": args.split,
                "query": item["question"],
                "answer": item["answer"],
                "frame_num": item["frame_num"],
                "clip_num": item["clip_num"],
                "motion": item["motion"],
                "existence": item["existence"],
                "raw_video_present": True,
                "metadata_license": "Apache-2.0 (dataset card)",
                "media_license": "inherited from LLaVA-Video source; verify before use",
            }
        )
        if args.limit and len(selected) >= args.limit:
            break

    if not selected:
        raise RuntimeError("no unused media-backed records matched the requested split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": len(selected), "split": args.split}, ensure_ascii=False))


if __name__ == "__main__":
    main()
