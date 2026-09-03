#!/usr/bin/env python
"""Select a deterministic pilot whose referenced media is actually present."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def relative_path(video: str) -> str:
    return video.split("/", 1)[1] if "/" in video else video


def group(video: str) -> str:
    parts = video.split("/")
    return parts[1] if len(parts) > 1 else "unknown"


def video_split(video: str) -> str:
    bucket = int(hashlib.sha256(video.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "dev" if bucket < 20 else "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    first_by_video = {}
    for item in records:
        first_by_video.setdefault(item["video"], item)
    available = collections.defaultdict(list)
    for video, item in first_by_video.items():
        path = args.raw_root / relative_path(video)
        if path.is_file():
            available[group(video)].append((video, item, path))
    for values in available.values():
        values.sort(key=lambda x: x[0])
    selected = []
    while len(selected) < args.limit and any(available.values()):
        for name in sorted(available):
            if available[name] and len(selected) < args.limit:
                video, item, path = available[name].pop(0)
                selected.append(
                    {
                        "sample_id": f"videoitg40k:media_pilot:{item['id']}",
                        "video_id": video,
                        "video_path": str(path),
                        "source_group": group(video),
                        "split": video_split(video),
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in selected) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(selected), "source_groups": collections.Counter(x["source_group"] for x in selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
