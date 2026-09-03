#!/usr/bin/env python
"""Export uniform 2-second full-video contact sheets for manual replay support."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--sample-id", action="append", required=True)
    args = parser.parse_args()

    from PIL import Image, ImageDraw, ImageFont
    from decord import VideoReader, cpu

    pool = load_jsonl(args.pool)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    results = []
    for sample_id in args.sample_id:
        row = pool[sample_id]
        reader = VideoReader(row["video_path"], ctx=cpu(0), num_threads=2)
        fps = float(reader.get_avg_fps())
        duration = len(reader) / fps
        times = [min(i * args.interval_seconds, max(0.0, duration - 0.01)) for i in range(math.ceil(duration / args.interval_seconds))]
        # Include annotation/prediction centers even when they fall between uniform samples.
        for segment in row.get("ground_truth_segments", []):
            times.append((float(segment[0]) + float(segment[1])) / 2.0)
        for prediction in row.get("predictions", [])[:3]:
            times.append((float(prediction["start_s"]) + float(prediction["end_s"])) / 2.0)
        times = sorted({round(max(0.0, min(duration - 0.01, t)), 2) for t in times})
        thumb_w, thumb_h = 240, 150
        label_h = 22
        cols = 5
        rows = max(1, math.ceil(len(times) / cols))
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, requested in enumerate(times):
            frame_index = max(0, min(len(reader) - 1, int(round(requested * fps))))
            actual = frame_index / fps
            image = Image.fromarray(reader[frame_index].asnumpy()).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            tile = Image.new("RGB", (thumb_w, thumb_h), "#dddddd")
            tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
            x = (index % cols) * thumb_w
            y = (index // cols) * (thumb_h + label_h)
            sheet.paste(tile, (x, y))
            draw.text((x + 4, y + thumb_h + 3), f"{actual:.1f}s", fill="black", font=font)
        out = args.output_dir / f"{sample_id.rsplit(':', 1)[-1]}.jpg"
        sheet.save(out, quality=90)
        results.append({"sample_id": sample_id, "video_path": row["video_path"], "duration_s": duration, "interval_s": args.interval_seconds, "frame_count": len(times), "output": str(out)})
    (args.output_dir / "index.json").write_text(json.dumps({"pool": str(args.pool), "rows": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
