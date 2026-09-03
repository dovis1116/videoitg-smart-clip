#!/usr/bin/env python
"""Export visual evidence sheets for metric-derived badcase candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def frame_at(reader, seconds: float):
    from PIL import Image

    fps = float(reader.get_avg_fps())
    index = max(0, min(len(reader) - 1, int(round(seconds * fps))))
    frame = reader[index].asnumpy()
    return Image.fromarray(frame), index / fps


def add_time(times: list[tuple[str, float]], label: str, value: float) -> None:
    if value < 0:
        return
    if all(abs(value - old) > 1.0 for _, old in times):
        times.append((label, value))


def times_for(row: dict, *extra_rows: dict | None) -> list[tuple[str, float]]:
    times: list[tuple[str, float]] = []
    for i, gt in enumerate(row.get("ground_truth_segments", [])):
        add_time(times, f"GT{i}: {(gt[0]+gt[1])/2:.1f}s", (gt[0] + gt[1]) / 2)
    for i, prediction in enumerate(row.get("predictions", [])[:3]):
        add_time(times, f"B1#{i+1}: {(prediction['start_s']+prediction['end_s'])/2:.1f}s", (prediction["start_s"] + prediction["end_s"]) / 2)
    for prefix, extra in zip(("B2R", "B2"), extra_rows):
        if extra:
            for i, prediction in enumerate(extra.get("predictions", [])[:3]):
                add_time(times, f"{prefix}#{i+1}: {(prediction['start_s']+prediction['end_s'])/2:.1f}s", (prediction["start_s"] + prediction["end_s"]) / 2)
    return times[:8]


def render_sheet(row: dict, video_path: Path, output: Path, *extra_rows: dict | None) -> list[dict]:
    from PIL import Image, ImageDraw, ImageFont
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    times = times_for(row, *extra_rows)
    thumb_w, thumb_h = 320, 180
    header_h = 86
    cols = 4
    rows = max(1, math.ceil(len(times) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, header_h + rows * (thumb_h + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    query = " ".join(row.get("query", "").split())
    header = f"{row['sample_id']} | {row.get('category', '')}\n{query[:210]}"
    draw.multiline_text((8, 8), header, fill="black", spacing=4)
    font = ImageFont.load_default()
    evidence = []
    for i, (label, seconds) in enumerate(times):
        image, actual = frame_at(reader, seconds)
        image.thumbnail((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = header_h + (i // cols) * (thumb_h + 28)
        tile = Image.new("RGB", (thumb_w, thumb_h), "#dddddd")
        tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + thumb_h + 4), f"{label} actual={actual:.2f}s", fill="black", font=font)
        evidence.append({"label": label, "requested_s": seconds, "actual_s": actual, "frame_index": int(round(actual * float(reader.get_avg_fps())))})
    sheet.save(output, quality=92)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--badcase-report", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--b2r", type=Path)
    parser.add_argument("--b2", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--split", choices=["train", "dev", "test"])
    args = parser.parse_args()

    manifest = {json.loads(line)["sample_id"]: json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()}
    b1 = load_jsonl(args.b1)
    b2r = load_jsonl(args.b2r) if args.b2r else {}
    b2 = load_jsonl(args.b2) if args.b2 else {}
    report = json.loads(args.badcase_report.read_text(encoding="utf-8"))
    candidates = [row for row in report["rows"] if ("weak_overlap" in row["category"] or "near_miss" in row["category"]) and (args.split is None or row.get("split") == args.split)]
    candidates.sort(key=lambda row: (0 if "weak_overlap" in row["category"] else 1, row["b1_top1_iou"], row["sample_id"]))
    selected = candidates[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "source_report": str(args.badcase_report),
        "source_predictions": str(args.b1),
        "manifest": str(args.manifest),
        "selection": "lowest B1 top-1 IoU among weak-overlap then near-miss candidates",
        "rows": [],
    }
    for row in selected:
        sample_id = row["sample_id"]
        manifest_row = manifest[sample_id]
        prediction_row = b1[sample_id]
        output = args.output_dir / (sample_id.rsplit(":", 1)[-1] + ".jpg")
        evidence = render_sheet({**prediction_row, "category": row["category"]}, Path(manifest_row["video_path"]), output, b2r.get(sample_id), b2.get(sample_id))
        index["rows"].append({
            **row,
            "video_path": manifest_row["video_path"],
            "query": manifest_row["query"],
            "answer": manifest_row.get("answer"),
            "ground_truth_segments": prediction_row["ground_truth_segments"],
            "predictions": prediction_row["predictions"],
            "b2r_predictions": b2r.get(sample_id, {}).get("predictions"),
            "b2_predictions": b2.get(sample_id, {}).get("predictions"),
            "keyframes": evidence,
        })
    (args.output_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "selected": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
