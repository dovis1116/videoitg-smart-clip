#!/usr/bin/env python
"""Screen exact duplicates and simple first-frame near duplicates in a pilot."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image


def average_hash(frame) -> int:
    gray = np.asarray(Image.fromarray(frame).convert("L").resize((8, 8)))
    bits = gray >= gray.mean()
    return sum((int(bit) << i) for i, bit in enumerate(bits.ravel()))


def sampled_hashes(row) -> dict[str, int]:
    from decord import VideoReader, cpu

    reader = VideoReader(row["video_path"], ctx=cpu(0), num_threads=1)
    n = len(reader)
    fps = float(reader.get_avg_fps())
    annotated = min(n - 1, max(0, int(round(row["frame_num"][0] * fps))))
    positions = {"first": 0, "middle": n // 2, "annotated": annotated}
    return {name: average_hash(reader[idx].asnumpy()) for name, idx in positions.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hamming", type=int, default=4)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    exact = {}
    hashes = {}
    for row in rows:
        path = Path(row["video_path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        exact.setdefault(digest, []).append(row["video_id"])
        hashes[row["video_id"]] = sampled_hashes(row)
    exact_groups = [values for values in exact.values() if len(values) > 1]
    near = []
    for a, b in itertools.combinations(rows, 2):
        distances = {
            position: (hashes[a["video_id"]][position] ^ hashes[b["video_id"]][position]).bit_count()
            for position in hashes[a["video_id"]]
        }
        close_positions = [position for position, distance in distances.items() if distance <= args.max_hamming]
        if len(close_positions) >= 2:
            near.append({"video_a": a["video_id"], "video_b": b["video_id"], "hamming_by_position": distances, "close_positions": close_positions})
    result = {
        "manifest": str(args.manifest),
        "video_count": len(rows),
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_threshold": args.max_hamming,
        "first_frame_near_duplicate_pairs": near,
        "note": "Three-position average hashes are only a screening signal; semantic near-duplicate review is still required.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
