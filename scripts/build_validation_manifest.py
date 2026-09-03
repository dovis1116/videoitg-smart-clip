#!/usr/bin/env python3
"""Build explicit present or synthetic-negative rows for offline evaluation.

The command never infers real semantic negatives. The caller must explicitly
declare whether the selected source rows are present or one named negative
family. Synthetic rows are marked ``label_status=synthetic`` so downstream
reports can distinguish constructed behavior tests from real labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping


NEGATIVE_TYPES = {"event_absent", "wrong_action", "wrong_object", "theme_unrelated"}


def _ground_truth(row: Mapping, *, clip_seconds: float) -> list[list[float]]:
    value = row.get("ground_truth_segments", row.get("ground_truth"))
    if value is not None:
        return [[float(segment[0]), float(segment[1])] for segment in value]
    clip_numbers = row.get("clip_num")
    if clip_numbers is None:
        raise ValueError(f"row {row.get('sample_id', '<unknown>')} has no ground_truth_segments or clip_num")
    numbers = sorted({int(number) for number in clip_numbers})
    if not numbers:
        raise ValueError(f"row {row.get('sample_id', '<unknown>')} has empty clip_num")
    segments: list[list[float]] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number != previous + 1:
            segments.append([start * clip_seconds, (previous + 1) * clip_seconds])
            start = number
        previous = number
    segments.append([start * clip_seconds, (previous + 1) * clip_seconds])
    return segments


def build_rows(
    rows: Iterable[Mapping],
    *,
    match_class: str,
    negative_type: str | None = None,
    clip_seconds: float = 5.0,
    split: str | None = None,
    synthetic: bool = False,
) -> list[dict]:
    """Convert source rows into explicit, still-auditable validation rows."""

    if match_class not in {"present", "absent"}:
        raise ValueError("match_class must be present or absent")
    if match_class == "absent" and negative_type not in NEGATIVE_TYPES:
        raise ValueError("absent scaffolds require one of the four negative types")
    if match_class == "present" and negative_type is not None:
        raise ValueError("present scaffolds must use negative_type=none")
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be positive")

    output: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        video_id = str(row.get("video_id", "")).strip()
        query = str(row.get("query", "")).strip()
        if not sample_id or not video_id or not query:
            raise ValueError("source rows require non-empty sample_id, video_id and query")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if match_class == "present":
            ground_truth = _ground_truth(row, clip_seconds=clip_seconds)
            actual_match = True
            negative = "none"
        else:
            ground_truth = []
            actual_match = False
            negative = str(negative_type)
        item = {
            "sample_id": sample_id,
            "video_id": video_id,
            "video_path": str(row.get("video_path", "")),
            "query": query,
            "actual_match": actual_match,
            "negative_type": negative,
            "ground_truth": ground_truth,
            "split": split or str(row.get("split", "dev")),
            "label_status": "synthetic" if synthetic else "pending",
            "source_sample_id": sample_id,
            "scaffold_assumption": f"synthetic_{match_class}" if synthetic else f"explicit_{match_class}",
        }
        output.append(item)
    if not output:
        raise ValueError("source manifest contains no rows")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="source JSONL pool")
    parser.add_argument("output", type=Path, help="generated validation JSONL")
    parser.add_argument("--match-class", choices=("present", "absent"), required=True)
    parser.add_argument("--negative-type", choices=sorted(NEGATIVE_TYPES), help="required for --match-class absent")
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--synthetic", action="store_true", help="mark explicitly constructed rows as synthetic offline evidence")
    args = parser.parse_args()
    try:
        source = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        generated = build_rows(
            source,
            match_class=args.match_class,
            negative_type=args.negative_type,
            clip_seconds=args.clip_seconds,
            split=args.split,
            synthetic=args.synthetic,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "row_count": len(generated), "output": str(args.output), "label_status": "synthetic" if args.synthetic else "pending"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
