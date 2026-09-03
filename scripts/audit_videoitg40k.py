#!/usr/bin/env python
"""Audit VideoITG-40K metadata and make a deterministic video-level pilot manifest."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


REQUIRED = {
    "id",
    "video",
    "question",
    "answer",
    "frame_num",
    "clip_num",
    "motion",
    "existence",
}


def split_name(video):
    parts = video.split("/")
    return parts[1] if len(parts) > 1 else "unknown"


def media_relative_path(video):
    """Strip the dataset namespace used in annotation paths before local join."""
    return video.split("/", 1)[1] if "/" in video else video


def video_split(video):
    # Keep every annotation of one source video in the same split.
    bucket = int(hashlib.sha256(video.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "dev" if bucket < 20 else "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--pilot-videos", type=int, default=50)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    errors = []
    id_values = []
    video_values = []
    video_question_values = []
    group_counts = collections.Counter()
    split_video_sets = collections.defaultdict(set)
    split_record_counts = collections.Counter()
    motion = collections.Counter()
    existence = collections.Counter()
    frame_lengths = []
    clip_lengths = []
    missing_media = set()
    by_video = collections.defaultdict(list)

    for pos, item in enumerate(records):
        missing = REQUIRED.difference(item)
        if missing:
            errors.append({"position": pos, "error": "missing_keys", "keys": sorted(missing)})
            continue
        if not isinstance(item["id"], int):
            errors.append({"position": pos, "error": "id_not_int"})
        if not isinstance(item["video"], str) or not item["video"].strip():
            errors.append({"position": pos, "error": "video_empty"})
        if not isinstance(item["question"], str) or not item["question"].strip():
            errors.append({"position": pos, "error": "question_empty"})
        if not isinstance(item["answer"], str) or not item["answer"].strip():
            errors.append({"position": pos, "error": "answer_empty"})
        for field, values, lengths in (
            ("frame_num", item["frame_num"], frame_lengths),
            ("clip_num", item["clip_num"], clip_lengths),
        ):
            if not isinstance(values, list) or not values or any(not isinstance(v, int) or v < 0 for v in values):
                errors.append({"position": pos, "error": f"{field}_invalid"})
            else:
                lengths.append(len(values))

        vid = item["video"]
        split = video_split(vid)
        id_values.append(item["id"])
        video_values.append(vid)
        video_question_values.append((vid, item["question"].strip()))
        group_counts[split_name(vid)] += 1
        split_video_sets[split].add(vid)
        split_record_counts[split] += 1
        motion[str(item["motion"])] += 1
        existence[str(item["existence"])] += 1
        by_video[vid].append(item)
        if not (args.raw_root / media_relative_path(vid)).is_file():
            missing_media.add(vid)

    def dup_count(values):
        counts = collections.Counter(values)
        return sum(n - 1 for n in counts.values() if n > 1), len(counts), max(counts.values())

    id_dups, id_unique, id_max = dup_count(id_values)
    video_dups, video_unique, video_max = dup_count(video_values)
    q_dups, q_unique, q_max = dup_count(video_question_values)
    overlap = {
        f"{a}_vs_{b}": sorted(split_video_sets[a] & split_video_sets[b])[:5]
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    pilot = []
    # Round-robin source groups gives the pilot coverage without pretending media exists.
    groups = collections.defaultdict(list)
    for vid, items in by_video.items():
        groups[split_name(vid)].append((vid, items[0]))
    for values in groups.values():
        values.sort(key=lambda pair: pair[0])
    while len(pilot) < args.pilot_videos and any(groups.values()):
        for group in sorted(groups):
            if groups[group] and len(pilot) < args.pilot_videos:
                vid, item = groups[group].pop(0)
                pilot.append(
                    {
                        "sample_id": f"videoitg40k:{video_split(vid)}:{item['id']}",
                        "video_id": vid,
                        "video_path": str(args.raw_root / media_relative_path(vid)),
                        "source_group": group,
                        "split": video_split(vid),
                        "query": item["question"],
                        "answer": item["answer"],
                        "frame_num": item["frame_num"],
                        "clip_num": item["clip_num"],
                        "has_target": True,
                        "metadata_license": "Apache-2.0 (dataset card)",
                        "media_license": "inherited from LLaVA-Video source; verify before use",
                        "raw_video_present": (args.raw_root / media_relative_path(vid)).is_file(),
                    }
                )

    summary = {
        "input": str(args.input),
        "record_count": len(records),
        "video_count": len(by_video),
        "source_group_record_counts": dict(sorted(group_counts.items())),
        "split_record_counts": dict(sorted(split_record_counts.items())),
        "split_video_counts": {k: len(v) for k, v in sorted(split_video_sets.items())},
        "duplicate_checks": {
            "id": {"unique": id_unique, "duplicate_records": id_dups, "max_count": id_max},
            "video": {"unique": video_unique, "duplicate_records": video_dups, "max_count": video_max},
            "video_question": {"unique": q_unique, "duplicate_records": q_dups, "max_count": q_max},
        },
        "split_video_overlap_examples": overlap,
        "label_counts": {"motion": dict(motion), "existence": dict(existence)},
        "frame_count": {"min": min(frame_lengths), "max": max(frame_lengths), "median": sorted(frame_lengths)[len(frame_lengths) // 2]},
        "clip_count": {"min": min(clip_lengths), "max": max(clip_lengths), "median": sorted(clip_lengths)[len(clip_lengths) // 2]},
        "raw_media": {
            "raw_root": str(args.raw_root),
            "expected_paths": len(by_video),
            "missing_annotation_media_paths": len(missing_media),
        },
        "validation_errors": errors[:100],
        "validation_error_count": len(errors),
        "pilot_count": len(pilot),
        "pilot_is_metadata_only": True,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.pilot.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in pilot) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
