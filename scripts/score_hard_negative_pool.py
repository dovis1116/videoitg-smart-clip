#!/usr/bin/env python
"""Score reviewed train-side temporal hard negatives against their GT intervals."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videoitg-model", default="/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/VideoITG-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--frame-score-topk", type=int, default=8)
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        import torch

        torch.cuda.set_device(args.device)
    from videoitg_smart_clip.reranker import CandidateSegment, VideoITGReranker

    rows = [json.loads(line) for line in args.pool.read_text().splitlines() if line.strip()]
    reranker = VideoITGReranker(args.videoitg_model, device=args.device, target_fps=args.target_fps, max_frames_per_candidate=args.max_frames, frame_score_topk=args.frame_score_topk)
    scored_rows = []
    for index, row in enumerate(rows, start=1):
        gt_segments = row["ground_truth_segments"]
        candidate_start, candidate_end = row["candidate_segment"]
        candidates = [
            CandidateSegment(row["video_path"], float(start), float(end), candidate_id="gt")
            for start, end in gt_segments
        ]
        candidates.append(CandidateSegment(row["video_path"], float(candidate_start), float(candidate_end), candidate_id=row["candidate_id"]))
        started = time.perf_counter()
        scored = reranker.rank(row["query"], candidates)
        by_id = {item.candidate.candidate_id: item for item in scored}
        gt_scores = [float(by_id["gt"].segment_score)]
        negative_score = float(by_id[row["candidate_id"]].segment_score)
        scored_rows.append({
            "candidate_id": row["candidate_id"],
            "sample_id": row["sample_id"],
            "source_badcase": row.get("source_badcase"),
            "ground_truth_segments": gt_segments,
            "candidate_segment": row["candidate_segment"],
            "gt_score": max(gt_scores),
            "negative_score": negative_score,
            "negative_minus_gt": negative_score - max(gt_scores),
            "negative_rank_beats_gt": negative_score >= max(gt_scores),
            "sampled_frames": sum(len(item.sampled_frame_indices) for item in scored),
            "wall_seconds": time.perf_counter() - started,
            "not_for_training": True,
        })
        print(f"{index}/{len(rows)} {row['candidate_id']}")
    report = {
        "status": "diagnostic_only",
        "pool": str(args.pool),
        "rows": len(scored_rows),
        "all_not_for_training": all(row["not_for_training"] for row in scored_rows),
        "negative_beats_gt_count": sum(row["negative_rank_beats_gt"] for row in scored_rows),
        "by_source_badcase": {
            code: {
                "rows": sum(row["source_badcase"] == code for row in scored_rows),
                "negative_beats_gt": sum(row["source_badcase"] == code and row["negative_rank_beats_gt"] for row in scored_rows),
            }
            for code in sorted({row["source_badcase"] for row in scored_rows})
        },
        "records": scored_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "rows", "negative_beats_gt_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
