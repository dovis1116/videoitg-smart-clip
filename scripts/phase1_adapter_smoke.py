#!/usr/bin/env python
"""Exercise the project adapter on two short intervals from imax.mp4."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from videoitg_smart_clip.reranker import CandidateSegment, VideoITGReranker

    video = Path(args.video)
    candidates = [
        CandidateSegment(video, 0.0, 12.0, candidate_id="c0"),
        CandidateSegment(video, 120.0, 132.0, candidate_id="c1"),
    ]
    reranker = VideoITGReranker(args.model, max_frames_per_candidate=8, frame_score_topk=4)
    ranked = reranker.rank("Which IMAX movie is not shown in the video?", candidates)
    result = [
        {
            "candidate_id": item.candidate.candidate_id,
            "start_s": item.candidate.start_s,
            "end_s": item.candidate.end_s,
            "segment_score": item.segment_score,
            "sampled_frame_indices": item.sampled_frame_indices,
            "frame_scores": item.frame_scores,
            "model_version": item.model_version,
            "runtime": item.runtime,
        }
        for item in ranked
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
