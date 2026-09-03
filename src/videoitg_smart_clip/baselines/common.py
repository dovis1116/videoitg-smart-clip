"""Shared sampling, candidate generation, and metadata helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from videoitg_smart_clip.evaluation.metrics import gt_segments_from_clip_num


def video_metadata(path: str) -> tuple[int, float, float]:
    from decord import VideoReader, cpu

    reader = VideoReader(path, ctx=cpu(0), num_threads=2)
    frames = len(reader)
    fps = float(reader.get_avg_fps())
    return frames, fps, frames / fps


def uniform_candidates(path: str, segment_seconds: float = 5.0, max_candidates: int = 64) -> list[dict]:
    _, _, duration = video_metadata(path)
    count = max(1, int((duration + segment_seconds - 1e-9) // segment_seconds))
    starts = [i * segment_seconds for i in range(count)]
    if len(starts) > max_candidates:
        positions = [round(i * (len(starts) - 1) / (max_candidates - 1)) for i in range(max_candidates)]
        starts = [starts[i] for i in positions]
    return [
        {"candidate_id": f"u{idx:04d}", "start_s": start, "end_s": min(duration, start + segment_seconds)}
        for idx, start in enumerate(starts)
    ]


def candidate_to_segment(candidate: dict):
    from videoitg_smart_clip.reranker import CandidateSegment

    return CandidateSegment(candidate["video_path"], candidate["start_s"], candidate["end_s"], candidate["candidate_id"])


def run_metadata(model_paths: dict, baseline: str, manifest: str, limit: int | None, seed: int) -> dict:
    import sys

    def git_revision(path: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None

    return {
        "run_id": f"{baseline}_{time.strftime('%Y%m%d_%H%M%S')}",
        "baseline": baseline,
        "manifest": manifest,
        "limit": limit,
        "seed": seed,
        "python": sys.version,
        "models": model_paths,
        "upstream_videoitg_revision": git_revision("/home/zjy/projects/videoitg_smart_clip/external/VideoITG"),
    }
