#!/usr/bin/env python3
"""Run a real SigLIP video-feature cache smoke with two queries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoitg_smart_clip.preprocessing import FeatureCache, SigLIPFeatureEncoder, cache_identity, decode_uniform_frames
from videoitg_smart_clip.retrieval import CachedCosineRetriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/raw/imax.mp4"))
    parser.add_argument("--model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"))
    parser.add_argument("--cache-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/features/g2_siglip_smoke"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=16)
    args = parser.parse_args()

    video_id = args.video.stem
    encoder = SigLIPFeatureEncoder(args.model, device="cuda:0", batch_size=8)
    # decord and CUDA model initialization can conflict in this environment;
    # load the model before opening a VideoReader.
    encoder.load()
    cache = FeatureCache(args.cache_root, feature_model="google/siglip-so400m-patch14-384")
    key = cache_identity(video_id, encoder.version, {"decode": "decord", "segment": "uniform"}, {"fps": 1.0, "max_frames": args.max_frames})
    retriever = CachedCosineRetriever(cache, query_encoder=encoder.encode_query, window_seconds=10.0)

    started = time.perf_counter()
    from decord import VideoReader, cpu
    metadata_reader = VideoReader(str(args.video), ctx=cpu(0), num_threads=2)
    duration = len(metadata_reader) / float(metadata_reader.get_avg_fps())
    state = {}

    def extract_once():
        if state.get("called"):
            raise AssertionError("feature extractor called more than once")
        state["called"] = True
        frames, timestamps, _ = decode_uniform_frames(args.video, sample_fps=1.0, max_frames=args.max_frames)
        state["sampled_frames"] = len(frames)
        state["duration"] = duration
        return encoder.encode(frames), timestamps

    retriever.index(args.video, video_id, key=key, extractor=extract_once, video_duration=duration)
    first_cache_hit = bool(retriever.last_metrics.get("cache_hit"))
    first = retriever.retrieve(video_id, "a person walking", top_n=5)
    retriever.index(args.video, video_id, key=key, extractor=lambda: (_ for _ in ()).throw(AssertionError("cache miss on second query")), video_duration=state["duration"])
    second_cache_hit = bool(retriever.last_metrics.get("cache_hit"))
    second = retriever.retrieve(video_id, "a person sitting", top_n=5)
    result = {
        "run_id": f"g2_siglip_cache_{time.strftime('%Y%m%d_%H%M%S')}",
        "video": str(args.video),
        "video_id": video_id,
        "feature_model": encoder.version,
        "sampled_frames": state.get("sampled_frames"),
        "video_duration": state.get("duration"),
        "first_query": {"cache_hit": first_cache_hit, "candidates": [c.__dict__ for c in first]},
        "second_query": {"cache_hit": second_cache_hit, "candidates": [c.__dict__ for c in second]},
        "cache_events": cache.cache_events,
        "feature_path": str(cache.load(key).feature_path),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "extractor_calls": int(state.get("called", False)),
    }
    output = args.output or ROOT / "records" / "phase_g2" / f"{result['run_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
