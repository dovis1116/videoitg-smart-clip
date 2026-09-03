#!/usr/bin/env python3
"""Profile the non-model coarse-to-fine orchestration before optimization."""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoitg_smart_clip.grounding import StubTimeLensGrounder
from videoitg_smart_clip.pipeline import CoarseToFinePipeline
from videoitg_smart_clip.preprocessing import FeatureCache, HashFeatureEncoder, cache_identity
from videoitg_smart_clip.retrieval import CachedCosineRetriever


def main() -> int:
    output = ROOT / "records/phase_g9/pipeline_profile.json"
    cache = FeatureCache(ROOT / "records/phase_g9/profile_cache")
    retriever = CachedCosineRetriever(cache)
    key = cache_identity("profile-video", "hash-v1", {"segment": "uniform"}, {"fps": 1.0})
    encoder = HashFeatureEncoder()
    retriever.index("profile.mp4", "profile-video", key=key, video_duration=60.0, extractor=lambda: (encoder.encode([np.full((2, 2, 3), i, dtype=np.uint8) for i in range(60)]), np.arange(60, dtype=np.float32)))
    stages = []
    stage_started = {}

    def stage_callback(stage: str, progress: float) -> None:
        now = time.perf_counter()
        if stage_started:
            previous = next(reversed(stage_started))
            stages.append({"stage": previous, "elapsed_ms": (now - stage_started[previous]) * 1000.0})
        stage_started[stage] = now

    pipeline = CoarseToFinePipeline(retriever, StubTimeLensGrounder(), top_n=20, top_k=3)
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    result = pipeline.search("profile.mp4", "profile-video", "profile query", stage_callback=stage_callback)
    profiler.disable()
    if stage_started:
        stages.append({"stage": next(reversed(stage_started)), "elapsed_ms": (time.perf_counter() - stage_started[next(reversed(stage_started))]) * 1000.0})
    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).sort_stats("cumulative").print_stats(12)
    record = {"run_id": f"pipeline_profile_{time.strftime('%Y%m%d_%H%M%S')}", "elapsed_ms": (time.perf_counter() - started) * 1000.0, "stages": stages, "top_functions": stats_stream.getvalue(), "result_status": result["status"], "scope": "orchestration_profile_with_stub_grounder_not_model_benchmark"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
