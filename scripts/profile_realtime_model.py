#!/usr/bin/env python3
"""Profile one real local TimeLens request before any model-path tuning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoitg_smart_clip.grounding import TimeLensGrounder
from videoitg_smart_clip.preprocessing import SigLIPFeatureEncoder
from videoitg_smart_clip.service.runtime import CoarseToFineWorker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/processed/g6/imax10s.mp4"))
    parser.add_argument("--feature-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"))
    parser.add_argument("--timelens-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B"))
    parser.add_argument("--feature-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/features/g9_profile"))
    parser.add_argument("--query", default="a person performing an action")
    parser.add_argument("--timelens-total-pixels", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--output", type=Path, default=ROOT / "records/phase_g9/realtime_model_profile.json")
    args = parser.parse_args()
    import torch

    encoder = SigLIPFeatureEncoder(args.feature_model, device="cuda:0", batch_size=8)
    grounder = TimeLensGrounder(args.timelens_model, device="cuda:0", batch_size=1, total_pixels=args.timelens_total_pixels)
    worker = CoarseToFineWorker(args.feature_root, feature_encoder=encoder, grounder=grounder, top_n=1, top_k=1)
    stages = []
    stage_started = {}

    def stage_callback(stage: str, progress: float) -> None:
        now = time.perf_counter()
        if stage_started:
            previous = next(reversed(stage_started))
            stages.append({"stage": previous, "elapsed_ms": (now - stage_started[previous]) * 1000.0})
        stage_started[stage] = now

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        result = worker.run_with_progress(args.video, args.query, stage_callback)
    if stage_started:
        previous = next(reversed(stage_started))
        stages.append({"stage": previous, "elapsed_ms": (time.perf_counter() - stage_started[previous]) * 1000.0})
    record = {
        "run_id": f"realtime_model_profile_{time.strftime('%Y%m%d_%H%M%S')}",
        "video": str(args.video),
        "query": args.query,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "stages": stages,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(torch.device("cuda:0")) / 2**30 if torch.cuda.is_available() else None,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(torch.device("cuda:0")) / 2**30 if torch.cuda.is_available() else None,
        "profiler_top_cuda": profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=20),
        "result": result,
        "scope": "single_real_model_request_profiler_not_throughput_or_quality_gate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
