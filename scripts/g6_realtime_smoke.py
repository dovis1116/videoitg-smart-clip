#!/usr/bin/env python3
"""Run one real local Coarse-to-Fine worker request and save lifecycle evidence."""

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
    parser.add_argument("--feature-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/features/g6_smoke"))
    parser.add_argument("--query", default="a person performing an action")
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "records/phase_g6/g6_realtime_smoke.json")
    args = parser.parse_args()
    encoder = SigLIPFeatureEncoder(args.feature_model, device="cuda:0", batch_size=8)
    grounder = TimeLensGrounder(args.timelens_model, device="cuda:0", batch_size=args.batch_size)
    worker = CoarseToFineWorker(args.feature_root, feature_encoder=encoder, grounder=grounder, top_n=args.top_n, top_k=min(3, args.top_n))
    stages = []
    started = time.perf_counter()
    result = worker.run_with_progress(args.video, args.query, lambda stage, progress: stages.append({"stage": stage, "progress": progress, "elapsed_ms": (time.perf_counter() - started) * 1000.0}))
    record = {"run_id": f"g6_realtime_smoke_{time.strftime('%Y%m%d_%H%M%S')}", "video": str(args.video), "query": args.query, "stages": stages, "elapsed_ms": (time.perf_counter() - started) * 1000.0, "result": result, "scope": "single_local_request_smoke_not_throughput_or_quality_gate"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
