#!/usr/bin/env python3
"""Run one real TimeLens candidate-window grounding smoke."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoitg_smart_clip.grounding import TimeLensGrounder
from videoitg_smart_clip.pipeline import CandidateWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/raw/imax.mp4"))
    parser.add_argument("--model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B"))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=10.0)
    parser.add_argument("--count", type=int, default=1, help="number of adjacent candidate windows")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--total-pixels", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--query", default="a person performing an action")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    grounder = TimeLensGrounder(args.model, device="cuda:0", batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, total_pixels=args.total_pixels)
    window_seconds = args.end - args.start
    windows = [CandidateWindow(args.start + i * window_seconds, args.end + i * window_seconds, 0.5, f"g4-smoke-{i}") for i in range(max(1, args.count))]
    started = time.perf_counter()
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    error = None
    predictions = []
    try:
        predictions = grounder.predict(args.video, args.query, windows)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    result = {"run_id": f"g4_timelens_smoke_{time.strftime('%Y%m%d_%H%M%S')}", "video": str(args.video), "query": args.query, "candidate_windows": [window.__dict__ for window in windows], "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "total_pixels": args.total_pixels, "predictions": [prediction.__dict__ for prediction in predictions], "error": error, "status": "failed" if error else "succeeded", "elapsed_ms": (time.perf_counter() - started) * 1000.0, "gpu_peak_memory_allocated_gib": (torch.cuda.max_memory_allocated(torch.device("cuda:0")) / 2**30 if torch.cuda.is_available() else None), "gpu_peak_memory_reserved_gib": (torch.cuda.max_memory_reserved(torch.device("cuda:0")) / 2**30 if torch.cuda.is_available() else None)}
    output = args.output or ROOT / "records/phase_g4" / f"{result['run_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
