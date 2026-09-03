#!/usr/bin/env python3
"""Run two real TimeLens workers on two isolated GPUs and audit cleanup."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoitg_smart_clip.grounding import TimeLensGrounder
from videoitg_smart_clip.preprocessing import SigLIPFeatureEncoder
from videoitg_smart_clip.service.models import TaskMode
from videoitg_smart_clip.service.runtime import BoundedTaskManager, CoarseToFineWorker


def snapshot(record) -> dict:
    return {
        "task_id": record.task_id,
        "status": record.status.value,
        "current_stage": record.current_stage,
        "error_code": record.error_code,
        "degraded": record.degraded,
        "started_at": record.started_at.isoformat() if record.started_at else None,
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
    }


def overlap_seconds(first: dict, second: dict) -> float:
    if not first.get("started_at") or not first.get("finished_at") or not second.get("started_at") or not second.get("finished_at"):
        return 0.0
    starts = [datetime.fromisoformat(first["started_at"]).timestamp(), datetime.fromisoformat(second["started_at"]).timestamp()]
    ends = [datetime.fromisoformat(first["finished_at"]).timestamp(), datetime.fromisoformat(second["finished_at"]).timestamp()]
    return max(0.0, min(ends) - max(starts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/processed/g6/imax10s.mp4"))
    parser.add_argument("--feature-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"))
    parser.add_argument("--timelens-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B"))
    parser.add_argument("--feature-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/features/g6_two_worker_matrix"))
    parser.add_argument("--state-path", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/runtime/g6_two_worker_matrix.jsonl"))
    parser.add_argument("--total-pixels", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--output", type=Path, default=ROOT / "records/phase_g6/g6_real_model_two_worker_matrix.json")
    args = parser.parse_args()

    devices = ["cuda:0", "cuda:1"]
    encoders = [SigLIPFeatureEncoder(args.feature_model, device=device, batch_size=8) for device in devices]
    grounders = [TimeLensGrounder(args.timelens_model, device=device, batch_size=1, total_pixels=args.total_pixels) for device in devices]
    workers = [
        CoarseToFineWorker(args.feature_root / f"worker_{index}", feature_encoder=encoders[index], grounder=grounders[index], top_n=1, top_k=1)
        for index in range(2)
    ]
    manager = BoundedTaskManager(workers, queue_size=2, state_path=args.state_path, task_timeout_s=180.0)
    records = []
    started = time.perf_counter()
    try:
        for index in range(2):
            record, deduplicated = manager.submit(
                args.video,
                f"a person performing an action {index}",
                TaskMode.burst_async,
                request_id=f"real-two-worker-{index}-{uuid.uuid4().hex}",
            )
            records.append((record, deduplicated))
        terminal = [manager.wait(record.task_id, 240.0) for record, _ in records]
        snapshots = [snapshot(record) for record in terminal]
        result = {
            "run_id": f"g6_real_model_two_worker_matrix_{time.strftime('%Y%m%d_%H%M%S')}",
            "video": str(args.video),
            "devices": devices,
            "total_pixels": args.total_pixels,
            "worker_count": len(workers),
            "tasks": snapshots,
            "overlap_seconds": overlap_seconds(snapshots[0], snapshots[1]),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "scope": "real_model_two_worker_cross_gpu_concurrency_and_cleanup_not_throughput",
        }
    finally:
        manager.shutdown(wait=True)
        result = locals().get("result", {"run_id": f"g6_real_model_two_worker_matrix_{time.strftime('%Y%m%d_%H%M%S')}", "error": "matrix did not reach terminal collection"})
        result["threads_alive_after_shutdown"] = [thread.name for thread in manager._threads if thread.is_alive()]
        cleanup = []
        try:
            import torch

            for index, (encoder, grounder) in enumerate(zip(encoders, grounders)):
                device = torch.device(f"cuda:{index}")
                torch.cuda.set_device(device)
                before = {
                    "cuda_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                    "cuda_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                }
                grounder._model = None
                grounder._processor = None
                encoder._model = None
                encoder._processor = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                after = {
                    "cuda_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                    "cuda_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                }
                cleanup.append({"device": str(device), "before": before, "after": after})
        except Exception as exc:
            cleanup.append({"error": f"{type(exc).__name__}: {exc}"})
        result["cleanup"] = cleanup
        result["passed"] = (
            len(result.get("tasks", [])) == 2
            and all(task.get("status") == "succeeded" and not task.get("degraded") for task in result["tasks"])
            and result.get("overlap_seconds", 0.0) > 0.0
            and not result.get("threads_alive_after_shutdown")
            and len(cleanup) == 2
            and all(item.get("after", {}).get("cuda_allocated_gib", 1.0) < 0.1 for item in cleanup)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"saved: {args.output}")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
