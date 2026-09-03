#!/usr/bin/env python3
"""Exercise queue, cancellation, and cleanup with one real TimeLens worker.

Two independent 8B workers do not fit safely on the available 24 GiB card, so
this matrix deliberately validates the real-model single-worker queue contract
and records the multi-worker limitation instead of claiming GPU parallelism.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import uuid
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
        "progress": record.progress,
        "error_code": record.error_code,
        "cancel_requested": record.cancel_requested,
        "degraded": record.degraded,
        "started": record.started_at is not None,
        "finished": record.finished_at is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/processed/g6/imax10s.mp4"))
    parser.add_argument("--feature-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"))
    parser.add_argument("--timelens-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B"))
    parser.add_argument("--feature-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/features/g6_runtime_matrix"))
    parser.add_argument("--state-path", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/runtime/g6_real_model_runtime_matrix.jsonl"))
    parser.add_argument("--total-pixels", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--wait-s", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=ROOT / "records/phase_g6/g6_real_model_runtime_matrix.json")
    args = parser.parse_args()

    encoder = SigLIPFeatureEncoder(args.feature_model, device="cuda:0", batch_size=8)
    grounder = TimeLensGrounder(args.timelens_model, device="cuda:0", batch_size=1, total_pixels=args.total_pixels)
    worker = CoarseToFineWorker(args.feature_root, feature_encoder=encoder, grounder=grounder, top_n=1, top_k=1)
    manager = BoundedTaskManager([worker], queue_size=2, state_path=args.state_path, task_timeout_s=args.wait_s)
    started = time.perf_counter()
    first = second = None
    cleanup_before = {}
    cleanup_after = {}
    try:
        first, first_dedup = manager.submit(args.video, "a person performing an action", TaskMode.burst_async, request_id=f"real-runtime-first-{uuid.uuid4().hex}")
        second, second_dedup = manager.submit(args.video, "a person performing an action", TaskMode.burst_async, request_id=f"real-runtime-second-{uuid.uuid4().hex}")
        before_cancel = {"first": snapshot(first), "second": snapshot(second), "first_deduplicated": first_dedup, "second_deduplicated": second_dedup}
        cancelled = manager.cancel(second.task_id)
        after_cancel = {"first": snapshot(first), "second": snapshot(cancelled)}
        first_done = manager.wait(first.task_id, args.wait_s + 30.0)
        second_done = manager.wait(second.task_id, 5.0)
        try:
            import torch

            torch.cuda.synchronize()
            cleanup_before = {
                "cuda_allocated_gib": torch.cuda.memory_allocated(torch.device("cuda:0")) / 2**30,
                "cuda_reserved_gib": torch.cuda.memory_reserved(torch.device("cuda:0")) / 2**30,
            }
        except Exception as exc:
            cleanup_before = {"error": f"{type(exc).__name__}: {exc}"}
        result = {
            "run_id": f"g6_real_model_runtime_matrix_{time.strftime('%Y%m%d_%H%M%S')}",
            "video": str(args.video),
            "total_pixels": args.total_pixels,
            "worker_count": 1,
            "before_cancel": before_cancel,
            "after_cancel": after_cancel,
            "first_terminal": snapshot(first_done),
            "second_terminal": snapshot(second_done),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "multi_worker_status": "not_run_single_24GiB_card_cannot_safely_host_two_TimeLens_workers",
            "scope": "real_model_single_worker_queue_cancel_resource_cleanup_not_multi_gpu_throughput",
        }
    finally:
        manager.shutdown(wait=True)
        result = locals().get("result", {"run_id": f"g6_real_model_runtime_matrix_{time.strftime('%Y%m%d_%H%M%S')}", "error": "matrix did not reach terminal collection"})
        result["threads_alive_after_shutdown"] = [thread.name for thread in manager._threads if thread.is_alive()]
        result["prefetch_executor_shutdown"] = getattr(manager._prefetch_executor, "_shutdown", None)
        try:
            import torch

            grounder._model = None
            grounder._processor = None
            encoder._model = None
            encoder._processor = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            cleanup_after = {
                "cuda_allocated_gib": torch.cuda.memory_allocated(torch.device("cuda:0")) / 2**30,
                "cuda_reserved_gib": torch.cuda.memory_reserved(torch.device("cuda:0")) / 2**30,
            }
        except Exception as exc:
            cleanup_after = {"error": f"{type(exc).__name__}: {exc}"}
        result["cleanup_before"] = cleanup_before
        result["cleanup_after"] = cleanup_after
        result["passed"] = (
            result.get("first_terminal", {}).get("status") == "succeeded"
            and result.get("second_terminal", {}).get("status") == "cancelled"
            and not result.get("threads_alive_after_shutdown")
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"saved: {args.output}")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
