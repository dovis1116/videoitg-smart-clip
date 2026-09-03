#!/usr/bin/env python3
"""Exercise the CPU stub runtime's terminal, overload, and cleanup matrix.

This is a runtime-contract artifact. It deliberately does not probe CUDA or
claim real TimeLens concurrency; those require a responsive GPU driver.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from videoitg_smart_clip.service.models import TaskMode, TaskStatus
from videoitg_smart_clip.service.runtime import BoundedTaskManager, QueueFullError, StubWorker


def _wait(manager: BoundedTaskManager, task_id: str, timeout_s: float = 2.0):
    return manager.wait(task_id, timeout_s)


def _snapshot(record) -> dict:
    return {
        "status": record.status.value,
        "current_stage": record.current_stage,
        "error_code": record.error_code,
        "degraded": record.degraded,
        "degrade_level": (record.result or {}).get("degrade_level") if record.result else None,
        "result_present": record.result is not None,
    }


def _cleanup(manager: BoundedTaskManager) -> dict:
    manager.shutdown()
    return {
        "worker_threads_alive_after_shutdown": sum(thread.is_alive() for thread in manager._threads),
        "prefetch_executor_shutdown": bool(getattr(manager._prefetch_executor, "_shutdown", False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="videoitg_service_matrix_") as temp:
        video = Path(temp) / "sample.mp4"
        video.write_bytes(b"stub")

        manager = BoundedTaskManager([StubWorker(delay_s=0.01)], queue_size=2)
        try:
            record, deduplicated = manager.submit(video, "success", TaskMode.burst_async, request_id="success-1")
            same, same_deduplicated = manager.submit(video, "success", TaskMode.burst_async, request_id="success-1")
            finished = _wait(manager, record.task_id)
            results["success_and_idempotency"] = {
                **_snapshot(finished),
                "initial_deduplicated": deduplicated,
                "repeat_deduplicated": same_deduplicated,
                "same_task_id": same.task_id == record.task_id,
            }
        finally:
            results["success_and_idempotency"].update(_cleanup(manager))

        class FailingWorker(StubWorker):
            def run(self, video_path: Path, query: str) -> dict:
                raise RuntimeError("synthetic backend failure")

        manager = BoundedTaskManager([FailingWorker()], queue_size=1)
        try:
            record, _ = manager.submit(video, "failure", TaskMode.burst_async)
            results["backend_failure"] = _snapshot(_wait(manager, record.task_id))
        finally:
            results["backend_failure"].update(_cleanup(manager))

        manager = BoundedTaskManager([StubWorker(delay_s=0.04)], queue_size=1, task_timeout_s=0.005)
        try:
            record, _ = manager.submit(video, "timeout", TaskMode.burst_async)
            results["async_timeout"] = _snapshot(_wait(manager, record.task_id))
        finally:
            results["async_timeout"].update(_cleanup(manager))

        manager = BoundedTaskManager([StubWorker(delay_s=0.04)], queue_size=1)
        try:
            record, _ = manager.submit(video, "cancel", TaskMode.burst_async)
            deadline = time.perf_counter() + 1.0
            while manager.get(record.task_id).status is not TaskStatus.running and time.perf_counter() < deadline:
                time.sleep(0.001)
            cancelled = manager.cancel(record.task_id)
            finished = _wait(manager, record.task_id)
            results["running_cancel"] = {
                **_snapshot(finished),
                "cancel_requested": cancelled.cancel_requested,
            }
        finally:
            results["running_cancel"].update(_cleanup(manager))

        manager = BoundedTaskManager([StubWorker(delay_s=0.12)], queue_size=1)
        try:
            first, _ = manager.submit(video, "queue-first", TaskMode.burst_async)
            deadline = time.perf_counter() + 1.0
            while manager.get(first.task_id).status is not TaskStatus.running and time.perf_counter() < deadline:
                time.sleep(0.001)
            second, _ = manager.submit(video, "queue-second", TaskMode.burst_async)
            rejected = False
            try:
                manager.submit(video, "queue-third", TaskMode.burst_async)
            except QueueFullError:
                rejected = True
            _wait(manager, first.task_id)
            _wait(manager, second.task_id)
            results["bounded_queue"] = {
                "third_request_rejected": rejected,
                "first_status": manager.get(first.task_id).status.value,
                "second_status": manager.get(second.task_id).status.value,
            }
        finally:
            results["bounded_queue"].update(_cleanup(manager))

        class OOMOnceWorker(StubWorker):
            def __init__(self):
                super().__init__()
                self.failed_once = False

            def run(self, video_path: Path, query: str) -> dict:
                if not self.failed_once:
                    self.failed_once = True
                    raise RuntimeError("CUDA out of memory (synthetic matrix)")
                return super().run(video_path, query)

        manager = BoundedTaskManager([OOMOnceWorker()], queue_size=2)
        try:
            first, _ = manager.submit(video, "oom-first", TaskMode.burst_async)
            second, _ = manager.submit(video, "oom-second", TaskMode.burst_async)
            first_done = _wait(manager, first.task_id)
            second_done = _wait(manager, second.task_id)
            results["oom_then_recovery"] = {
                "first": _snapshot(first_done),
                "second": _snapshot(second_done),
            }
        finally:
            results["oom_then_recovery"].update(_cleanup(manager))

        workers = [StubWorker(delay_s=0.01), StubWorker(delay_s=0.01)]
        manager = BoundedTaskManager(workers, queue_size=8)
        try:
            records = [manager.submit(video, f"concurrent-{index}", TaskMode.burst_async)[0] for index in range(8)]
            finished = [_wait(manager, record.task_id) for record in records]
            results["two_worker_concurrency"] = {
                "submitted": len(records),
                "success_count": sum(item.status is TaskStatus.succeeded for item in finished),
                "terminal_statuses": [item.status.value for item in finished],
            }
        finally:
            results["two_worker_concurrency"].update(_cleanup(manager))

    report = {
        "run_id": f"service_contract_matrix_{time.strftime('%Y%m%d_%H%M%S')}",
        "scope": "cpu_stub_runtime_contract_not_real_model_throughput",
        "results": results,
        "limitations": [
            "No CUDA or nvidia-smi probe was executed.",
            "Stub timing and concurrency do not represent TimeLens-8B latency, GPU memory, or cancellation behavior.",
            "Real GPU resource-release validation remains pending driver recovery.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
