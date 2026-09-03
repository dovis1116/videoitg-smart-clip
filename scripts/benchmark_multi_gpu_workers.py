#!/usr/bin/env python
"""Benchmark a shared request queue with one VideoITG worker per local GPU.

This is a capacity experiment, not the Phase 6 service.  Each worker owns one
model instance and consumes the same fixed16 request protocol from a shared
queue.  The parent controls request arrival times and computes queue/tail
metrics from worker timestamps.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


def read_manifest(path: Path, limit: int | None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        row
        for row in rows
        if row.get("raw_video_present", True) and Path(row["video_path"]).is_file()
    ]
    return rows[:limit] if limit else rows


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q))] if values else 0.0


def worker_loop(
    worker_id: int,
    device: str,
    jobs,
    results,
    ready,
    model_path: str,
    warmup_row: dict | None,
) -> None:
    """Load one model and process jobs until a sentinel is received."""

    try:
        import torch

        torch.cuda.set_device(device) if device.startswith("cuda") else None
        from run_baseline import frame_score_predictions
        from videoitg_smart_clip.evaluation.metrics import evaluate_sample, gt_segments_from_clip_num
        from videoitg_smart_clip.reranker import CandidateSegment, VideoITGReranker

        reranker = VideoITGReranker(
            model_path,
            device=device,
            target_fps=2.0,
            max_frames_per_candidate=16,
            frame_score_topk=8,
        )
        if warmup_row is not None:
            warmup = CandidateSegment(warmup_row["video_path"], 0.0, None, candidate_id="warmup")
            reranker.rank(warmup_row["query"], [warmup])
        ready.put({"worker_id": worker_id, "status": "ready", "device": device})
    except Exception as exc:  # pragma: no cover - exercised by spawned runtime
        ready.put({"worker_id": worker_id, "status": "error", "device": device, "error": repr(exc)})
        return

    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    job = jobs.get()
    prepared_future: Future | None = None
    try:
        while job is not None:
            row = job["row"]
            arrival_ms = float(job["arrival_ms"])
            arrival_abs = float(job["arrival_abs"])

            # Take one queued job as lookahead and decode it while the current
            # request occupies the GPU. If the queue is empty, keep the current
            # job's normal synchronous read path.
            next_job = None
            next_future: Future | None = None
            stop_after_current = False
            try:
                lookahead = jobs.get_nowait()
                if lookahead is None:
                    stop_after_current = True
                else:
                    next_job = lookahead
                    next_candidate = CandidateSegment(next_job["row"]["video_path"], 0.0, None, candidate_id="full")
                    next_future = prefetch_executor.submit(reranker.read_candidate, next_candidate)
            except queue.Empty:
                pass

            started = time.perf_counter()
            try:
                candidate = CandidateSegment(row["video_path"], 0.0, None, candidate_id="full")
                prepared = prepared_future.result() if prepared_future is not None else None
                prepared_map = {candidate: prepared} if prepared is not None else None
                scored = reranker.rank(row["query"], [candidate], prepared_candidates=prepared_map)[0]
                fps = float(scored.runtime["fps"])
                duration = float(scored.runtime["duration_s"])
                predictions = frame_score_predictions(scored.frame_scores, fps, duration, 5.0, 3)
                metrics = evaluate_sample(predictions, gt_segments_from_clip_num(row["clip_num"]), output_top_k=3)
                finished = time.perf_counter()
                results.put(
                    {
                        "index": int(job["index"]),
                        "sample_id": row["sample_id"],
                        "worker_id": worker_id,
                        "device": device,
                        "arrival_ms": arrival_ms,
                        "queue_delay_ms": (started - arrival_abs) * 1000.0,
                        "service_ms": (finished - started) * 1000.0,
                        "total_ms": (finished - arrival_abs) * 1000.0,
                        "timed_out": (finished - arrival_abs) * 1000.0 > float(job["deadline_ms"]),
                        "mode": "fixed_16",
                        "max_frames": 16,
                        "prefetched": bool(scored.runtime.get("prefetched", 0)),
                        "metrics": metrics,
                    }
                )
            except Exception as exc:  # pragma: no cover - exercised by spawned runtime
                finished = time.perf_counter()
                results.put(
                    {
                        "index": int(job["index"]),
                        "sample_id": row.get("sample_id", ""),
                        "worker_id": worker_id,
                        "device": device,
                        "arrival_ms": arrival_ms,
                        "queue_delay_ms": (started - arrival_abs) * 1000.0,
                        "service_ms": (finished - started) * 1000.0,
                        "total_ms": (finished - arrival_abs) * 1000.0,
                        "timed_out": True,
                        "mode": "error",
                        "max_frames": 16,
                        "error": repr(exc),
                    }
                )
            prepared_future = next_future
            if next_job is not None:
                job = next_job
            elif stop_after_current:
                break
            else:
                job = jobs.get()
                prepared_future = None
    finally:
        prefetch_executor.shutdown(wait=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videoitg-model", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="comma-separated devices, e.g. cuda:0,cuda:1")
    parser.add_argument("--arrival-gap-ms", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=float, default=5000.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    if args.arrival_gap_ms < 0 or args.deadline_ms <= 0:
        raise SystemExit("arrival gap must be non-negative and deadline must be positive")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("--devices must contain at least one device")

    rows = read_manifest(args.manifest, args.limit)
    if not rows:
        raise SystemExit("manifest has no existing media rows")

    ctx = mp.get_context("spawn")
    jobs = ctx.Queue()
    results = ctx.Queue()
    ready = ctx.Queue()
    processes = [
        ctx.Process(
            target=worker_loop,
            args=(worker_id, device, jobs, results, ready, str(args.videoitg_model), rows[0] if args.warmup else None),
        )
        for worker_id, device in enumerate(devices)
    ]
    for process in processes:
        process.start()

    ready_messages = []
    try:
        for _ in processes:
            ready_messages.append(ready.get(timeout=300))
        errors = [item for item in ready_messages if item.get("status") != "ready"]
        if errors:
            raise RuntimeError(f"worker startup failed: {errors}")

        # Start the measured arrival clock only after every model is loaded and
        # warmed up; cold-start time is reported separately by the caller.
        benchmark_start = time.perf_counter()
        for index, row in enumerate(rows):
            arrival_ms = index * args.arrival_gap_ms
            target = benchmark_start + arrival_ms / 1000.0
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            jobs.put(
                {
                    "index": index,
                    "row": row,
                    "arrival_ms": arrival_ms,
                    "arrival_abs": target,
                    "deadline_ms": args.deadline_ms,
                }
            )
        for _ in processes:
            jobs.put(None)

        received = []
        while len(received) < len(rows):
            try:
                received.append(results.get(timeout=300))
            except queue.Empty as exc:
                raise RuntimeError("timed out waiting for worker results") from exc
    finally:
        for process in processes:
            process.join(timeout=30)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

    received.sort(key=lambda item: item["index"])
    service_values = [float(item["service_ms"]) for item in received]
    total_values = [float(item["total_ms"]) for item in received]
    valid = [item for item in received if "metrics" in item]
    quality = {
        key: sum(float(item["metrics"][key]) for item in valid) / len(valid)
        for key in ("recall_at_1_iou_0.5", "topk_hit_iou_0.5", "max_iou_topk")
    } if valid else {}
    report = {
        "status": "real_multi_gpu_queue_benchmark",
        "policy": "E0_fixed16",
        "manifest": str(args.manifest),
        "model": str(args.videoitg_model),
        "devices": devices,
        "workers": len(devices),
        "request_count": len(received),
        "arrival_gap_ms": args.arrival_gap_ms,
        "deadline_ms": args.deadline_ms,
        "warmup": args.warmup,
        "ready": ready_messages,
        "summary": {
            "p50_total_ms": percentile(total_values, 0.50),
            "p95_total_ms": percentile(total_values, 0.95),
            "mean_total_ms": sum(total_values) / len(total_values),
            "p95_service_ms": percentile(service_values, 0.95),
            "timeout_count": sum(bool(item["timed_out"]) for item in received),
            "error_count": sum(item.get("mode") == "error" for item in received),
            "worker_counts": {str(worker_id): sum(item["worker_id"] == worker_id for item in received) for worker_id in range(len(devices))},
            "quality": quality,
        },
        "rows": received,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
