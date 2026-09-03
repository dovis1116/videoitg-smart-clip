#!/usr/bin/env python
"""Run a real, single-GPU request benchmark for fixed and queue-bypass policies.

This deliberately benchmarks one worker and one process.  It is not a service
implementation, but unlike ``replay_latency_policy.py`` every request executes
the local CLIP/VideoITG model and its measured wall time is recorded.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


def read_manifest(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("raw_video_present", True) and Path(row["video_path"]).is_file()]
    return rows[:limit] if limit else rows


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q))] if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["E0_fixed16", "E3_queue_bypass"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videoitg-model", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arrival-gap-ms", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=float, default=5000.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", action="store_true", help="run one unmeasured request for each loaded model before timing")
    parser.add_argument(
        "--retain-cuda-cache",
        action="store_true",
        help="do not call torch.cuda.empty_cache() after each VideoITG candidate; quality is unchanged",
    )
    parser.add_argument(
        "--prefetch-read",
        action="store_true",
        help="prefetch the next full-video frame batch on CPU while the current request runs on GPU",
    )
    args = parser.parse_args()
    if args.arrival_gap_ms < 0 or args.deadline_ms <= 0:
        raise SystemExit("arrival gap must be non-negative and deadline must be positive")
    if args.prefetch_read and args.policy != "E0_fixed16":
        raise SystemExit("--prefetch-read is only supported with --policy E0_fixed16")

    import torch

    torch.cuda.set_device(args.device) if args.device.startswith("cuda") else None
    from videoitg_smart_clip.baselines.common import uniform_candidates
    from videoitg_smart_clip.baselines.clip_retriever import ClipRetriever
    from videoitg_smart_clip.evaluation.metrics import evaluate_sample, gt_segments_from_clip_num
    from videoitg_smart_clip.reranker import CandidateSegment, VideoITGReranker
    from videoitg_smart_clip.budgeting.policy import BudgetPolicy
    from run_baseline import frame_score_predictions

    rows = read_manifest(args.manifest, args.limit)
    if not rows:
        raise SystemExit("manifest has no existing media rows")
    reranker = VideoITGReranker(
        str(args.videoitg_model),
        device=args.device,
        target_fps=2.0,
        max_frames_per_candidate=16,
        frame_score_topk=8,
        empty_cache_each_candidate=not args.retain_cuda_cache,
    )
    retriever = ClipRetriever(str(args.clip_model), args.device, 16) if args.policy == "E3_queue_bypass" else None
    policy = BudgetPolicy(policy_version="phase5-e3-real-v1", bypass_queue_threshold=1) if retriever else BudgetPolicy(policy_version="phase5-e0-real-v1")
    if args.warmup:
        warm = rows[0]
        reranker.rank(warm["query"], [CandidateSegment(warm["video_path"], 0.0, None, candidate_id="warmup")])
        if retriever is not None:
            retriever.rank(warm["query"], warm["video_path"], uniform_candidates(warm["video_path"], 5.0, 64))
    prefetch_executor = ThreadPoolExecutor(max_workers=1) if args.prefetch_read and retriever is None else None
    prefetch_future: Future | None = None
    if prefetch_executor is not None:
        first_candidate = CandidateSegment(rows[0]["video_path"], 0.0, None, candidate_id="full")
        prefetch_future = prefetch_executor.submit(reranker.read_candidate, first_candidate)
    virtual_available_ms = 0.0
    started_wall = time.perf_counter()
    results = []
    try:
        for index, row in enumerate(rows):
            arrival_ms = index * args.arrival_gap_ms
            if args.arrival_gap_ms > 0:
                elapsed_ms = (time.perf_counter() - started_wall) * 1000.0
                if elapsed_ms < arrival_ms:
                    time.sleep((arrival_ms - elapsed_ms) / 1000.0)
            queue_length = int(virtual_available_ms > arrival_ms)
            decision = policy.choose(queue_length=queue_length, deadline_ms=args.deadline_ms)
            mode = "retrieval_only" if decision.mode == "retrieval_only" else "fixed_16"
            request_started = time.perf_counter()
            prepared = None
            candidate = None
            if mode == "retrieval_only":
                candidates = uniform_candidates(row["video_path"], 5.0, 64)
                scored = retriever.rank(row["query"], row["video_path"], candidates)
                predictions = [
                    {"candidate_id": item["candidate_id"], "start_s": item["start_s"], "end_s": item["end_s"], "score": item["score"]}
                    for item in scored[:3]
                ]
            else:
                candidate = CandidateSegment(row["video_path"], 0.0, None, candidate_id="full")
                if prefetch_future is not None:
                    prepared = prefetch_future.result()
                if prefetch_executor is not None and index + 1 < len(rows):
                    next_candidate = CandidateSegment(rows[index + 1]["video_path"], 0.0, None, candidate_id="full")
                    prefetch_future = prefetch_executor.submit(reranker.read_candidate, next_candidate)
                elif prefetch_executor is not None:
                    prefetch_future = None
                prepared_map = {candidate: prepared} if prepared is not None else None
                scored = reranker.rank(row["query"], [candidate], prepared_candidates=prepared_map)[0]
                fps = float(scored.runtime["fps"])
                duration = float(scored.runtime["duration_s"])
                predictions = frame_score_predictions(scored.frame_scores, fps, duration, 5.0, 3)
            service_ms = (time.perf_counter() - request_started) * 1000.0
            start_ms = max(arrival_ms, virtual_available_ms)
            end_ms = start_ms + service_ms
            virtual_available_ms = end_ms
            total_ms = end_ms - arrival_ms
            metrics = evaluate_sample(predictions, gt_segments_from_clip_num(row["clip_num"]), output_top_k=3)
            results.append({
                "sample_id": row["sample_id"],
                "policy": args.policy,
                "mode": mode,
                "arrival_ms": arrival_ms,
                "queue_delay_ms": start_ms - arrival_ms,
                "service_ms": service_ms,
                "total_ms": total_ms,
                "timed_out": total_ms > args.deadline_ms,
                "queue_length": queue_length,
                "max_frames": decision.max_frames,
                "reason": decision.reason,
                "prefetched": bool(scored.runtime.get("prefetched", 0)) if mode != "retrieval_only" else False,
                "metrics": metrics,
            })
            print(f"{args.policy} {index + 1}/{len(rows)} mode={mode} service_ms={service_ms:.1f} total_ms={total_ms:.1f}")
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
    total_values = [item["total_ms"] for item in results]
    service_values = [item["service_ms"] for item in results]
    report = {
        "status": "real_single_gpu_benchmark",
        "policy": args.policy,
        "workers": 1,
        "manifest": str(args.manifest),
        "request_count": len(results),
        "arrival_gap_ms": args.arrival_gap_ms,
        "deadline_ms": args.deadline_ms,
        "warmup": args.warmup,
        "retain_cuda_cache": args.retain_cuda_cache,
        "prefetch_read": args.prefetch_read,
        "model": str(args.videoitg_model),
        "summary": {
            "p50_total_ms": percentile(total_values, 0.50),
            "p95_total_ms": percentile(total_values, 0.95),
            "mean_total_ms": sum(total_values) / len(total_values),
            "p95_service_ms": percentile(service_values, 0.95),
            "timeout_count": sum(item["timed_out"] for item in results),
            "mode_counts": {mode: sum(item["mode"] == mode for item in results) for mode in ("fixed_16", "retrieval_only")},
            "quality": {key: sum(item["metrics"][key] for item in results) / len(results) for key in ("recall_at_1_iou_0.5", "topk_hit_iou_0.5", "max_iou_topk")},
        },
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
