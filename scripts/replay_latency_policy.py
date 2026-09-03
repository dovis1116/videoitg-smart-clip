#!/usr/bin/env python
"""Replay measured per-sample runtimes under an explicit budget policy.

This is a queueing simulation, not a service benchmark.  It uses sequential
pilot runtimes from completed runs and reports policy mechanics only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from videoitg_smart_clip.budgeting.policy import BudgetPolicy


def load_jsonl(path: Path) -> dict[str, dict]:
    return {row["sample_id"]: row for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    return values[min(len(values) - 1, int((len(values) - 1) * q))]


def uncertainty(row: dict) -> float:
    scores = [float(item["score"]) for item in row.get("predictions", [])]
    if len(scores) < 2:
        return 1.0
    # This is an explicitly documented proxy for replay mechanics only; it is
    # not a calibrated uncertainty estimate and is not used for G4 claims.
    return max(0.0, min(1.0, 1.0 - (scores[0] - scores[1])))


def simulate(
    rows: list[dict],
    runtimes: dict[str, dict[str, float]],
    policy: BudgetPolicy,
    *,
    workers: int,
    arrival_mode: str,
    arrival_gap_ms: float,
    deadline_ms: float | None,
) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    availability = [0.0] * workers
    scheduled: list[dict] = []
    for index, row in enumerate(rows):
        arrival = 0.0 if arrival_mode == "burst" else index * arrival_gap_ms
        queue_length = sum(end > arrival for end in availability)
        decision = policy.choose(queue_length=queue_length, uncertainty=uncertainty(row), deadline_ms=deadline_ms)
        mode = {"fixed": "fixed_16", "reduced": "fixed_32", "full": "fixed_64", "retrieval_only": "retrieval_only"}[decision.mode]
        duration = runtimes[row["sample_id"]][mode]
        worker = min(range(workers), key=lambda i: availability[i])
        start = max(arrival, availability[worker])
        end = start + duration
        availability[worker] = end
        scheduled.append({
            "sample_id": row["sample_id"],
            "arrival_ms": arrival,
            "start_ms": start,
            "end_ms": end,
            "queue_delay_ms": start - arrival,
            "total_ms": end - arrival,
            "timed_out": deadline_ms is not None and end - arrival > deadline_ms,
            "mode": mode,
            "max_frames": decision.max_frames,
            "reason": decision.reason,
        })
    totals = [item["total_ms"] for item in scheduled]
    return {
        "workers": workers,
        "arrival_mode": arrival_mode,
        "arrival_gap_ms": arrival_gap_ms,
        "deadline_ms": deadline_ms,
        "policy_version": policy.policy_version,
        "request_count": len(scheduled),
        "p50_total_ms": percentile(totals, 0.50),
        "p95_total_ms": percentile(totals, 0.95),
        "mean_total_ms": statistics.mean(totals) if totals else 0.0,
        "max_total_ms": max(totals) if totals else 0.0,
        "timeout_count": sum(item["timed_out"] for item in scheduled),
        "mode_counts": {mode: sum(item["mode"] == mode for item in scheduled) for mode in ("fixed_16", "fixed_32", "fixed_64", "retrieval_only")},
        "mean_max_frames": statistics.mean(item["max_frames"] for item in scheduled) if scheduled else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot16", type=Path, required=True)
    parser.add_argument("--pilot32", type=Path, required=True)
    parser.add_argument("--pilot64", type=Path, required=True)
    parser.add_argument("--pilot0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-ms", type=float, default=5000.0)
    args = parser.parse_args()

    p16, p32, p64, p0 = map(load_jsonl, (args.pilot16, args.pilot32, args.pilot64, args.pilot0))
    common_ids = sorted(set(p16) & set(p32) & set(p64) & set(p0))
    rows = [p16[sid] for sid in common_ids]
    runtimes = {
        sid: {
            "fixed_16": p16[sid]["runtime"]["rerank_seconds"] * 1000.0,
            "fixed_32": p32[sid]["runtime"]["rerank_seconds"] * 1000.0,
            "fixed_64": p64[sid]["runtime"]["rerank_seconds"] * 1000.0,
            "retrieval_only": p0[sid]["runtime"].get("retrieval_seconds", 1000.0) * 1000.0,
        }
        for sid in common_ids
    }
    policies = {
        "E0_fixed16": BudgetPolicy(policy_version="phase5-e0-fixed16"),
        "E2_preflight": BudgetPolicy(
            policy_version="phase5-e2-preflight-v0",
            full_uncertainty_threshold=0.8,
            reduced_queue_threshold=1,
            bypass_queue_threshold=4,
        ),
    }
    simulations = []
    for name, policy in policies.items():
        for workers in (1, 2, 4):
            simulations.append({"policy": name, **simulate(rows, runtimes, policy, workers=workers, arrival_mode="burst", arrival_gap_ms=0.0, deadline_ms=args.deadline_ms)})
            simulations.append({"policy": name, **simulate(rows, runtimes, policy, workers=workers, arrival_mode="steady", arrival_gap_ms=1000.0, deadline_ms=args.deadline_ms)})
    report = {
        "status": "simulation_only",
        "quality_not_simulated": True,
        "source_rows": len(rows),
        "source_runs": {"fixed16": str(args.pilot16), "fixed32": str(args.pilot32), "fixed64": str(args.pilot64), "retrieval": str(args.pilot0)},
        "uncertainty_proxy": "1 - clipped(top1_score - top2_score); mechanics-only, not calibrated",
        "simulations": simulations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "source_rows": len(rows), "simulations": len(simulations)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
