#!/usr/bin/env python
"""Compare paired fixed-frame predictions without tuning a policy."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "recall_at_1_iou_0.3",
    "recall_at_1_iou_0.5",
    "topk_hit_iou_0.3",
    "topk_hit_iou_0.5",
)


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--candidate-label", default="candidate")
    args = parser.parse_args()

    base, candidate = load(args.base), load(args.candidate)
    ids = sorted(set(base) & set(candidate))
    if set(base) != set(candidate):
        raise ValueError("paired runs do not contain the same sample ids")
    report = {
        "scope": {"sample_count": len(ids), "base": str(args.base), "candidate": str(args.candidate)},
        "metrics": {},
        "by_source_group": {},
        "latency": {},
        "decision": "quality_difference_is_paired_and_not_policy_tuned",
    }
    for metric in METRICS:
        gains = losses = 0
        base_mean = sum(float(base[s]["metrics"][metric]) for s in ids) / len(ids)
        cand_mean = sum(float(candidate[s]["metrics"][metric]) for s in ids) / len(ids)
        for sid in ids:
            x = base[sid]["metrics"][metric]
            y = candidate[sid]["metrics"][metric]
            gains += y > x
            losses += y < x
        report["metrics"][metric] = {
            args.base_label: base_mean,
            args.candidate_label: cand_mean,
            "delta": cand_mean - base_mean,
            "paired_gains": gains,
            "paired_losses": losses,
            "unchanged": len(ids) - gains - losses,
        }

    grouped: dict[str, list[str]] = defaultdict(list)
    for sid in ids:
        grouped[base[sid]["video_id"].split("/")[1] if "/" in base[sid]["video_id"] else "unknown"].append(sid)
    for group, group_ids in sorted(grouped.items()):
        report["by_source_group"][group] = {
            "sample_count": len(group_ids),
            "topk_hit_iou_0.5": {
                args.base_label: sum(base[s]["metrics"]["topk_hit_iou_0.5"] for s in group_ids) / len(group_ids),
                args.candidate_label: sum(candidate[s]["metrics"]["topk_hit_iou_0.5"] for s in group_ids) / len(group_ids),
            },
        }
        report["by_source_group"][group]["topk_hit_iou_0.5"]["delta"] = (
            report["by_source_group"][group]["topk_hit_iou_0.5"][args.candidate_label]
            - report["by_source_group"][group]["topk_hit_iou_0.5"][args.base_label]
        )

    base_latency = [float(base[s]["runtime"]["rerank_seconds"]) for s in ids]
    cand_latency = [float(candidate[s]["runtime"]["rerank_seconds"]) for s in ids]
    report["latency"] = {
        args.base_label: {"mean_s": statistics.mean(base_latency), "p95_s": sorted(base_latency)[int(0.95 * (len(ids) - 1))]},
        args.candidate_label: {"mean_s": statistics.mean(cand_latency), "p95_s": sorted(cand_latency)[int(0.95 * (len(ids) - 1))]},
    }
    report["latency"]["relative_mean_increase"] = report["latency"][args.candidate_label]["mean_s"] / report["latency"][args.base_label]["mean_s"] - 1.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_count": len(ids), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
