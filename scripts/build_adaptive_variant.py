#!/usr/bin/env python
"""Build an offline adaptive B1 variant from narrow/wide prediction dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from videoitg_smart_clip.evaluation.metrics import aggregate_metrics, evaluate_sample


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--narrow", type=Path, required=True)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    narrow = load_jsonl(args.narrow)
    wide = load_jsonl(args.wide)
    results = []
    selected = 0
    for sample_id, base in narrow.items():
        candidate = wide[sample_id]
        top1_score = float(base["predictions"][0]["score"]) if base.get("predictions") else float("-inf")
        use_wide = top1_score < args.threshold
        chosen = candidate if use_wide else base
        predictions = chosen["predictions"]
        metrics = evaluate_sample(predictions, base["ground_truth_segments"], output_top_k=3)
        results.append({
            "sample_id": sample_id,
            "video_id": base["video_id"],
            "split": base.get("split", "unknown"),
            "query": base["query"],
            "baseline": "B1A",
            "ground_truth_segments": base["ground_truth_segments"],
            "predictions": predictions,
            "metrics": metrics,
            "adaptive": {"threshold": args.threshold, "narrow_top1_score": top1_score, "selected_variant": "wide_8s" if use_wide else "narrow_5s"},
        })
        selected += int(use_wide)
    results.sort(key=lambda row: row["sample_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")
    summary = {
        "run": {"baseline": "B1A", "narrow": str(args.narrow), "wide": str(args.wide), "threshold": args.threshold, "sample_count": len(results)},
        "metrics": aggregate_metrics([row["metrics"] for row in results]),
        "selection": {"wide_count": selected, "narrow_count": len(results) - selected, "wide_fraction": selected / max(1, len(results))},
        "output": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
