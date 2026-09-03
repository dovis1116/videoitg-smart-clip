#!/usr/bin/env python
"""Apply a train/dev-selected temporal post-processor to a frame-score dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from videoitg_smart_clip.evaluation.metrics import evaluate_sample
from evaluate_boundary_postprocess import predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["raw", "mean_r1", "mean_r2", "neighbor_sum", "neighbor_max", "asymmetric_sum"], required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--margin-threshold", type=float)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for row in rows:
        pred = predictions(row, args.method, args.alpha, args.margin_threshold, args.beta)
        item = {key: row[key] for key in row if key not in {"predictions", "metrics"}}
        item["predictions"] = pred
        item["metrics"] = evaluate_sample(pred, row["ground_truth_segments"], output_top_k=3)
        item["postprocess"] = {"method": args.method, "alpha": args.alpha, "beta": args.beta, "margin_threshold": args.margin_threshold}
        out.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in out) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(out), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
