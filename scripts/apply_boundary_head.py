#!/usr/bin/env python
"""Apply a frozen boundary-head model to frame-score rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from videoitg_smart_clip.boundary_head import calibrated_predictions
from videoitg_smart_clip.evaluation.metrics import evaluate_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for row in rows:
        predictions = calibrated_predictions(row, model)
        item = {key: row[key] for key in row if key not in {"predictions", "metrics"}}
        item["predictions"] = predictions
        item["metrics"] = evaluate_sample(predictions, row["ground_truth_segments"], output_top_k=3)
        item["postprocess"] = {"type": model["format"], "model": str(args.model)}
        output.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
