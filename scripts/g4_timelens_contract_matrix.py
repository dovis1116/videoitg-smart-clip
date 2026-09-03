#!/usr/bin/env python3
"""Exercise TimeLens adapter parsing and window remapping without CUDA/model IO."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from videoitg_smart_clip.grounding import TimeLensGrounder
from videoitg_smart_clip.pipeline import CandidateWindow


class _FakeGrounder(TimeLensGrounder):
    def __init__(self, answers: list[str], *, batch_size: int, wrong_count: bool = False):
        super().__init__(model_path=Path("/tmp/fake-timelens"), device="cpu", batch_size=batch_size)
        self.answers = list(answers)
        self.wrong_count = wrong_count

    def _load(self) -> None:
        self._model = object()

    def _materialize_window(self, video_path, window, directory):
        clip = directory / f"{window.candidate_id}.mp4"
        clip.write_bytes(b"fake clip")
        return clip

    def _infer_window(self, clip_path, query):
        return self.answers.pop(0)

    def _infer_batch(self, clips, query):
        values = [self.answers.pop(0) for _ in clips]
        return values[:-1] if self.wrong_count else values


def _run_case(name: str, grounder: _FakeGrounder, windows: list[CandidateWindow]) -> dict:
    try:
        predictions = grounder.predict("fake.mp4", "find event", windows)
        return {
            "case": name,
            "status": "passed",
            "prediction_count": len(predictions),
            "bounds": [{"candidate_id": p.candidate_id, "start": p.raw_start, "end": p.raw_end} for p in predictions],
        }
    except Exception as exc:
        return {"case": name, "status": "expected_error", "error_type": type(exc).__name__, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = [CandidateWindow(10.0, 20.0, 0.8, "a"), CandidateWindow(30.0, 35.0, 0.7, "b")]
    cases = [
        _run_case("serial_batch_size_1", _FakeGrounder(["The event happens in 1 - 3 seconds", "0.5 - 2 seconds"], batch_size=1), windows),
        _run_case("true_batch_size_2", _FakeGrounder(["1 - 3 seconds", "0.5 - 2 seconds"], batch_size=2), windows),
        _run_case("local_timestamp_clamp", _FakeGrounder(["2 - 99 seconds"], batch_size=1), [CandidateWindow(40.0, 45.0, 0.5, "clamp")]),
        _run_case("invalid_range", _FakeGrounder(["5 - 2 seconds"], batch_size=1), [CandidateWindow(0.0, 5.0, 0.5, "invalid")]),
        _run_case("wrong_batch_output_count", _FakeGrounder(["1 - 2 seconds", "1 - 2 seconds"], batch_size=2, wrong_count=True), windows),
    ]
    report = {
        "run_id": f"g4_timelens_contract_matrix_{time.strftime('%Y%m%d_%H%M%S')}",
        "scope": "adapter_contract_only_fake_inference_no_cuda",
        "cases": cases,
        "expected_error_cases": ["invalid_range", "wrong_batch_output_count"],
        "limitations": [
            "Fake inference bypasses checkpoint loading, video decoding, GPU memory, latency, and real model failure behavior.",
            "Real TimeLens failure/timeout/OOM/window-budget matrix remains pending GPU recovery.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
