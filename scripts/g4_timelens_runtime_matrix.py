#!/usr/bin/env python3
"""Run a reproducible real-TimeLens window/budget/failure matrix.

Each case is a separate process so a failed or high-memory probe cannot
contaminate the following case.  The matrix is an operational contract probe,
not a grounding-quality evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _case_definitions() -> list[dict]:
    return [
        {"case_id": "one_window_4m", "start": 0.0, "end": 10.0, "count": 1, "batch_size": 1, "total_pixels": 4194304, "max_new_tokens": 128, "expected": "succeeded"},
        {"case_id": "two_window_serial_4m", "start": 0.0, "end": 5.0, "count": 2, "batch_size": 1, "total_pixels": 4194304, "max_new_tokens": 128, "expected": "succeeded"},
        {"case_id": "two_window_batch2_4m", "start": 0.0, "end": 5.0, "count": 2, "batch_size": 2, "total_pixels": 4194304, "max_new_tokens": 128, "expected": "succeeded"},
        {"case_id": "four_window_batch2_4m", "start": 0.0, "end": 2.5, "count": 4, "batch_size": 2, "total_pixels": 4194304, "max_new_tokens": 128, "expected": "succeeded"},
        {"case_id": "one_window_low_pixel", "start": 0.0, "end": 10.0, "count": 1, "batch_size": 1, "total_pixels": 2097152, "max_new_tokens": 128, "expected": "succeeded"},
        {"case_id": "one_window_short_generation", "start": 0.0, "end": 10.0, "count": 1, "batch_size": 1, "total_pixels": 4194304, "max_new_tokens": 64, "expected": "succeeded"},
        {"case_id": "one_window_high_pixel_14m", "start": 0.0, "end": 10.0, "count": 1, "batch_size": 1, "total_pixels": 14680064, "max_new_tokens": 128, "expected": "succeeded_or_oom"},
        {"case_id": "one_window_high_pixel_33m", "start": 0.0, "end": 10.0, "count": 1, "batch_size": 1, "total_pixels": 33554432, "max_new_tokens": 128, "expected": "succeeded_or_oom"},
        {"case_id": "decode_failure_corrupt_video", "start": 0.0, "end": 10.0, "count": 1, "batch_size": 1, "total_pixels": 4194304, "max_new_tokens": 128, "expected": "decode_failure", "corrupt": True},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/processed/g6/imax10s.mp4"))
    parser.add_argument("--corrupt-video", type=Path, default=ROOT / "tests/fixtures/corrupt.mp4")
    parser.add_argument("--model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "records/phase_g4/timelens_runtime_matrix_20260903")
    parser.add_argument("--output", type=Path, default=ROOT / "records/phase_g4/g4_timelens_runtime_matrix_recovery_20260903.json")
    parser.add_argument("--case-timeout-s", type=int, default=180)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for case in _case_definitions():
        case_started = time.perf_counter()
        case_output = args.output_dir / f"{case['case_id']}.json"
        video = args.corrupt_video if case.get("corrupt") else args.video
        command = [
            sys.executable,
            str(ROOT / "scripts/g4_timelens_smoke.py"),
            "--video", str(video),
            "--model", str(args.model),
            "--start", str(case["start"]),
            "--end", str(case["end"]),
            "--count", str(case["count"]),
            "--batch-size", str(case["batch_size"]),
            "--total-pixels", str(case["total_pixels"]),
            "--max-new-tokens", str(case["max_new_tokens"]),
            "--output", str(case_output),
        ]
        timed_out = False
        process_error = None
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'external/VideoITG'}"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=args.case_timeout_s,
                check=False,
            )
            return_code = completed.returncode
            stderr_tail = completed.stderr[-2000:]
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            return_code = None
            stderr_tail = str(exc)[-2000:]
        except Exception as exc:  # pragma: no cover - operational wrapper
            return_code = None
            process_error = f"{type(exc).__name__}: {exc}"
            stderr_tail = ""

        record = None
        if case_output.is_file():
            try:
                record = json.loads(case_output.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                process_error = f"invalid_case_json: {exc}"
        error = (record or {}).get("error")
        error_text = json.dumps(error, ensure_ascii=False) if error else (process_error or stderr_tail)
        if timed_out:
            status = "timeout"
        elif record and record.get("status") == "failed":
            status = "failed"
        elif record and record.get("status") == "succeeded":
            status = "succeeded"
        else:
            status = "runner_failed"
        if "OutOfMemory" in error_text or "out of memory" in error_text.lower():
            classification = "oom_reproduced"
        elif case.get("corrupt") and status == "failed":
            classification = "decode_failure_reproduced"
        elif timed_out:
            classification = "process_timeout"
        elif case.get("total_pixels", 0) > 4194304 and status == "succeeded":
            classification = "high_budget_oom_not_reproduced"
        else:
            classification = status
        expected_ok = (
            (case["expected"] == "succeeded" and status == "succeeded")
            or (case["expected"] == "succeeded_or_oom" and classification in {"succeeded", "high_budget_oom_not_reproduced", "oom_reproduced"})
            or (case["expected"] == "decode_failure" and classification == "decode_failure_reproduced")
        )
        rows.append({
            "case": case,
            "video": str(video),
            "status": status,
            "classification": classification,
            "expected": case["expected"],
            "case_passed": expected_ok,
            "return_code": return_code,
            "timed_out": timed_out,
            "record": record,
            "stderr_tail": stderr_tail,
            "elapsed_ms": (time.perf_counter() - case_started) * 1000.0,
        })

    timeout_probe = ROOT / "records/phase_g6/g6_realtime_service_timeout_1000ms_recovery_20260903.json"
    timeout_record = json.loads(timeout_probe.read_text(encoding="utf-8")) if timeout_probe.is_file() else None
    result = {
        "run_id": f"g4_timelens_runtime_matrix_{time.strftime('%Y%m%d_%H%M%S')}",
        "video": str(args.video),
        "model": str(args.model),
        "case_count": len(rows),
        "cases": rows,
        "async_timeout_probe": {"source": str(timeout_probe), "status": (timeout_record or {}).get("result", {}).get("canonical_status"), "degrade_reason": (timeout_record or {}).get("result", {}).get("result", {}).get("degrade_reason")},
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "scope": "real_model_window_budget_failure_matrix_not_quality_or_capacity_gate",
        "passed": bool(rows) and all(row["case_passed"] for row in rows) and (timeout_record or {}).get("result", {}).get("canonical_status") == "TIMEOUT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": result["run_id"], "case_count": result["case_count"], "classifications": [row["classification"] for row in rows], "async_timeout_status": result["async_timeout_probe"]["status"], "passed": result["passed"]}, ensure_ascii=False))
    print(f"saved: {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
