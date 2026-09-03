"""Phase 8 user test protocol executor.

Simulates the fixed procedure for each of the 10 query tasks (repeated to 20+):
  upload -> poll -> inspect candidate -> submit feedback -> record.
This uses the stub backend and is explicitly NOT a VideoITG quality evaluation.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

API_BASE = "http://127.0.0.1:8000"
VIDEO_PATH = Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/raw/imax.mp4")
N_OPERATIONS = 22  # >= 20 as required

QUERIES = [
    "找到视频开头的主要场景",
    "找到人物出现在画面中的片段",
    "找到画面中主体最清晰的片段",
    "找到人物在场景中移动的片段",
    "找到画面变化最明显的片段",
    "找到包含主要物体的片段",
    "找到视频中部的场景",
    "找到视频结尾附近的场景",
    "输入一个视频中不存在的目标",
    "找到画面中主体最清晰的片段",  # repeat of #3 — test task independence
]

FEEDBACK_LABELS = [
    "accepted",
    "start_too_late",
    "accepted",
    "end_too_early",
    "irrelevant",
    "accepted",
    "start_too_early",
    "end_too_late",
    "no_target_in_video",
    "duplicate",
    "accepted",
    "start_too_late",
    "accepted",
    "end_too_early",
    "irrelevant",
    "accepted",
    "start_too_early",
    "end_too_late",
    "no_target_in_video",
    "duplicate",
    "accepted",
    "irrelevant",
]


def run() -> dict:
    sessions = []
    failures = 0
    accepted_count = 0
    label_counts: dict[str, int] = {}
    first_candidate_usable = 0
    boundary_adjustments: list[dict] = []
    completion_times: list[float] = []

    for i in range(N_OPERATIONS):
        query = QUERIES[i % len(QUERIES)]
        label = FEEDBACK_LABELS[i]
        request_id = f"user-test-{i}-{uuid.uuid4().hex[:8]}"

        print(f"\n[{i+1}/{N_OPERATIONS}] query={query!r} label={label} rid={request_id}")

        # Step 1: Upload video + submit task
        try:
            with open(VIDEO_PATH, "rb") as f:
                resp = requests.post(
                    f"{API_BASE}/v1/tasks/upload",
                    data={"query": query, "request_id": request_id},
                    files={"file": ("imax.mp4", f, "video/mp4")},
                    timeout=10,
                )
            resp.raise_for_status()
            task = resp.json()
            task_id = task["task_id"]
            print(f"  task_id={task_id} status={task['status']}")
        except Exception as e:
            print(f"  FAIL submit: {e}")
            failures += 1
            sessions.append({
                "iteration": i + 1, "query": query, "task_id": None,
                "status": "submit_failed", "error": str(e), "feedback": None,
                "completion_s": None, "first_candidate_usable": False,
                "boundary_adjusted": False,
            })
            continue

        # Step 2: Poll until terminal state
        t0 = time.time()
        terminal = None
        for attempt in range(200):
            try:
                poll = requests.get(f"{API_BASE}/v1/tasks/{task_id}", timeout=5).json()
            except Exception:
                poll = {"status": "unknown"}
            if poll["status"] not in {"queued", "running"}:
                terminal = poll
                break
            time.sleep(0.05)
        elapsed = time.time() - t0

        if terminal is None:
            print(f"  FAIL polling timeout")
            failures += 1
            sessions.append({
                "iteration": i + 1, "query": query, "task_id": task_id,
                "status": "poll_timeout", "error": "polling exceeded max attempts",
                "feedback": None, "completion_s": elapsed,
                "first_candidate_usable": False, "boundary_adjusted": False,
            })
            continue

        completion_times.append(elapsed)
        status = terminal["status"]
        predictions = (terminal.get("result") or {}).get("predictions", [])
        first_candidate = predictions[0] if predictions else None

        print(f"  done: status={status} elapsed={elapsed:.2f}s predictions={len(predictions)}")

        if status != "succeeded":
            print(f"  FAIL status={status}")
            failures += 1
            sessions.append({
                "iteration": i + 1, "query": query, "task_id": task_id,
                "status": status, "error": terminal.get("error"),
                "feedback": None, "completion_s": elapsed,
                "first_candidate_usable": False, "boundary_adjusted": False,
            })
            continue

        # Step 3: Inspect first candidate usability
        usable = first_candidate is not None and first_candidate.get("score", 0) > 0
        if usable:
            first_candidate_usable += 1

        # Determine boundary adjustment
        adjusted_start = None
        adjusted_end = None
        boundary_adjusted = False
        if first_candidate and label in ("start_too_early", "start_too_late", "end_too_early", "end_too_late"):
            cs = first_candidate["start_s"]
            ce = first_candidate["end_s"]
            if label == "start_too_early":
                adjusted_start = round(cs + 1.5, 2)
                boundary_adjusted = True
            elif label == "start_too_late":
                adjusted_start = round(max(0, cs - 1.5), 2)
                boundary_adjusted = True
            elif label == "end_too_early":
                adjusted_end = round(ce + 2.0, 2)
                boundary_adjusted = True
            elif label == "end_too_late":
                adjusted_end = round(max(cs + 1, ce - 2.0), 2)
                boundary_adjusted = True
            if boundary_adjusted:
                boundary_adjustments.append({
                    "iteration": i + 1, "task_id": task_id,
                    "original_start": cs, "original_end": ce,
                    "adjusted_start": adjusted_start, "adjusted_end": adjusted_end,
                })

        # Step 4: Submit feedback
        try:
            feedback_payload = {"task_id": task_id, "label": label}
            if adjusted_start is not None:
                feedback_payload["adjusted_start_s"] = adjusted_start
            if adjusted_end is not None:
                feedback_payload["adjusted_end_s"] = adjusted_end
            feedback_resp = requests.post(
                f"{API_BASE}/v1/feedback",
                json=feedback_payload,
                timeout=5,
            )
            feedback_resp.raise_for_status()
            print(f"  feedback: {label}")
            if label == "accepted":
                accepted_count += 1
            label_counts[label] = label_counts.get(label, 0) + 1
        except Exception as e:
            print(f"  FAIL feedback: {e}")

        sessions.append({
            "iteration": i + 1,
            "query": query,
            "task_id": task_id,
            "status": status,
            "error": None,
            "feedback": label,
            "completion_s": round(elapsed, 3),
            "first_candidate_usable": usable,
            "boundary_adjusted": boundary_adjusted,
            "first_candidate": {
                "start_s": first_candidate["start_s"],
                "end_s": first_candidate["end_s"],
                "score": first_candidate["score"],
            } if first_candidate else None,
        })

    # Summary
    total = len(sessions)
    successful = sum(1 for s in sessions if s["status"] == "succeeded")
    return {
        "run_id": datetime.now(timezone.utc).strftime("user_test_%Y%m%dT%H%M%SZ"),
        "scope": "local_user_test_protocol",
        "backend": "stub",
        "video": str(VIDEO_PATH),
        "total_operations": total,
        "successful_tasks": successful,
        "failed_tasks": failures,
        "accepted_count": accepted_count,
        "first_candidate_usable_count": first_candidate_usable,
        "label_counts": label_counts,
        "boundary_adjustments": boundary_adjustments,
        "completion_times": {
            "count": len(completion_times),
            "mean_s": round(sum(completion_times) / len(completion_times), 3) if completion_times else None,
            "p50_s": round(sorted(completion_times)[len(completion_times) // 2], 3) if completion_times else None,
            "p95_s": round(sorted(completion_times)[int(len(completion_times) * 0.95)], 3) if completion_times and len(completion_times) >= 20 else None,
        },
        "first_candidate_adoption_rate": round(first_candidate_usable / successful, 3) if successful else None,
        "sessions": sessions,
        "human_user_test": False,
        "videoitg_quality_evaluation": False,
        "note": "Simulated protocol run with stub backend. The fixed steps are executed via REST API."
    }


def main():
    output_dir = Path("/home/zjy/projects/videoitg_smart_clip/records/phase8")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    output_path = output_dir / "user_test_execution.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Results written to {output_path}")
    print(f"Total: {result['total_operations']} operations, "
          f"{result['successful_tasks']} successful, "
          f"{result['failed_tasks']} failed")
    print(f"Accepted: {result['accepted_count']}, "
          f"First candidate usable: {result['first_candidate_usable_count']}, "
          f"Adoption rate: {result['first_candidate_adoption_rate']}")
    print(f"Completion P50: {result['completion_times'].get('p50_s')}s")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
