"""Run a deterministic local workflow check for upload -> task -> feedback.

This is an API/protocol validation using the stub backend. It is deliberately
not reported as a human usability study or a VideoITG quality evaluation.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from videoitg_smart_clip.service.app import ServiceSettings, create_app
from videoitg_smart_clip.service.runtime import BoundedTaskManager, StubWorker


def run(iterations: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="videoitg_workflow_") as tmp:
        root = Path(tmp)
        manager = BoundedTaskManager([StubWorker(delay_s=0.002)], queue_size=8, estimated_service_ms=10)
        feedback_path = root / "runtime" / "feedback.jsonl"
        app = create_app(
            manager,
            ServiceSettings(
                backend_name="stub",
                upload_root=root / "uploads",
                allowed_video_roots=(root,),
                max_upload_bytes=1024 * 1024,
                deadline_ms=500,
                feedback_path=feedback_path,
            ),
        )
        client = TestClient(app)
        sessions = []
        try:
            for index in range(iterations):
                response = client.post(
                    "/v1/tasks/upload",
                    data={"query": f"workflow query {index}", "request_id": f"workflow-{index}"},
                    files={"file": (f"clip-{index}.mp4", b"stub-video", "video/mp4")},
                )
                response.raise_for_status()
                task = response.json()
                for _ in range(100):
                    current = client.get(f"/v1/tasks/{task['task_id']}").json()
                    if current["status"] not in {"queued", "running"}:
                        break
                    time.sleep(0.005)
                if current["status"] != "succeeded":
                    raise RuntimeError(f"task did not succeed: {current}")
                feedback = client.post(
                    "/v1/feedback",
                    json={"task_id": task["task_id"], "label": "accepted" if index % 2 == 0 else "start_too_late"},
                )
                feedback.raise_for_status()
                sessions.append({"task_id": task["task_id"], "status": current["status"], "feedback": feedback.json()["label"]})
        finally:
            manager.shutdown()
        saved = feedback_path.read_text(encoding="utf-8").splitlines()
        return {
            "run_id": datetime.now(timezone.utc).strftime("workflow_%Y%m%dT%H%M%SZ"),
            "scope": "automated_protocol_only",
            "backend": "stub",
            "iterations": iterations,
            "successful_tasks": len(sessions),
            "feedback_events": len(saved),
            "sessions": sessions,
            "human_user_test": False,
            "videoitg_quality_evaluation": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    result = run(args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
