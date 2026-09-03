#!/usr/bin/env python3
"""Exercise the real local worker through the asynchronous HTTP task API."""

from __future__ import annotations

import json
import argparse
import sys
import time
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoitg_smart_clip.grounding import TimeLensGrounder
from videoitg_smart_clip.preprocessing import SigLIPFeatureEncoder
from videoitg_smart_clip.service.app import ServiceSettings, create_app
from videoitg_smart_clip.service.runtime import BoundedTaskManager, CoarseToFineWorker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/processed/g6/imax10s.mp4"))
    parser.add_argument("--feature-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"))
    parser.add_argument("--timelens-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B"))
    parser.add_argument("--feature-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/features/g6_service"))
    parser.add_argument("--async-timeout-ms", type=int, default=120000)
    parser.add_argument("--timelens-total-pixels", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.async_timeout_ms <= 0:
        raise SystemExit("--async-timeout-ms must be positive")
    video = args.video
    feature_model = args.feature_model
    timelens_model = args.timelens_model
    state_path = Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/runtime/g6_real_service_tasks.jsonl")
    worker = CoarseToFineWorker(args.feature_root, feature_encoder=SigLIPFeatureEncoder(feature_model, device="cuda:0"), grounder=TimeLensGrounder(timelens_model, device="cuda:0", total_pixels=args.timelens_total_pixels), top_n=1, top_k=1)
    manager = BoundedTaskManager([worker], queue_size=2, state_path=state_path, task_timeout_s=args.async_timeout_ms / 1000.0)
    app = create_app(manager, ServiceSettings(backend_name="coarse_to_fine", allowed_video_roots=(video.parent,), upload_root=video.parent, deadline_ms=120000, async_timeout_ms=args.async_timeout_ms, service_version="g6-real-service-smoke"))
    observations = []
    started = time.perf_counter()
    try:
        with TestClient(app) as client:
            created = client.post("/tasks", json={"query": "a person performing an action", "video_path": str(video), "request_id": f"g6-real-service-{uuid.uuid4().hex}"})
            created.raise_for_status()
            task_id = created.json()["task_id"]
            while time.perf_counter() - started < 180:
                payload = client.get(f"/tasks/{task_id}").json()
                observations.append({"status": payload["status"], "canonical_status": payload.get("canonical_status"), "current_stage": payload.get("current_stage"), "progress": payload.get("progress"), "degraded": payload.get("degraded")})
                if payload["status"] in {"succeeded", "failed", "cancelled", "timeout"}:
                    break
                time.sleep(0.5)
            result = None
            if observations[-1]["status"] in {"succeeded", "timeout"}:
                # A timeout may be exposed by the watchdog before a still-
                # running backend has materialized its late coarse fallback.
                for _ in range(360):
                    candidate_result = client.get(f"/tasks/{task_id}/results")
                    if candidate_result.status_code == 200:
                        result = candidate_result
                        break
                    time.sleep(0.5)
        record = {"run_id": f"g6_real_service_{time.strftime('%Y%m%d_%H%M%S')}", "task_id": task_id, "async_timeout_ms": args.async_timeout_ms, "timelens_total_pixels": args.timelens_total_pixels, "observations": observations, "elapsed_ms": (time.perf_counter() - started) * 1000.0, "result": result.json() if result is not None else None, "scope": "single_real_async_http_smoke_or_timeout_contract_not_throughput_or_quality_gate"}
    finally:
        manager.shutdown()
    if args.async_timeout_ms == 120000 and args.timelens_total_pixels == 4 * 1024 * 1024:
        output_name = "g6_realtime_service_smoke.json"
    elif args.timelens_total_pixels == 4 * 1024 * 1024:
        output_name = f"g6_realtime_service_timeout_{args.async_timeout_ms}ms.json"
    else:
        output_name = f"g6_realtime_service_{args.async_timeout_ms}ms_{args.timelens_total_pixels}pixels.json"
    output = args.output or ROOT / "records/phase_g6" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
