import json
import time
from pathlib import Path

import numpy as np
import pytest

from fastapi.testclient import TestClient

from videoitg_smart_clip.service.app import ServiceSettings, _safe_video_path, create_app
from videoitg_smart_clip.service.models import TaskMode, TaskStatus
from videoitg_smart_clip.service.runtime import BoundedTaskManager, QueueFullError, StubWorker


def make_app(tmp_path: Path, *, delay_s: float = 0.01, queue_size: int = 4):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"not a real video; the stub backend does not decode it")
    manager = BoundedTaskManager([StubWorker(delay_s=delay_s)], queue_size=queue_size, estimated_service_ms=50)
    settings = ServiceSettings(
        backend_name="stub",
        upload_root=tmp_path / "uploads",
        allowed_video_roots=(tmp_path,),
        max_upload_bytes=1024,
        deadline_ms=200,
    )
    return create_app(manager, settings), manager, video


def test_health_and_steady_sync(tmp_path):
    app, manager, video = make_app(tmp_path)
    client = TestClient(app)
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ready"] is True
    response = client.post("/v1/steady/rerank", json={"query": "find the segment", "video_path": str(video)})
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    assert response.json()["canonical_status"] == "SUCCESS"
    results = client.get(f"/tasks/{response.json()['task_id']}/results")
    assert results.status_code == 200
    manager.shutdown()


def test_local_frontend_origin_has_explicit_cors_access(tmp_path):
    app, manager, _ = make_app(tmp_path)
    client = TestClient(app)
    try:
        allowed = client.options(
            "/healthz",
            headers={
                "Origin": "http://127.0.0.1:8080",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:8080"
        denied = client.get("/healthz", headers={"Origin": "https://untrusted.example"})
        assert "access-control-allow-origin" not in denied.headers
    finally:
        manager.shutdown()


def test_service_settings_rejects_non_loopback_cors_origin():
    for origin in ("https://untrusted.example", "http://127.0.0.1:bad", "http://127.0.0.1:8080/?x=1"):
        with pytest.raises(ValueError, match="loopback"):
            ServiceSettings(frontend_origins=(origin,))


def test_coarse_to_fine_second_query_hits_cache_without_reencoding(tmp_path):
    from videoitg_smart_clip.grounding import StubTimeLensGrounder
    from videoitg_smart_clip.preprocessing import HashFeatureEncoder
    from videoitg_smart_clip.service.runtime import CoarseToFineWorker

    class CountingEncoder(HashFeatureEncoder):
        model_name = "counting-encoder"
        version = "counting-encoder-v1"
        device = "cpu"

        def __init__(self):
            self.encode_calls = 0
            self.load_calls = 0

        def load(self):
            self.load_calls += 1

        def encode(self, frames):
            self.encode_calls += 1
            return super().encode(frames)

        def encode_query(self, query):
            return np.ones(32, dtype=np.float32)

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    encoder = CountingEncoder()
    worker = CoarseToFineWorker(
        tmp_path / "features",
        feature_encoder=encoder,
        grounder=StubTimeLensGrounder(),
        top_n=1,
        top_k=1,
    )
    first = worker.run(video, "first query")
    second = worker.run(video, "second query")
    assert encoder.encode_calls == 1
    assert first["runtime"]["cache_event"] == "cache_miss"
    assert second["runtime"]["cache_event"] == "cache_hit"
    assert second["runtime"]["cache_hit"] == 1


def test_coarse_to_fine_cache_identity_records_sampling_budget(tmp_path):
    from videoitg_smart_clip.grounding import StubTimeLensGrounder
    from videoitg_smart_clip.preprocessing import HashFeatureEncoder
    from videoitg_smart_clip.service.runtime import CoarseToFineWorker

    class SamplingEncoder(HashFeatureEncoder):
        model_name = "sampling-encoder"
        version = "sampling-encoder-v1"
        device = "cpu"

        def __init__(self):
            self.kwargs = None

        def load(self):
            return None

        def decode_and_encode(self, video_path, *, sample_fps=1.0, max_frames=None):
            self.kwargs = {"sample_fps": sample_fps, "max_frames": max_frames}
            return np.ones((2, 32), dtype=np.float32), np.asarray([0.0, 1.0], dtype=np.float32)

        def encode_query(self, query):
            return np.ones(32, dtype=np.float32)

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    encoder = SamplingEncoder()
    worker = CoarseToFineWorker(
        tmp_path / "features",
        feature_encoder=encoder,
        grounder=StubTimeLensGrounder(),
        feature_sample_fps=0.5,
        feature_max_frames=7,
        top_n=1,
        top_k=1,
    )
    worker.run(video, "query")
    assert encoder.kwargs == {"sample_fps": 0.5, "max_frames": 7}
    metadata = next((tmp_path / "features").glob("*.json")).read_text(encoding="utf-8")
    assert '"sampling_config": "{\\"fps\\":0.5,\\"max_frames\\":7}"' in metadata


def test_videoitg_readiness_requires_all_workers_loaded(tmp_path):
    class UnloadedWorker(StubWorker):
        model_version = "VideoITG-8B"

        @property
        def loaded(self):
            return False

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    manager = BoundedTaskManager([UnloadedWorker()], queue_size=1)
    settings = ServiceSettings(
        backend_name="videoitg",
        upload_root=tmp_path,
        allowed_video_roots=(tmp_path,),
        max_upload_bytes=1024,
    )
    app = create_app(manager, settings)
    client = TestClient(app)
    assert client.get("/healthz").json()["ready"] is False
    assert client.get("/readyz").status_code == 503
    steady = client.post("/v1/steady/rerank", json={"query": "x", "video_path": str(video)})
    assert steady.status_code == 503
    assert steady.json()["detail"]["message"] == "all model workers must be loaded"
    manager.shutdown()


def test_backend_failure_is_explicit_500(tmp_path):
    class FailingWorker(StubWorker):
        model_version = "failing-test"

        def run(self, video_path, query):
            raise RuntimeError("synthetic backend failure")

    app, manager, video = make_app(tmp_path)
    manager.shutdown()
    failing_manager = BoundedTaskManager([FailingWorker()], queue_size=1)
    failing_app = create_app(
        failing_manager,
        ServiceSettings(backend_name="stub", allowed_video_roots=(tmp_path,), upload_root=tmp_path),
    )
    response = TestClient(failing_app).post(
        "/v1/steady/rerank", json={"query": "x", "video_path": str(video)}
    )
    assert response.status_code == 500
    assert "synthetic backend failure" in response.json()["detail"]["error"]
    failing_manager.shutdown()


def test_steady_timeout_returns_explicit_task_state(tmp_path):
    app, manager, video = make_app(tmp_path, delay_s=0.2)
    manager.shutdown()
    timeout_manager = BoundedTaskManager([StubWorker(delay_s=0.2)], queue_size=1)
    timeout_app = create_app(
        timeout_manager,
        ServiceSettings(
            backend_name="stub", allowed_video_roots=(tmp_path,), upload_root=tmp_path, deadline_ms=20
        ),
    )
    response = TestClient(timeout_app).post(
        "/v1/steady/rerank", json={"query": "x", "video_path": str(video)}
    )
    assert response.status_code == 504
    assert response.json()["detail"]["status"] == "timeout"
    assert response.json()["detail"]["canonical_status"] == "TIMEOUT"
    timeout_manager.shutdown()


def test_async_timeout_persists_canonical_timeout_state(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    manager = BoundedTaskManager([StubWorker(delay_s=0.05)], queue_size=1, task_timeout_s=0.005)
    try:
        record, _ = manager.submit(video, "async timeout", TaskMode.burst_async)
        manager.wait(record.task_id, 1.0)
        finished = manager.get(record.task_id)
        assert finished.status.value == "timeout"
        assert finished.error_code == "TIMEOUT"
        assert finished.current_stage == "TIMEOUT"
        assert finished.degraded is True
        assert finished.result["degrade_level"] == 2
    finally:
        manager.shutdown()


def test_async_watchdog_exposes_timeout_before_slow_worker_returns(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    manager = BoundedTaskManager([StubWorker(delay_s=0.15)], queue_size=1, task_timeout_s=0.01)
    try:
        record, _ = manager.submit(video, "watchdog", TaskMode.burst_async)
        deadline = time.perf_counter() + 0.10
        while manager.get(record.task_id).status.value != "timeout" and time.perf_counter() < deadline:
            time.sleep(0.002)
        assert manager.get(record.task_id).status.value == "timeout"
    finally:
        manager.shutdown()


def test_async_timeout_keeps_explicit_level2_coarse_fallback(tmp_path):
    class SlowResultWorker(StubWorker):
        def run(self, video_path, query):
            time.sleep(0.03)
            return {"predictions": [], "candidates": [{"candidate_id": "c", "coarse_start": 2.0, "coarse_end": 7.0, "retrieval_score": 0.8}]}

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    manager = BoundedTaskManager([SlowResultWorker()], queue_size=1, task_timeout_s=0.005)
    try:
        record, _ = manager.submit(video, "async timeout fallback", TaskMode.burst_async)
        manager.wait(record.task_id, 1.0)
        finished = manager.get(record.task_id)
        assert finished.status.value == "timeout"
        assert finished.degraded is True
        assert finished.result["degrade_level"] == 2
        assert finished.result["predictions"][0]["raw_start"] == 2.0
        results = TestClient(create_app(manager, ServiceSettings(backend_name="stub", allowed_video_roots=(tmp_path,), upload_root=tmp_path))).get(
            f"/tasks/{record.task_id}/results"
        )
        assert results.status_code == 200
        assert results.json()["status"] == "timeout"
        assert results.json()["canonical_status"] == "TIMEOUT"
        assert results.json()["result"]["degrade_level"] == 2
    finally:
        manager.shutdown()


def test_oom_like_failure_clears_and_worker_remains_usable(tmp_path):
    class OOMOnceWorker(StubWorker):
        def __init__(self):
            super().__init__()
            self.failed_once = False

        def run(self, video_path, query):
            if not self.failed_once:
                self.failed_once = True
                raise RuntimeError("CUDA out of memory (synthetic test)")
            return super().run(video_path, query)

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    worker = OOMOnceWorker()
    manager = BoundedTaskManager([worker], queue_size=2)
    app = create_app(manager, ServiceSettings(backend_name="stub", allowed_video_roots=(tmp_path,), upload_root=tmp_path))
    client = TestClient(app)
    first = client.post("/v1/steady/rerank", json={"query": "oom", "video_path": str(video)})
    second = client.post("/v1/steady/rerank", json={"query": "after oom", "video_path": str(video)})
    assert first.status_code == 500
    assert second.status_code == 200
    manager.shutdown()


def test_async_lifecycle_and_idempotency(tmp_path):
    app, manager, video = make_app(tmp_path)
    client = TestClient(app)
    payload = {"query": "find the segment", "video_path": str(video), "request_id": "req-1"}
    first = client.post("/v1/tasks", json=payload)
    second = client.post("/v1/tasks", json=payload)
    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["task_id"] == first.json()["task_id"]
    assert second.json()["deduplicated"] is True
    task_id = first.json()["task_id"]
    for _ in range(20):
        result = client.get(f"/v1/tasks/{task_id}")
        if result.json()["status"] == "succeeded":
            break
        time.sleep(0.01)
    assert result.json()["status"] == "succeeded"
    manager.shutdown()


def test_cancel_queued_task_and_reject_path_escape(tmp_path):
    app, manager, video = make_app(tmp_path, delay_s=0.3, queue_size=2)
    client = TestClient(app)
    first = client.post("/v1/tasks", json={"query": "a", "video_path": str(video)})
    second = client.post("/v1/tasks", json={"query": "b", "video_path": str(video)})
    cancelled = client.delete(f"/v1/tasks/{second.json()['task_id']}")
    assert first.status_code == 202 and second.status_code == 202
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] in {"cancelled", "running"}
    escaped = client.post("/v1/tasks", json={"query": "a", "video_path": "/etc/passwd.mp4"})
    assert escaped.status_code == 400
    manager.shutdown()


def test_cancel_running_task_remains_cancelled_after_backend_returns(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    manager = BoundedTaskManager([StubWorker(delay_s=0.05)], queue_size=1)
    app = create_app(manager, ServiceSettings(backend_name="stub", allowed_video_roots=(tmp_path,), upload_root=tmp_path))
    client = TestClient(app)
    try:
        created = client.post("/v1/tasks", json={"query": "cancel running", "video_path": str(video)})
        task_id = created.json()["task_id"]
        deadline = time.perf_counter() + 1.0
        while manager.get(task_id).status != TaskStatus.running and time.perf_counter() < deadline:
            time.sleep(0.002)
        cancelled = client.delete(f"/v1/tasks/{task_id}")
        assert cancelled.status_code == 200
        manager.wait(task_id, 1.0)
        finished = manager.get(task_id)
        assert finished.status == TaskStatus.cancelled
        assert finished.error_code == "CANCELLED"
        assert finished.result is None
    finally:
        manager.shutdown()


def test_upload_validation_and_task_creation(tmp_path):
    app, manager, _ = make_app(tmp_path)
    client = TestClient(app)
    good = client.post(
        "/v1/tasks/upload",
        data={"query": "uploaded segment"},
        files={"file": ("clip.mp4", b"stub bytes", "video/mp4")},
    )
    bad = client.post(
        "/v1/tasks/upload",
        data={"query": "uploaded segment"},
        files={"file": ("clip.txt", b"stub bytes", "text/plain")},
    )
    oversized = client.post(
        "/v1/tasks/upload",
        data={"query": "uploaded segment"},
        files={"file": ("large.mp4", b"x" * 2048, "video/mp4")},
    )
    assert good.status_code == 202
    assert bad.status_code == 400
    assert oversized.status_code == 413
    manager.shutdown()


def test_bounded_manager_rejects_when_queue_is_full(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    manager = BoundedTaskManager([StubWorker(delay_s=0.5)], queue_size=1)
    try:
        first, _ = manager.submit(video, "first", TaskMode.burst_async)
        for _ in range(50):
            if manager.get(first.task_id).status.value == "running":
                break
            time.sleep(0.01)
        manager.submit(video, "second", TaskMode.burst_async)
        with pytest.raises(QueueFullError):
            manager.submit(video, "third", TaskMode.burst_async)
    finally:
        manager.shutdown()


def test_terminal_task_state_recovers_after_manager_restart(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    state_path = tmp_path / "service_tasks.jsonl"
    first_manager = BoundedTaskManager([StubWorker()], queue_size=1, state_path=state_path)
    first, _ = first_manager.submit(video, "recover me", TaskMode.burst_async, request_id="recover-1")
    first_manager.wait(first.task_id, 1.0)
    first_manager.shutdown()

    second_manager = BoundedTaskManager([StubWorker()], queue_size=1, state_path=state_path)
    try:
        recovered = second_manager.get(first.task_id)
        assert recovered.status.value == "succeeded"
        same, deduplicated = second_manager.submit(video, "recover me", TaskMode.burst_async, request_id="recover-1")
        assert deduplicated is True
        assert same.task_id == first.task_id
    finally:
        second_manager.shutdown()


def test_interrupted_task_recovery_exposes_terminal_error(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    state_path = tmp_path / "service_tasks.jsonl"
    state_path.write_text(json.dumps({
        "task_id": "interrupted-1",
        "request_id": "recover-interrupted",
        "mode": "burst_async",
        "video_path": str(video),
        "query": "interrupted query",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:01+00:00",
        "status": "running",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": None,
        "result": None,
        "error": None,
        "cancel_requested": False,
        "sequence": 1,
        "priority": 10,
        "video_id": "sample",
        "model_version": "TimeLens-8B",
        "progress": 0.25,
        "current_stage": "INDEXING",
        "error_code": None,
        "degraded": False,
    }) + "\n", encoding="utf-8")
    manager = BoundedTaskManager([StubWorker()], queue_size=1, state_path=state_path)
    try:
        recovered = manager.get("interrupted-1")
        assert recovered.status is TaskStatus.failed
        assert recovered.current_stage == "FAILED"
        assert recovered.error_code == "SERVICE_RESTART_RECOVERY"
        assert recovered.model_version == "TimeLens-8B"
    finally:
        manager.shutdown()


def test_feedback_is_versioned_and_associates_query_without_video_path(tmp_path):
    app, manager, video = make_app(tmp_path)
    feedback_path = tmp_path / "runtime" / "feedback.jsonl"
    app = create_app(manager, ServiceSettings(
        backend_name="stub",
        upload_root=tmp_path / "uploads",
        allowed_video_roots=(tmp_path,),
        max_upload_bytes=1024,
        deadline_ms=200,
        feedback_path=feedback_path,
    ))
    client = TestClient(app)
    response = client.post("/v1/steady/rerank", json={"query": "private query", "video_path": str(video)})
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    feedback = client.post(
        "/v1/feedback",
        json={"task_id": task_id, "candidate_id": "stub", "label": "start_too_late", "adjusted_start_s": 2.0, "adjusted_end_s": 6.0},
    )
    assert feedback.status_code == 201
    assert feedback.json()["label"] == "START_TOO_LATE"
    assert feedback.json()["feedback_version"] == "feedback-v1"
    saved = feedback_path.read_text(encoding="utf-8")
    assert '"query": "private query"' in saved
    assert str(video) not in saved
    assert '"label": "START_TOO_LATE"' in saved
    assert '"model_start": 0.0' in saved
    assert '"model_end": 5.0' in saved
    assert '"user_start": 2.0' in saved
    assert '"user_end": 6.0' in saved
    manager.shutdown()


def test_feedback_rejects_partial_manual_boundary_adjustment(tmp_path):
    app, manager, video = make_app(tmp_path)
    client = TestClient(app)
    try:
        response = client.post("/v1/steady/rerank", json={"query": "segment", "video_path": str(video)})
        assert response.status_code == 200
        task_id = response.json()["task_id"]
        partial_start = client.post(
            "/v1/feedback",
            json={"task_id": task_id, "candidate_id": "stub", "label": "START_TOO_LATE", "adjusted_start_s": 1.0},
        )
        partial_end = client.post(
            "/v1/feedback",
            json={"task_id": task_id, "candidate_id": "stub", "label": "END_TOO_EARLY", "adjusted_end_s": 4.0},
        )
        assert partial_start.status_code == 422
        assert partial_end.status_code == 422
    finally:
        manager.shutdown()


def test_feedback_rejects_unknown_or_unfinished_task(tmp_path):
    app, manager, video = make_app(tmp_path, delay_s=0.2)
    client = TestClient(app)
    missing = client.post("/v1/feedback", json={"task_id": "missing", "label": "accepted"})
    assert missing.status_code == 404
    queued = client.post("/v1/tasks", json={"query": "x", "video_path": str(video)})
    rejected = client.post("/v1/feedback", json={"task_id": queued.json()["task_id"], "label": "accepted"})
    assert rejected.status_code == 409
    manager.shutdown()


def test_feedback_accepts_completed_timeout_fallback(tmp_path):
    class SlowResultWorker(StubWorker):
        def run(self, video_path, query):
            time.sleep(0.02)
            return {
                "predictions": [],
                "candidates": [{
                    "candidate_id": "coarse-1",
                    "coarse_start": 2.0,
                    "coarse_end": 7.0,
                    "retrieval_score": 0.8,
                }],
            }

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    feedback_path = tmp_path / "runtime" / "feedback.jsonl"
    manager = BoundedTaskManager([SlowResultWorker()], queue_size=1, task_timeout_s=0.005)
    app = create_app(manager, ServiceSettings(
        backend_name="stub",
        upload_root=tmp_path,
        allowed_video_roots=(tmp_path,),
        feedback_path=feedback_path,
    ))
    client = TestClient(app)
    try:
        created = client.post("/v1/tasks", json={"query": "late fallback", "video_path": str(video)})
        assert created.status_code == 202
        task_id = created.json()["task_id"]
        manager.wait(task_id, 1.0)
        record = manager.get(task_id)
        assert record.status == TaskStatus.timeout
        assert record.result is not None
        saved = client.post("/v1/feedback", json={
            "task_id": task_id,
            "candidate_id": "coarse-1",
            "label": "MISS",
        })
        assert saved.status_code == 201
        assert '"model_start": 2.0' in feedback_path.read_text(encoding="utf-8")
    finally:
        manager.shutdown()


def test_feedback_model_bounds_prefer_raw_grounding(tmp_path):
    class RefinedWorker(StubWorker):
        def run(self, video_path, query):
            return {
                "predictions": [{
                    "candidate_id": "c1",
                    "raw_start": 3.0,
                    "raw_end": 4.0,
                    "refined_start": 2.5,
                    "refined_end": 4.5,
                    "final_score": 0.9,
                }],
                "candidates": [],
            }

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    feedback_path = tmp_path / "runtime" / "feedback.jsonl"
    manager = BoundedTaskManager([RefinedWorker()], queue_size=1)
    app = create_app(manager, ServiceSettings(
        backend_name="stub",
        upload_root=tmp_path,
        allowed_video_roots=(tmp_path,),
        feedback_path=feedback_path,
    ))
    client = TestClient(app)
    try:
        created = client.post("/v1/steady/rerank", json={"query": "raw bounds", "video_path": str(video)})
        assert created.status_code == 200
        saved = client.post("/v1/feedback", json={"task_id": created.json()["task_id"], "candidate_id": "c1", "label": "ACCEPT"})
        assert saved.status_code == 201
        text = feedback_path.read_text(encoding="utf-8")
        assert '"model_start": 3.0' in text
        assert '"model_end": 4.0' in text
    finally:
        manager.shutdown()


def test_feedback_adjustment_requires_candidate_id(tmp_path):
    app, manager, video = make_app(tmp_path)
    client = TestClient(app)
    response = client.post("/v1/steady/rerank", json={"query": "x", "video_path": str(video)})
    assert response.status_code == 200
    rejected = client.post("/v1/feedback", json={"task_id": response.json()["task_id"], "label": "START_TOO_LATE", "adjusted_start_s": 1.0, "adjusted_end_s": 2.0})
    assert rejected.status_code == 400
    manager.shutdown()


def test_coarse_to_fine_lifecycle_persists_stage_progress(tmp_path):
    from videoitg_smart_clip.grounding import StubTimeLensGrounder
    from videoitg_smart_clip.pipeline.postprocess import BoundaryRefiner
    from videoitg_smart_clip.service.runtime import CoarseToFineWorker

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    state_path = tmp_path / "runtime" / "tasks.jsonl"
    worker = CoarseToFineWorker(
        tmp_path / "features",
        grounder=StubTimeLensGrounder(),
        boundary_refiner=BoundaryRefiner(expansion_seconds=0.5),
        top_n=2,
        top_k=1,
    )
    manager = BoundedTaskManager([worker], queue_size=2, state_path=state_path)
    app = create_app(manager, ServiceSettings(backend_name="coarse_to_fine", allowed_video_roots=(tmp_path,), upload_root=tmp_path))
    client = TestClient(app)
    try:
        created = client.post("/tasks", json={"query": "event", "video_path": str(video)})
        assert created.status_code == 202
        task_id = created.json()["task_id"]
        for _ in range(100):
            response = client.get(f"/tasks/{task_id}")
            if response.json()["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert response.json()["canonical_status"] == "SUCCESS"
        assert response.json()["result"]["runtime"]["cache_event"] in {"cache_hit", "cache_miss"}
        candidate = response.json()["result"]["predictions"][0]
        assert candidate["refined_end"] >= candidate["raw_end"]
        stages = {json.loads(line)["current_stage"] for line in state_path.read_text().splitlines()}
        assert {"PREPROCESSING", "INDEXING", "RETRIEVING", "GROUNDING", "POSTPROCESSING", "SUCCESS"} <= stages
        payloads = [json.loads(line) for line in state_path.read_text().splitlines()]
        assert all(payload.get("model_version") == "TimeLens-8B-stub-v1" for payload in payloads)
        assert all(payload.get("updated_at") for payload in payloads)
    finally:
        manager.shutdown()


def test_video_duration_limit_is_enforced_when_probe_succeeds():
    video = Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/processed/g6/imax10s.mp4")
    if not video.is_file():
        pytest.skip("real smoke video is not available")
    from fastapi import HTTPException

    settings = ServiceSettings(allowed_video_roots=(video.parent,), max_video_duration_seconds=5.0)
    with pytest.raises(HTTPException) as error:
        _safe_video_path(str(video), settings)
    assert error.value.status_code == 413
