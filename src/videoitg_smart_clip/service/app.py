"""FastAPI application for steady synchronous and burst asynchronous modes."""

from __future__ import annotations

import uuid
import json
import threading
import hashlib
import shutil
import subprocess
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from .models import FeedbackLabel, FeedbackRequest, FeedbackResponse, HealthResponse, RerankRequest, TaskMode, TaskResponse, TaskStatus
from .runtime import BoundedTaskManager, QueueFullError, TaskNotFoundError, TaskRecord, utcnow


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _feedback_candidate_score(result: dict | None, candidate_id: str | None) -> float | None:
    """Extract score without mutating or replacing the original prediction."""
    if not result or not candidate_id:
        return None
    for row in [*result.get("predictions", []), *result.get("candidates", [])]:
        if row.get("candidate_id") == candidate_id:
            value = row.get("final_score", row.get("score"))
            return float(value) if value is not None else None
    return None


def _canonical_feedback_label(value: str) -> str:
    # Accept legacy lowercase aliases at the API boundary, but persist only
    # the canonical eight-label vocabulary required by the new contract.
    aliases = {
        "accepted": "ACCEPT",
        "irrelevant": "IRRELEVANT",
        "start_too_early": "START_TOO_EARLY",
        "start_too_late": "START_TOO_LATE",
        "end_too_early": "END_TOO_EARLY",
        "end_too_late": "END_TOO_LATE",
        "no_target_in_video": "MISS",
        "duplicate": "DUPLICATE",
    }
    return aliases.get(value, value)


def _feedback_candidate_bounds(result: dict | None, candidate_id: str | None) -> tuple[float | None, float | None]:
    if not result or not candidate_id:
        return None, None
    for row in [*result.get("predictions", []), *result.get("candidates", [])]:
        if row.get("candidate_id") == candidate_id:
            # Feedback must retain the model's original grounding boundary;
            # refined/user-facing bounds are separate postprocess artifacts.
            start = row.get("raw_start", row.get("refined_start", row.get("start_s")))
            end = row.get("raw_end", row.get("refined_end", row.get("end_s")))
            return (float(start) if start is not None else None, float(end) if end is not None else None)
    return None, None


@dataclass(frozen=True)
class ServiceSettings:
    backend_name: str = "stub"
    upload_root: Path = Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/raw/service_uploads")
    allowed_video_roots: tuple[Path, ...] = (Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip"),)
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    max_video_duration_seconds: float = 1800.0
    deadline_ms: int = 5000
    async_timeout_ms: int = 120000
    policy_version: str = "E0_fixed16"
    service_version: str = "phase6-restricted-v1"
    feedback_path: Path = Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/runtime/service_feedback.jsonl")
    # The documented local frontend runs on port 8080 while the API defaults
    # to 8000.  Keep cross-origin access explicit and loopback-only; never use
    # a wildcard origin for this service.
    frontend_origins: tuple[str, ...] = ("http://127.0.0.1:8080", "http://localhost:8080")

    def __post_init__(self) -> None:
        invalid = []
        for origin in self.frontend_origins:
            try:
                parsed = urlparse(str(origin))
                port = parsed.port
            except ValueError:
                invalid.append(str(origin))
                continue
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username
                or parsed.password
                or parsed.path not in {"", "/"}
                or parsed.params
                or parsed.query
                or parsed.fragment
                or (port is not None and not 1 <= port <= 65535)
            ):
                invalid.append(str(origin))
        if invalid:
            raise ValueError("frontend_origins must contain loopback HTTP origins only: " + ",".join(invalid))


def _safe_video_path(value: str, settings: ServiceSettings) -> Path:
    raw = Path(value).expanduser()
    if raw.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="unsupported video extension")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="video file does not exist") from exc
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail="video path is not a regular file")
    roots = tuple(root.expanduser().resolve() for root in settings.allowed_video_roots)
    if roots and not any(resolved == root or root in resolved.parents for root in roots):
        raise HTTPException(status_code=400, detail="video path is outside allowed roots")
    if resolved.stat().st_size > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="video exceeds configured size limit")
    duration = _probe_video_duration(resolved)
    if duration is not None and duration > settings.max_video_duration_seconds:
        raise HTTPException(status_code=413, detail="video exceeds configured duration limit")
    return resolved


def _probe_video_duration(path: Path) -> float | None:
    """Probe duration without opening a CUDA/decord reader in the API thread."""

    executable = shutil.which("ffprobe")
    if executable is None:
        try:
            import static_ffmpeg

            static_ffmpeg.add_paths()
            executable = shutil.which("ffprobe")
        except Exception:
            executable = None
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _canonical_task_status(record: TaskRecord) -> str:
    return {
        TaskStatus.queued: "PENDING",
        TaskStatus.running: record.current_stage if record.current_stage in {"PREPROCESSING", "INDEXING", "RETRIEVING", "GROUNDING", "POSTPROCESSING"} else "GROUNDING",
        TaskStatus.succeeded: "SUCCESS",
        TaskStatus.failed: "FAILED",
        TaskStatus.cancelled: "CANCELLED",
        TaskStatus.timeout: "TIMEOUT",
        TaskStatus.overloaded: "FAILED",
    }.get(record.status, "PENDING")


def _response(record: TaskRecord, manager: BoundedTaskManager, settings: ServiceSettings, *, deduplicated: bool = False) -> TaskResponse:
    position = manager.queue_position(record.task_id)
    workers = max(1, len(manager.workers))
    estimate = int(position * manager.estimated_service_ms / workers) if position is not None else None
    canonical = _canonical_task_status(record)
    return TaskResponse(
        task_id=record.task_id,
        request_id=record.request_id,
        mode=record.mode,
        status=record.status,
        queue_position=position,
        estimated_wait_ms=estimate,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        model_version=record.model_version or manager.model_version,
        policy_version=settings.policy_version,
        result=record.result,
        error=record.error,
        cancel_requested=record.cancel_requested,
        deduplicated=deduplicated,
        video_id=record.video_id,
        progress=record.progress,
        current_stage=record.current_stage,
        error_code=record.error_code,
        degraded=record.degraded,
        canonical_status=canonical,
    )


def create_app(manager: BoundedTaskManager, settings: ServiceSettings | None = None) -> FastAPI:
    settings = settings or ServiceSettings()

    def is_ready() -> bool:
        if not manager.workers:
            return False
        if settings.backend_name == "videoitg":
            return manager.loaded_workers == len(manager.workers)
        return True

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title="Query-aware Smart Clip Restricted Service", version=settings.service_version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.frontend_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.state.manager = manager
    app.state.settings = settings
    feedback_lock = threading.Lock()

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            ready=is_ready(),
            backend=settings.backend_name,
            workers=len(manager.workers),
            queue_depth=manager.queue_depth,
            queue_capacity=manager.queue_size,
            policy_version=settings.policy_version,
            model_version=manager.model_version,
            loaded_workers=manager.loaded_workers,
        )

    @app.get("/readyz", response_model=HealthResponse)
    def readyz() -> HealthResponse:
        response = healthz()
        if not response.ready:
            raise HTTPException(status_code=503, detail=response.model_dump())
        return response

    @app.post("/v1/steady/rerank", response_model=TaskResponse)
    def steady_rerank(payload: RerankRequest) -> TaskResponse:
        if not is_ready():
            raise HTTPException(status_code=503, detail={"reason": "service_not_ready", "message": "all model workers must be loaded"})
        video_path = _safe_video_path(payload.video_path, settings)
        try:
            record, deduplicated = manager.submit(video_path, payload.query, TaskMode.steady, payload.request_id)
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail={"reason": "overloaded", "message": str(exc)}) from exc
        record = manager.wait(record.task_id, settings.deadline_ms / 1000.0)
        response = _response(record, manager, settings, deduplicated=deduplicated)
        if response.status in {TaskStatus.queued, TaskStatus.running}:
            record = manager.mark_timeout(record.task_id)
            response = _response(record, manager, settings, deduplicated=deduplicated)
            raise HTTPException(status_code=504, detail=response.model_dump(mode="json"))
        if response.status == TaskStatus.failed:
            raise HTTPException(status_code=500, detail=response.model_dump(mode="json"))
        if response.status == TaskStatus.cancelled:
            raise HTTPException(status_code=409, detail=response.model_dump(mode="json"))
        if response.status == TaskStatus.timeout:
            raise HTTPException(status_code=504, detail=response.model_dump(mode="json"))
        return response

    @app.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED, include_in_schema=True)
    @app.post("/v1/tasks", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
    def create_task(payload: RerankRequest) -> TaskResponse:
        video_path = _safe_video_path(payload.video_path, settings)
        try:
            record, deduplicated = manager.submit(video_path, payload.query, TaskMode.burst_async, payload.request_id)
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail={"reason": "overloaded", "message": str(exc)}) from exc
        return _response(record, manager, settings, deduplicated=deduplicated)

    @app.post("/v1/tasks/upload", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
    async def create_upload_task(
        query: str = Form(..., min_length=1, max_length=4096),
        request_id: str | None = Form(default=None, max_length=128),
        file: UploadFile = File(...),
    ) -> TaskResponse:
        query = query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query must not be blank")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="unsupported video extension")
        settings.upload_root.mkdir(parents=True, exist_ok=True)
        destination = settings.upload_root / f"{uuid.uuid4().hex}{suffix}"
        size = 0
        try:
            with destination.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="uploaded video exceeds size limit")
                    handle.write(chunk)
            video_path = _safe_video_path(str(destination), settings)
            try:
                record, deduplicated = manager.submit(video_path, query, TaskMode.burst_async, request_id)
            except QueueFullError as exc:
                raise HTTPException(status_code=429, detail={"reason": "overloaded", "message": str(exc)}) from exc
            return _response(record, manager, settings, deduplicated=deduplicated)
        except HTTPException:
            destination.unlink(missing_ok=True)
            raise
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get("/tasks/{task_id}", response_model=TaskResponse, include_in_schema=True)
    @app.get("/v1/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str) -> TaskResponse:
        try:
            record = manager.get(task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        return _response(record, manager, settings)

    @app.get("/tasks/{task_id}/results", response_model=dict, include_in_schema=True)
    @app.get("/v1/tasks/{task_id}/results", response_model=dict)
    def get_task_results(task_id: str) -> dict:
        """Return results separately so polling metadata and candidate payloads can evolve independently."""
        try:
            record = manager.get(task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        # TIMEOUT may still carry an explicit Level-2 coarse fallback when the
        # backend finished after the asynchronous budget.  Expose that
        # diagnostic result instead of forcing clients to read task metadata.
        if record.status not in {TaskStatus.succeeded, TaskStatus.timeout} or (
            record.status == TaskStatus.timeout and record.result is None
        ):
            raise HTTPException(status_code=409, detail={"status": record.status.value, "stage": record.current_stage})
        return {
            "task_id": task_id,
            "video_id": record.video_id,
            "status": record.status.value,
            "canonical_status": _canonical_task_status(record),
            "result": record.result,
            "degraded": record.degraded,
        }

    @app.delete("/tasks/{task_id}", response_model=TaskResponse, include_in_schema=True)
    @app.delete("/v1/tasks/{task_id}", response_model=TaskResponse)
    def cancel_task(task_id: str) -> TaskResponse:
        try:
            record = manager.cancel(task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        return _response(record, manager, settings)

    @app.post("/v1/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
    def save_feedback(payload: FeedbackRequest) -> FeedbackResponse:
        try:
            record = manager.get(payload.task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        # A TIMEOUT task may expose an explicit Level-2 coarse fallback after
        # late backend completion.  That result is user-visible and must be
        # feedback-able just like a normal SUCCESS result; unfinished
        # timeout tasks (without a result) remain rejected.
        if record.status not in {TaskStatus.succeeded, TaskStatus.timeout} or (
            record.status == TaskStatus.timeout and record.result is None
        ):
            raise HTTPException(status_code=409, detail="feedback requires a completed task result")
        if (payload.adjusted_start_s is not None or payload.adjusted_end_s is not None) and not payload.candidate_id:
            raise HTTPException(status_code=400, detail="candidate_id is required when submitting adjusted boundaries")
        settings.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "feedback_version": "feedback-v1",
            "task_id": record.task_id,
            "request_id": record.request_id,
            "label": _canonical_feedback_label(payload.label.value),
            "adjusted_start_s": payload.adjusted_start_s,
            "adjusted_end_s": payload.adjusted_end_s,
            "comment": payload.comment,
            "model_version": record.model_version or manager.model_version,
            "policy_version": settings.policy_version,
            "created_at": utcnow().isoformat().replace("+00:00", "Z"),
            "video_id": record.video_id,
            "query": record.query,
            "candidate_id": payload.candidate_id,
            "final_score": _feedback_candidate_score(record.result, payload.candidate_id),
            "query_hash": hashlib.sha256(record.query.encode("utf-8")).hexdigest(),
        }
        model_start, model_end = _feedback_candidate_bounds(record.result, payload.candidate_id)
        event.update({
            "model_start": model_start,
            "model_end": model_end,
            "user_start": payload.adjusted_start_s,
            "user_end": payload.adjusted_end_s,
        })
        with feedback_lock, settings.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return FeedbackResponse(
            saved=True,
            task_id=record.task_id,
            label=FeedbackLabel(_canonical_feedback_label(payload.label.value)),
            feedback_version=event["feedback_version"],
        )

    return app
