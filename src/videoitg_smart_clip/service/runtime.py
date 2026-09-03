"""Bounded task runtime and backend workers for the restricted service."""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

from .models import TaskMode, TaskStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def frame_score_predictions(frame_scores, fps: float, duration: float, segment_seconds: float = 5.0, top_k: int = 3) -> list[dict]:
    """Convert VideoITG frame scores into non-overlapping preview intervals."""

    selected: list[dict] = []
    for item in frame_scores:
        center = float(item["frame_index"]) / fps
        start = max(0.0, center - segment_seconds / 2.0)
        end = min(duration, start + segment_seconds)
        if end <= start:
            continue
        candidate = {
            "candidate_id": f"frame_{int(item['frame_index'])}",
            "start_s": start,
            "end_s": end,
            "score": float(item["score"]),
        }
        if any(max(start, old["start_s"]) < min(end, old["end_s"]) for old in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


class BackendWorker(Protocol):
    model_version: str
    device: str

    def run(self, video_path: Path, query: str) -> dict:
        ...

    @property
    def loaded(self) -> bool:
        ...


class StubWorker:
    """Deterministic worker used for service contract tests and smoke runs."""

    model_version = "stub-v1"
    device = "cpu"

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s

    @property
    def loaded(self) -> bool:
        return True

    def run(self, video_path: Path, query: str) -> dict:
        if self.delay_s:
            time.sleep(self.delay_s)
        return {
            "predictions": [{"candidate_id": "stub", "start_s": 0.0, "end_s": 5.0, "score": 1.0}],
            "runtime": {"backend": "stub", "device": self.device, "sampled_frames": 0},
            "video_path": str(video_path),
            "query_length": len(query),
        }


class VideoITGWorker:
    """One lazily loaded VideoITG-8B instance bound to one GPU."""

    def __init__(self, model_path: Path, device: str, *, target_fps: float = 2.0, max_frames: int = 16) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.model_version = self.model_path.name
        self.target_fps = target_fps
        self.max_frames = max_frames
        self._reranker = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._reranker is not None

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        with self._load_lock:
            if self._reranker is None:
                import torch

                if self.device.startswith("cuda"):
                    torch.cuda.set_device(self.device)
                from videoitg_smart_clip.reranker import VideoITGReranker

                self._reranker = VideoITGReranker(
                    self.model_path,
                    device=self.device,
                    target_fps=self.target_fps,
                    max_frames_per_candidate=self.max_frames,
                    frame_score_topk=8,
                )
        return self._reranker

    def run(self, video_path: Path, query: str) -> dict:
        return self.run_prepared(video_path, query, None)

    def prepare(self, video_path: Path):
        """Decode the fixed16 CPU-side sample for lookahead prefetch."""

        from videoitg_smart_clip.reranker import CandidateSegment

        reranker = self._get_reranker()
        candidate = CandidateSegment(video_path, 0.0, None, candidate_id="full")
        return reranker.read_candidate(candidate)

    def run_prepared(self, video_path: Path, query: str, prepared=None) -> dict:
        from videoitg_smart_clip.reranker import CandidateSegment

        reranker = self._get_reranker()
        candidate = CandidateSegment(video_path, 0.0, None, candidate_id="full")
        prepared_map = {candidate: prepared} if prepared is not None else None
        scored = reranker.rank(query, [candidate], prepared_candidates=prepared_map)[0]
        fps = float(scored.runtime["fps"])
        duration = float(scored.runtime["duration_s"])
        return {
            "predictions": frame_score_predictions(scored.frame_scores, fps, duration),
            "runtime": dict(scored.runtime) | {"backend": "videoitg", "device": self.device},
            "video_path": str(video_path),
        }


class CoarseToFineWorker:
    """Service worker for the new pipeline contract.

    The default local worker uses deterministic CPU feature extraction and the
    TimeLens adapter.  A verified TimeLens implementation can be injected
    without changing task/service contracts.
    """

    model_version = "TimeLens-8B-pending"
    device = "cpu"

    def __init__(
        self,
        feature_root: Path,
        *,
        grounder=None,
        feature_encoder=None,
        boundary_refiner=None,
        ranker=None,
        deduplicator=None,
        no_match=None,
        top_n: int = 20,
        top_k: int = 3,
        feature_sample_fps: float = 1.0,
        feature_max_frames: int = 16,
    ) -> None:
        from videoitg_smart_clip.grounding import TimeLensGrounder
        from videoitg_smart_clip.preprocessing import FeatureCache, HashFeatureEncoder, cache_identity
        from videoitg_smart_clip.retrieval import CachedCosineRetriever

        self.encoder = feature_encoder or HashFeatureEncoder()
        feature_model = getattr(self.encoder, "model_name", "lightweight-vision-text-v1")
        query_encoder = getattr(self.encoder, "encode_query", None)
        self.cache = FeatureCache(feature_root, feature_model=feature_model)
        self.retriever = CachedCosineRetriever(self.cache, query_encoder=query_encoder)
        self.grounder = grounder or TimeLensGrounder()
        self.boundary_refiner = boundary_refiner
        self.ranker = ranker
        self.deduplicator = deduplicator
        self.no_match = no_match
        self.device = getattr(self.encoder, "device", self.device)
        self.model_version = getattr(self.grounder, "model_version", self.model_version)
        self.top_n = top_n
        self.top_k = top_k
        if feature_sample_fps <= 0:
            raise ValueError("feature_sample_fps must be positive")
        if feature_max_frames <= 0:
            raise ValueError("feature_max_frames must be positive")
        self.feature_sample_fps = float(feature_sample_fps)
        self.feature_max_frames = int(feature_max_frames)
        self._indexed: set[str] = set()
        self._cache_identity = cache_identity

    @property
    def loaded(self) -> bool:
        return True

    @staticmethod
    def _duration(video_path: Path) -> float:
        try:
            from decord import VideoReader, cpu
            reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
            return max(1.0, len(reader) / float(reader.get_avg_fps()))
        except Exception:
            return 5.0

    def _ensure_index(self, video_path: Path) -> str:
        video_id = video_path.stem
        # Initialize CUDA-backed encoders before importing/using decord; this
        # avoids a known decoder/CUDA initialization conflict in the local env.
        if hasattr(self.encoder, "load"):
            self.encoder.load()
        duration = self._duration(video_path)
        key = self._cache_identity(
            video_id,
            self.encoder.version,
            {"segment": "uniform", "sample_fps": self.feature_sample_fps},
            {"fps": self.feature_sample_fps, "max_frames": self.feature_max_frames},
        )
        count = max(1, int(round(duration * self.feature_sample_fps)))
        count = min(count, self.feature_max_frames)

        def extractor():
            import numpy as np
            if hasattr(self.encoder, "decode_and_encode"):
                return self.encoder.decode_and_encode(
                    video_path,
                    sample_fps=self.feature_sample_fps,
                    max_frames=self.feature_max_frames,
                )
            timestamps = np.linspace(
                0.0,
                max(0.0, duration - 1.0 / self.feature_sample_fps),
                count,
                dtype=np.float32,
            )
            frames = [np.full((2, 2, 3), int(i) % 255, dtype=np.uint8) for i in range(count)]
            return self.encoder.encode(frames), timestamps
        self.retriever.index(video_path, video_id, key=key, extractor=extractor, video_duration=duration)
        self._indexed.add(video_id)
        return video_id

    def run(self, video_path: Path, query: str) -> dict:
        return self.run_with_progress(video_path, query, None)

    def run_with_progress(self, video_path: Path, query: str, stage_callback=None) -> dict:
        from videoitg_smart_clip.pipeline import CoarseToFinePipeline
        if stage_callback:
            stage_callback("INDEXING", 0.25)
        video_id = self._ensure_index(video_path)
        pipeline = CoarseToFinePipeline(
            self.retriever,
            self.grounder,
            boundary_refiner=self.boundary_refiner,
            ranker=self.ranker,
            deduplicator=self.deduplicator,
            no_match=self.no_match,
            top_n=self.top_n,
            top_k=self.top_k,
        )
        duration = self.retriever.video_duration(video_id) if hasattr(self.retriever, "video_duration") else None
        result = pipeline.search(video_path, video_id, query, duration=duration, stage_callback=stage_callback)
        result["video_path"] = str(video_path)
        cache_hit = int(self.retriever.last_metrics.get("cache_hit", 0))
        result["runtime"] = {
            "backend": "coarse_to_fine",
            "device": self.device,
            "cache_hit": cache_hit,
            "cache_event": "cache_hit" if cache_hit else "cache_miss",
            "retrieval_latency_ms": self.retriever.last_metrics.get("retrieval_latency_ms", 0.0),
            "grounder_attention_implementation": getattr(self.grounder, "attention_implementation", None),
        }
        return result


@dataclass
class TaskRecord:
    task_id: str
    request_id: str | None
    mode: TaskMode
    video_path: Path
    query: str
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    status: TaskStatus = TaskStatus.queued
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False
    event: threading.Event = field(default_factory=threading.Event, repr=False)
    sequence: int = 0
    priority: int = 10
    prepared_future: Future | None = field(default=None, repr=False)
    video_id: str | None = None
    model_version: str | None = None
    progress: float = 0.0
    current_stage: str = "PENDING"
    error_code: str | None = None
    degraded: bool = False
    timeout_timer: threading.Timer | None = field(default=None, repr=False)


class QueueFullError(RuntimeError):
    pass


class TaskNotFoundError(KeyError):
    pass


class BoundedTaskManager:
    """Priority queue with steady requests ahead of asynchronous burst tasks."""

    def __init__(
        self,
        workers: Sequence[BackendWorker],
        *,
        queue_size: int = 8,
        policy_version: str = "E0_fixed16",
        estimated_service_ms: int = 2000,
        state_path: Path | None = None,
        task_timeout_s: float | None = None,
    ) -> None:
        if not workers:
            raise ValueError("at least one backend worker is required")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self.workers = list(workers)
        self.queue_size = queue_size
        self.policy_version = policy_version
        self.estimated_service_ms = estimated_service_ms
        if task_timeout_s is not None and task_timeout_s <= 0:
            raise ValueError("task_timeout_s must be positive when configured")
        self.task_timeout_s = task_timeout_s
        self.state_path = Path(state_path).expanduser() if state_path is not None else None
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        self._records: dict[str, TaskRecord] = {}
        self._request_index: dict[str, str] = {}
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._sequence = 0
        self._load_state()
        self._prefetch_index = 0
        self._prefetch_executor = ThreadPoolExecutor(max_workers=len(self.workers), thread_name_prefix="videoitg-prefetch")
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._worker_loop, args=(index,), name=f"videoitg-worker-{index}", daemon=True)
            for index in range(len(self.workers))
        ]
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _record_payload(self, record: TaskRecord) -> dict:
        return {
            "task_id": record.task_id,
            "request_id": record.request_id,
            "mode": record.mode.value,
            "video_path": str(record.video_path),
            "query": record.query,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "status": record.status.value,
            "started_at": record.started_at.isoformat() if record.started_at else None,
            "finished_at": record.finished_at.isoformat() if record.finished_at else None,
            "result": record.result,
            "error": record.error,
            "cancel_requested": record.cancel_requested,
            "sequence": record.sequence,
            "priority": record.priority,
            "video_id": record.video_id,
            "model_version": record.model_version,
            "progress": record.progress,
            "current_stage": record.current_stage,
            "error_code": record.error_code,
            "degraded": record.degraded,
        }

    def _persist_record(self, record: TaskRecord) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        record.updated_at = utcnow()
        payload = self._record_payload(record)
        with self._state_lock, self.state_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        latest: dict[str, dict] = {}
        try:
            lines = self.state_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                payload = json.loads(line)
                latest[str(payload["task_id"])] = payload
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        for task_id, payload in latest.items():
            try:
                status = TaskStatus(payload["status"])
                recovered_interrupted = status in {TaskStatus.queued, TaskStatus.running}
                previously_recovered = status is TaskStatus.failed and str(payload.get("error", "")).startswith("service_restart_recovery:")
                if status in {TaskStatus.queued, TaskStatus.running}:
                    status = TaskStatus.failed
                    error = "service_restart_recovery: task interrupted before completion"
                    finished_at = utcnow()
                else:
                    error = payload.get("error")
                    finished_at = self._parse_datetime(payload.get("finished_at"))
                record = TaskRecord(
                    task_id=task_id,
                    request_id=payload.get("request_id"),
                    mode=TaskMode(payload["mode"]),
                    video_path=Path(payload["video_path"]),
                    query=payload.get("query", ""),
                    created_at=self._parse_datetime(payload.get("created_at")) or utcnow(),
                    updated_at=self._parse_datetime(payload.get("updated_at")) or self._parse_datetime(payload.get("created_at")) or utcnow(),
                    status=status,
                    started_at=self._parse_datetime(payload.get("started_at")),
                    finished_at=finished_at,
                    result=payload.get("result"),
                    error=error,
                    cancel_requested=bool(payload.get("cancel_requested", False)),
                    sequence=int(payload.get("sequence", 0)),
                    priority=int(payload.get("priority", 10)),
                    video_id=payload.get("video_id"),
                    # Older JSONL records predate explicit model-version
                    # persistence; hydrate them from the current worker set
                    # while preserving the stored value when available.
                    model_version=payload.get("model_version") or self.model_version,
                    progress=float(payload.get("progress", 0.0)),
                    # A recovered interrupted task is terminal.  Do not leave
                    # the stale in-flight stage/error fields visible to API
                    # clients; they must be able to distinguish recovery from
                    # a normal stage snapshot.
                    current_stage=("FAILED" if (recovered_interrupted or previously_recovered) else str(payload.get("current_stage", "PENDING"))),
                    error_code=("SERVICE_RESTART_RECOVERY" if (recovered_interrupted or previously_recovered) else payload.get("error_code")),
                    degraded=bool(payload.get("degraded", False)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            record.event.set()
            self._records[record.task_id] = record
            if record.request_id:
                self._request_index[record.request_id] = record.task_id
            self._sequence = max(self._sequence, record.sequence)
            if record.status == TaskStatus.failed and record.error == "service_restart_recovery: task interrupted before completion":
                self._persist_record(record)

    @property
    def model_version(self) -> str:
        versions = {worker.model_version for worker in self.workers}
        return next(iter(versions)) if len(versions) == 1 else "+".join(sorted(versions))

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return sum(record.status == TaskStatus.queued for record in self._records.values())

    @property
    def loaded_workers(self) -> int:
        return sum(worker.loaded for worker in self.workers)

    def queue_position(self, task_id: str) -> int | None:
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.status != TaskStatus.queued:
                return None
            queued = [item for item in self._records.values() if item.status == TaskStatus.queued]
            return 1 + sum((item.priority, item.sequence) < (record.priority, record.sequence) for item in queued)

    def submit(self, video_path: Path, query: str, mode: TaskMode, request_id: str | None = None) -> tuple[TaskRecord, bool]:
        with self._lock:
            if request_id and request_id in self._request_index:
                return self._records[self._request_index[request_id]], True
            if self._queue.full():
                raise QueueFullError("bounded task queue is full")
            self._sequence += 1
            record = TaskRecord(
                task_id=uuid.uuid4().hex,
                request_id=request_id,
                mode=mode,
                video_path=video_path,
                query=query,
                sequence=self._sequence,
                video_id=Path(video_path).stem,
                model_version=self.model_version,
                current_stage="PENDING",
            )
            self._records[record.task_id] = record
            if request_id:
                self._request_index[request_id] = record.task_id
            priority = 0 if mode == TaskMode.steady else 10
            record.priority = priority
            if mode == TaskMode.steady and hasattr(self.workers[self._prefetch_index % len(self.workers)], "prepare"):
                prefetch_worker = self.workers[self._prefetch_index % len(self.workers)]
                self._prefetch_index += 1
                record.prepared_future = self._prefetch_executor.submit(prefetch_worker.prepare, video_path)
            self._queue.put_nowait((priority, record.sequence, record.task_id))
            self._persist_record(record)
            return record, False

    def wait(self, task_id: str, timeout_s: float) -> TaskRecord:
        record = self.get(task_id)
        record.event.wait(timeout=max(0.0, timeout_s))
        return self.get(task_id)

    def get(self, task_id: str) -> TaskRecord:
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise TaskNotFoundError(task_id)
            return record

    def cancel(self, task_id: str) -> TaskRecord:
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise TaskNotFoundError(task_id)
            if record.status == TaskStatus.queued:
                record.status = TaskStatus.cancelled
                record.finished_at = utcnow()
                if record.prepared_future is not None:
                    record.prepared_future.cancel()
                record.event.set()
                self._persist_record(record)
            elif record.status == TaskStatus.running:
                record.cancel_requested = True
            return record

    def mark_timeout(self, task_id: str) -> TaskRecord:
        """Persist an explicit timeout terminal state for deadline callers."""

        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise TaskNotFoundError(task_id)
            if record.status in {TaskStatus.queued, TaskStatus.running}:
                self._cancel_timeout_timer(record)
                record.status = TaskStatus.timeout
                record.error = "task exceeded request deadline"
                record.error_code = "TIMEOUT"
                record.current_stage = "TIMEOUT"
                record.finished_at = utcnow()
                record.event.set()
                self._persist_record(record)
            return record

    def _watchdog_timeout(self, task_id: str) -> None:
        """Expose async timeout at the budget; running GPU work is not killed."""
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.status != TaskStatus.running:
                return
            record.status = TaskStatus.timeout
            record.error = "task exceeded asynchronous timeout"
            record.error_code = "TIMEOUT"
            record.current_stage = "TIMEOUT"
            record.degraded = True
            record.finished_at = utcnow()
            self._persist_record(record)

    @staticmethod
    def _cancel_timeout_timer(record: TaskRecord) -> None:
        if record.timeout_timer is not None:
            record.timeout_timer.cancel()
            record.timeout_timer = None

    @staticmethod
    def _timeout_fallback(result: dict | None) -> dict | None:
        """Convert a late coarse-to-fine result into explicit Level-2 output."""
        if not isinstance(result, dict):
            return None
        source = result.get("candidates") or []
        coarse = []
        for index, original in enumerate(source):
            if not isinstance(original, dict):
                continue
            if original.get("coarse_start") is None or original.get("coarse_end") is None:
                continue
            row = dict(original)
            row["raw_start"] = row["coarse_start"]
            row["raw_end"] = row["coarse_end"]
            row["refined_start"] = row["coarse_start"]
            row["refined_end"] = row["coarse_end"]
            row["grounding_score"] = 0.0
            row["final_score"] = row.get("retrieval_score", 0.0)
            row["rank"] = index + 1
            row["degraded"] = True
            row["degrade_level"] = 2
            row["degrade_reason"] = "async_timeout_coarse_fallback"
            coarse.append(row)
        return {
            "status": "POSSIBLE" if coarse else "NO_MATCH",
            "predictions": coarse[:3],
            "candidates": coarse,
            "degraded": True,
            "degrade_level": 2,
            "degrade_reason": "async_timeout_coarse_fallback" if coarse else "async_timeout_no_candidate_fallback",
        }

    def _worker_loop(self, worker_index: int) -> None:
        worker = self.workers[worker_index]
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item[2] is None:
                self._queue.task_done()
                return
            _, _, task_id = item
            try:
                record = self.get(task_id)
                with self._lock:
                    if record.status != TaskStatus.queued:
                        continue
                    record.status = TaskStatus.running
                    record.started_at = utcnow()
                    record.current_stage = "PREPROCESSING"
                    record.progress = 0.10
                    if self.task_timeout_s is not None:
                        record.timeout_timer = threading.Timer(
                            self.task_timeout_s,
                            self._watchdog_timeout,
                            args=(record.task_id,),
                        )
                        record.timeout_timer.daemon = True
                        record.timeout_timer.start()
                    self._persist_record(record)
                try:
                    prepared = None
                    if record.prepared_future is not None:
                        try:
                            prepared = record.prepared_future.result()
                        except Exception:
                            # A decode failure is reported by the normal
                            # backend path rather than returning stale data.
                            prepared = None
                    if prepared is not None and hasattr(worker, "run_prepared"):
                        with self._lock:
                            record.current_stage = "GROUNDING"
                            record.progress = 0.65
                        result = worker.run_prepared(record.video_path, record.query, prepared)
                    else:
                        with self._lock:
                            record.current_stage = "GROUNDING"
                            record.progress = 0.65
                        if hasattr(worker, "run_with_progress"):
                            def update_stage(stage: str, progress: float) -> None:
                                with self._lock:
                                    if record.status == TaskStatus.running:
                                        record.current_stage = stage
                                        record.progress = max(0.0, min(1.0, float(progress)))
                                        self._persist_record(record)
                            result = worker.run_with_progress(record.video_path, record.query, update_stage)
                        else:
                            result = worker.run(record.video_path, record.query)
                    elapsed_s = (utcnow() - (record.started_at or utcnow())).total_seconds()
                    with self._lock:
                        if record.status == TaskStatus.timeout:
                            # Watchdog already exposed TIMEOUT; retain late
                            # coarse windows when the backend eventually ends.
                            fallback = self._timeout_fallback(result if isinstance(result, dict) else None)
                            record.degraded = True
                            record.result = fallback or {
                                "status": "NO_MATCH",
                                "predictions": [],
                                "candidates": [],
                                "degraded": True,
                                "degrade_level": 2,
                                "degrade_reason": "async_timeout_no_candidate_fallback",
                            }
                            self._persist_record(record)
                        elif record.status == TaskStatus.running and record.cancel_requested:
                            # A running backend call cannot be force-killed
                            # safely (especially on CUDA), but a user cancel
                            # must still win over the eventual backend result.
                            # Discard the late result and expose CANCELLED.
                            record.status = TaskStatus.cancelled
                            record.error = "task cancelled by user"
                            record.error_code = "CANCELLED"
                            record.current_stage = "CANCELLED"
                            record.result = None
                            record.finished_at = utcnow()
                            self._persist_record(record)
                        elif record.status == TaskStatus.running and self.task_timeout_s is not None and elapsed_s > self.task_timeout_s:
                            fallback = self._timeout_fallback(result if isinstance(result, dict) else None)
                            record.status = TaskStatus.timeout
                            record.error = "task exceeded asynchronous timeout"
                            record.error_code = "TIMEOUT"
                            record.current_stage = "TIMEOUT"
                            record.degraded = True
                            record.result = fallback or {
                                "status": "NO_MATCH",
                                "predictions": [],
                                "candidates": [],
                                "degraded": True,
                                "degrade_level": 2,
                                "degrade_reason": "async_timeout_no_candidate_fallback",
                            }
                            record.finished_at = utcnow()
                            self._persist_record(record)
                        elif record.status == TaskStatus.running:
                            record.current_stage = "POSTPROCESSING"
                            record.progress = 0.90
                            if isinstance(result, dict):
                                record.degraded = bool(result.get("degraded", False))
                            record.status = TaskStatus.succeeded
                            record.result = result
                            record.current_stage = "SUCCESS"
                            record.progress = 1.0
                            record.finished_at = utcnow()
                            self._persist_record(record)
                except Exception as exc:  # pragma: no cover - exercised by real backend
                    with self._lock:
                        elapsed_s = (utcnow() - (record.started_at or utcnow())).total_seconds()
                        if self.task_timeout_s is not None and elapsed_s > self.task_timeout_s:
                            record.status = TaskStatus.timeout
                            record.error = "task exceeded asynchronous timeout"
                            record.error_code = "TIMEOUT"
                            record.current_stage = "TIMEOUT"
                            record.degraded = True
                            record.result = {
                                "status": "NO_MATCH",
                                "predictions": [],
                                "candidates": [],
                                "degraded": True,
                                "degrade_level": 2,
                                "degrade_reason": "async_timeout_no_candidate_fallback",
                            }
                        else:
                            record.status = TaskStatus.failed
                            record.error = f"{type(exc).__name__}: {exc}"
                            record.error_code = "OOM" if "out of memory" in str(exc).lower() else type(exc).__name__.upper()
                            record.current_stage = "FAILED"
                        record.finished_at = utcnow()
                        self._persist_record(record)
                    # Persist the terminal state before optional CUDA cleanup:
                    # importing torch can be slow on a cold worker and must
                    # not turn a deterministic backend failure into an API
                    # timeout.
                    if "out of memory" in str(exc).lower():
                        # Do not import/initialize torch from an exception
                        # handler running on a service worker thread.  A cold
                        # import can block for a long time (or contend with
                        # model initialization), turning an explicit backend
                        # failure into an apparent task hang.  If torch is
                        # already loaded by the backend, best-effort cleanup
                        # remains safe and non-blocking.
                        import sys

                        torch = sys.modules.get("torch")
                        try:
                            if torch is not None and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                finally:
                    with self._lock:
                        self._cancel_timeout_timer(record)
                    record.event.set()
            finally:
                self._queue.task_done()

    def shutdown(self, wait: bool = True) -> None:
        self._stop.set()
        # Cancel queued work before inserting sentinels. This keeps shutdown
        # bounded even when the queue is full; running backend calls are
        # allowed to finish and are joined below.
        with self._lock:
            for record in self._records.values():
                self._cancel_timeout_timer(record)
                if record.status == TaskStatus.queued:
                    record.status = TaskStatus.cancelled
                    record.finished_at = utcnow()
                    if record.prepared_future is not None:
                        record.prepared_future.cancel()
                    record.event.set()
                    self._persist_record(record)
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        # Worker loops poll the stop event while waiting on the queue, so no
        # sentinel insertion is needed (and this remains safe when there are
        # more workers than queue slots).
        if wait:
            for thread in self._threads:
                thread.join(timeout=5)
        self._prefetch_executor.shutdown(wait=wait)
