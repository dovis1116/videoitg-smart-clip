"""Pydantic contracts for the restricted local service."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskMode(str, Enum):
    steady = "steady"
    burst_async = "burst_async"


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    overloaded = "overloaded"
    timeout = "timeout"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            aliases = {"PENDING": cls.queued, "PREPROCESSING": cls.running, "INDEXING": cls.running, "RETRIEVING": cls.running, "GROUNDING": cls.running, "POSTPROCESSING": cls.running, "SUCCESS": cls.succeeded, "FAILED": cls.failed, "CANCELLED": cls.cancelled, "TIMEOUT": cls.timeout}
            return aliases.get(value.upper())
        return None


class TaskLifecycleStatus(str, Enum):
    """Canonical coarse-to-fine lifecycle names used by new clients."""

    PENDING = "PENDING"
    PREPROCESSING = "PREPROCESSING"
    INDEXING = "INDEXING"
    RETRIEVING = "RETRIEVING"
    GROUNDING = "GROUNDING"
    POSTPROCESSING = "POSTPROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


class FeedbackLabel(str, Enum):
    ACCEPT = "ACCEPT"
    IRRELEVANT = "IRRELEVANT"
    START_TOO_EARLY = "START_TOO_EARLY"
    START_TOO_LATE = "START_TOO_LATE"
    END_TOO_EARLY = "END_TOO_EARLY"
    END_TOO_LATE = "END_TOO_LATE"
    DUPLICATE = "DUPLICATE"
    MISS = "MISS"
    # Lowercase aliases remain accepted for existing browser clients.
    accepted = "accepted"
    irrelevant = "irrelevant"
    start_too_early = "start_too_early"
    start_too_late = "start_too_late"
    end_too_early = "end_too_early"
    end_too_late = "end_too_late"
    no_target_in_video = "no_target_in_video"
    duplicate = "duplicate"


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4096)
    video_path: str = Field(min_length=1, max_length=4096)
    request_id: str | None = Field(default=None, max_length=128)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class TaskResponse(BaseModel):
    task_id: str
    request_id: str | None
    mode: TaskMode
    status: TaskStatus
    queue_position: int | None = None
    estimated_wait_ms: int | None = None
    created_at: datetime
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    model_version: str
    policy_version: str
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False
    deduplicated: bool = False
    # New lifecycle observability fields. Defaults keep old API clients valid.
    video_id: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    current_stage: str | None = None
    error_code: str | None = None
    degraded: bool = False
    canonical_status: str | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    backend: str
    workers: int
    queue_depth: int
    queue_capacity: int
    policy_version: str
    model_version: str
    loaded_workers: int


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=128)
    label: FeedbackLabel
    adjusted_start_s: float | None = Field(default=None, ge=0)
    adjusted_end_s: float | None = Field(default=None, ge=0)
    comment: str | None = Field(default=None, max_length=1000)
    candidate_id: str | None = Field(default=None, max_length=256)

    @field_validator("adjusted_end_s")
    @classmethod
    def end_after_start(cls, value: float | None, info):
        start = info.data.get("adjusted_start_s")
        if value is not None and start is not None and value < start:
            raise ValueError("adjusted_end_s must be >= adjusted_start_s")
        return value

    @model_validator(mode="after")
    def require_complete_adjustment_pair(self):
        if (self.adjusted_start_s is None) != (self.adjusted_end_s is None):
            raise ValueError("adjusted_start_s and adjusted_end_s must be provided together")
        return self


class FeedbackResponse(BaseModel):
    saved: bool
    task_id: str
    label: FeedbackLabel
    feedback_version: str
