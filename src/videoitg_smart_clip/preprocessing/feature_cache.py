"""Versioned video-side feature cache.

The cache is deliberately model-agnostic: a feature encoder can be replaced
without changing retrieval or service code.  Cache validity requires all four
identity fields, preventing accidental reuse after preprocessing changes.
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureCacheKey:
    video_id: str
    feature_model_version: str
    preprocessing_config: str
    sampling_config: str


@dataclass
class FeatureBundle:
    key: FeatureCacheKey
    video_duration: float
    feature_model: str
    feature_model_version: str
    preprocessing_config: str
    sampling_config: str
    feature_path: str
    created_at: str
    features: np.ndarray
    timestamps: np.ndarray
    cache_hit: bool = False
    extraction_latency_ms: float | None = None

    def metadata(self) -> dict:
        return {
            "video_id": self.key.video_id,
            "video_duration": self.video_duration,
            "feature_model": self.feature_model,
            "feature_model_version": self.feature_model_version,
            "preprocessing_config": self.preprocessing_config,
            "sampling_config": self.sampling_config,
            "feature_path": self.feature_path,
            "created_at": self.created_at,
            "cache_hit": self.cache_hit,
            "extraction_latency_ms": self.extraction_latency_ms,
        }


class FeatureCache:
    def __init__(self, root: str | Path, *, feature_model: str = "deterministic-frame-v1") -> None:
        self.root = Path(root).expanduser()
        self.feature_model = feature_model
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_events: list[dict[str, str | bool]] = []
        self._lock = threading.RLock()

    def _stem(self, key: FeatureCacheKey) -> str:
        raw = "|".join((key.video_id, key.feature_model_version, key.preprocessing_config, key.sampling_config))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _paths(self, key: FeatureCacheKey) -> tuple[Path, Path]:
        stem = self._stem(key)
        return self.root / f"{stem}.npz", self.root / f"{stem}.json"

    @contextmanager
    def _file_lock(self, key: FeatureCacheKey):
        """Serialize first-writer extraction across cache instances/processes."""

        _, metadata_path = self._paths(key)
        lock_path = metadata_path.with_suffix(metadata_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+")
        try:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                # Thread-level locking below still protects platforms without
                # POSIX advisory locks; the project deployment target is Linux.
                pass
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            handle.close()

    def load(self, key: FeatureCacheKey) -> FeatureBundle | None:
        feature_path, metadata_path = self._paths(key)
        if not feature_path.is_file() or not metadata_path.is_file():
            self.cache_events.append({"video_id": key.video_id, "event": "cache_miss", "hit": False})
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(metadata.get(name) != getattr(key, name) for name in (
                "video_id", "feature_model_version", "preprocessing_config", "sampling_config"
            )):
                self.cache_events.append({"video_id": key.video_id, "event": "cache_miss", "hit": False})
                return None
            arrays = np.load(feature_path)
            bundle = FeatureBundle(
                key=key,
                video_duration=float(metadata["video_duration"]),
                feature_model=str(metadata["feature_model"]),
                feature_model_version=str(metadata["feature_model_version"]),
                preprocessing_config=str(metadata["preprocessing_config"]),
                sampling_config=str(metadata["sampling_config"]),
                feature_path=str(feature_path),
                created_at=str(metadata["created_at"]),
                features=np.asarray(arrays["features"]),
                timestamps=np.asarray(arrays["timestamps"]),
                cache_hit=True,
                extraction_latency_ms=(
                    float(metadata["extraction_latency_ms"])
                    if metadata.get("extraction_latency_ms") is not None
                    else None
                ),
            )
            self.cache_events.append({"video_id": key.video_id, "event": "cache_hit", "hit": True})
            return bundle
        except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
            self.cache_events.append({"video_id": key.video_id, "event": "cache_miss", "hit": False})
            return None

    def get_or_create(
        self,
        key: FeatureCacheKey,
        *,
        video_duration: float,
        extractor: Callable[[], tuple[np.ndarray, Sequence[float]]],
    ) -> FeatureBundle:
        with self._lock, self._file_lock(key):
            cached = self.load(key)
            if cached is not None:
                return cached
            extraction_started = time.perf_counter()
            features, timestamps = extractor()
            extraction_latency_ms = (time.perf_counter() - extraction_started) * 1000.0
            features = np.asarray(features, dtype=np.float32)
            timestamps = np.asarray(timestamps, dtype=np.float32)
            if features.ndim != 2 or len(features) != len(timestamps):
                raise ValueError("extractor must return [N,D] features and N timestamps")
            feature_path, metadata_path = self._paths(key)
            np.savez_compressed(feature_path, features=features, timestamps=timestamps)
            metadata = FeatureBundle(
                key=key,
                video_duration=float(video_duration),
                feature_model=self.feature_model,
                feature_model_version=key.feature_model_version,
                preprocessing_config=key.preprocessing_config,
                sampling_config=key.sampling_config,
                feature_path=str(feature_path),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                features=features,
                timestamps=timestamps,
                cache_hit=False,
                extraction_latency_ms=extraction_latency_ms,
            )
            metadata_path.write_text(json.dumps(metadata.metadata(), ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata


class HashFeatureEncoder:
    """Small deterministic encoder for smoke tests and CPU-only development."""

    version = "hash-frame-v1"

    def encode(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for frame in frames:
            digest = hashlib.sha256(np.asarray(frame).tobytes()).digest()
            row = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
            rows.append(row)
        return np.vstack(rows) if rows else np.zeros((0, 32), dtype=np.float32)


def cache_identity(video_id: str, feature_model_version: str, preprocessing_config: dict, sampling_config: dict) -> FeatureCacheKey:
    return FeatureCacheKey(
        video_id=video_id,
        feature_model_version=feature_model_version,
        preprocessing_config=json.dumps(preprocessing_config, sort_keys=True, separators=(",", ":")),
        sampling_config=json.dumps(sampling_config, sort_keys=True, separators=(",", ":")),
    )
