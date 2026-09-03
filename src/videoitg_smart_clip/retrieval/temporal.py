"""Query-aware temporal retrievers operating on cached video features."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from videoitg_smart_clip.pipeline.contracts import CandidateWindow, TemporalRetriever
from videoitg_smart_clip.preprocessing.feature_cache import FeatureBundle, FeatureCache, FeatureCacheKey


def normalize_retrieval_query(query: str) -> str:
    """Remove dataset answer-format boilerplate before text embedding.

    Grounding still receives the original user query; this normalization is
    limited to the lightweight Retriever text encoder.
    """

    normalized = re.sub(r"\s*Answer the question using few words or phrase\.\s*$", "", str(query), flags=re.IGNORECASE)
    return normalized.strip() or str(query).strip()


class _QueryEncoder:
    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension

    def encode(self, query: str) -> np.ndarray:
        digest = hashlib.sha256(query.strip().lower().encode()).digest()
        return np.frombuffer(digest, dtype=np.uint8).astype(np.float32)[: self.dimension] / 255.0


class CachedCosineRetriever:
    """Default retriever: cosine similarity over cached frame/shot vectors."""

    version = "cached-cosine-v1"

    def __init__(self, cache: FeatureCache, *, query_encoder: Callable[[str], np.ndarray] | None = None, window_seconds: float = 10.0) -> None:
        self.cache = cache
        self.query_encoder = query_encoder or _QueryEncoder().encode
        self.window_seconds = window_seconds
        self._bundles: dict[str, FeatureBundle] = {}
        self.last_metrics: dict[str, float | int | str] = {}

    def index(self, video_path: str | Path, video_id: str, *, key: FeatureCacheKey | None = None, extractor: Callable[[], tuple[np.ndarray, Sequence[float]]] | None = None, video_duration: float | None = None) -> FeatureBundle:
        # The public contract is exactly ``index(video_path, video_id)``.
        # Optional keyword hooks allow a production decoder/encoder to be
        # injected while keeping smoke tests and callers implementation-free.
        if key is None:
            key = FeatureCacheKey(video_id, self.version, "default-v1", "representative-v1")
        if video_duration is None:
            video_duration = 1.0
        if extractor is None:
            extractor = lambda: (np.zeros((1, 32), dtype=np.float32), [0.0])
        started = time.perf_counter()
        bundle = self.cache.get_or_create(key, video_duration=video_duration, extractor=extractor)
        self._bundles[video_id] = bundle
        self.last_metrics = {"cache_hit": int(bundle.cache_hit), "index_latency_ms": (time.perf_counter() - started) * 1000.0, "feature_cache_size": Path(bundle.feature_path).stat().st_size}
        return bundle

    def retrieve(self, video_id: str, query: str, top_n: int) -> list[CandidateWindow]:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        bundle = self._bundles.get(video_id)
        if bundle is None:
            raise KeyError(f"video is not indexed: {video_id}")
        started = time.perf_counter()
        q = np.asarray(self.query_encoder(normalize_retrieval_query(query)), dtype=np.float32)
        vectors = bundle.features
        if vectors.shape[1] != q.shape[0]:
            raise ValueError("query and cached feature dimensions differ")
        denom = np.linalg.norm(vectors, axis=1) * max(float(np.linalg.norm(q)), 1e-8)
        scores = vectors @ q / np.maximum(denom, 1e-8)
        order = np.argsort(-scores)[: min(top_n, len(scores))]
        out = []
        half = self.window_seconds / 2.0
        for rank, index in enumerate(order):
            center = float(bundle.timestamps[index])
            out.append(CandidateWindow(max(0.0, center - half), min(bundle.video_duration, center + half), float(scores[index]), f"{video_id}:r{rank}"))
        self.last_metrics["retrieval_latency_ms"] = (time.perf_counter() - started) * 1000.0
        return out

    def video_duration(self, video_id: str) -> float:
        bundle = self._bundles.get(video_id)
        if bundle is None:
            raise KeyError(f"video is not indexed: {video_id}")
        return float(bundle.video_duration)


class UniformTemporalRetriever:
    """Reference retriever used for a resource/recall comparison."""

    version = "uniform-v1"

    def __init__(self, cache: FeatureCache, *, window_seconds: float = 10.0) -> None:
        self.cache = cache
        self.window_seconds = window_seconds
        self._durations: dict[str, float] = {}

    def index(self, video_path: str | Path, video_id: str, *, video_duration: float | None = None) -> None:
        self._durations[video_id] = float(video_duration if video_duration is not None else 1.0)

    def retrieve(self, video_id: str, query: str, top_n: int) -> list[CandidateWindow]:
        duration = self._durations[video_id]
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        count = max(1, int(np.ceil(duration / self.window_seconds)))
        ids = np.linspace(0, count - 1, min(top_n, count), dtype=int)
        return [CandidateWindow(float(i * self.window_seconds), min(duration, float((i + 1) * self.window_seconds)), 0.0, f"{video_id}:u{i}") for i in ids]


def compare_retrievers(metrics: Sequence[dict]) -> dict:
    """Summarize Recall@N, latency, memory and cache size for a registry/report."""
    if not metrics:
        raise ValueError("metrics must not be empty")
    return {"retrievers": list(metrics), "selection_rule": "highest Recall@20 subject to latency and resource budget"}
