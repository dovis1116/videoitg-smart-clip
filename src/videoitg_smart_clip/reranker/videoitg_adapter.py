"""Thin, memory-bounded adapter around the official VideoITG inference path.

The upstream model scores sampled frames rather than temporal intervals.  This
adapter keeps the frame-to-video mapping and turns each candidate interval into
one scored object.  Candidates are processed sequentially by default because
the Phase 0 measurement showed that 512 frames do not fit on one 24 GiB GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateSegment:
    """A candidate interval produced by the coarse retriever."""

    video_path: str | Path
    start_s: float
    end_s: float | None
    candidate_id: str = ""
    frame_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise ValueError("candidate interval must satisfy start_s >= 0")
        if self.end_s is None:
            if self.start_s != 0:
                raise ValueError("full-video candidate must start at 0")
        elif self.end_s <= self.start_s:
            raise ValueError("candidate interval must satisfy 0 <= start_s < end_s")
        if any(i < 0 for i in self.frame_indices):
            raise ValueError("frame_indices must be non-negative")


@dataclass
class ScoredCandidate:
    """VideoITG scores with provenance needed by evaluation and the service."""

    candidate: CandidateSegment
    segment_score: float
    frame_scores: list[dict[str, float | int]]
    sampled_frame_indices: list[int]
    model_version: str
    runtime: dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class PreparedCandidate:
    """CPU-side frames prepared ahead of a model call.

    A prepared candidate is intentionally kept separate from ``ScoredCandidate``
    so callers can overlap CPU decoding of the next request with GPU inference
    for the current request without changing the sampled frame ids.
    """

    candidate: CandidateSegment
    frames: np.ndarray
    frame_indices: list[int]
    fps: float
    duration_s: float


def interval_frame_indices(
    total_frames: int,
    fps: float,
    start_s: float,
    end_s: float,
    target_fps: float = 2.0,
    max_frames: int = 32,
) -> list[int]:
    """Return deterministic, clamped frame ids for one time interval."""

    if total_frames <= 0 or fps <= 0:
        raise ValueError("total_frames and fps must be positive")
    if start_s < 0 or end_s <= start_s:
        raise ValueError("candidate interval must satisfy 0 <= start_s < end_s")
    if target_fps <= 0 or max_frames <= 0:
        raise ValueError("target_fps and max_frames must be positive")
    first = max(0, min(total_frames - 1, int(start_s * fps)))
    last = max(first + 1, min(total_frames, int(np.ceil(end_s * fps))))
    stride = max(1, round(fps / target_fps))
    ids = list(range(first, last, stride))
    if not ids:
        ids = [first]
    if len(ids) <= max_frames:
        return ids
    scale = len(ids) / max_frames
    return [ids[round((i + 1) * scale - 1)] for i in range(max_frames)]


class VideoITGReranker:
    """Lazy-loading VideoITG adapter with a bounded per-candidate frame budget."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        target_fps: float = 2.0,
        max_frames_per_candidate: int = 32,
        frame_score_topk: int = 8,
        empty_cache_each_candidate: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.target_fps = target_fps
        self.max_frames_per_candidate = max_frames_per_candidate
        self.frame_score_topk = frame_score_topk
        self.empty_cache_each_candidate = empty_cache_each_candidate
        self._tokenizer = None
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from eagle.mm_utils import get_model_name_from_path
        from eagle.model.builder import load_pretrained_model

        self._tokenizer, self._model, self._processor, _ = load_pretrained_model(
            self.model_path,
            None,
            get_model_name_from_path(self.model_path),
            device_map=self.device,
        )
        self._model.half().eval().to(self.device)

    def _read_candidate(self, candidate: CandidateSegment) -> tuple[np.ndarray, list[int], float, float]:
        from decord import VideoReader, cpu

        reader = VideoReader(str(candidate.video_path), ctx=cpu(0), num_threads=4)
        fps = float(reader.get_avg_fps())
        if candidate.frame_indices:
            ids = [i for i in candidate.frame_indices if i < len(reader)]
            if not ids:
                raise ValueError(f"no candidate frame is inside video: {candidate.video_path}")
        else:
            end_s = candidate.end_s if candidate.end_s is not None else len(reader) / fps
            ids = interval_frame_indices(
                len(reader),
                fps,
                candidate.start_s,
                end_s,
                target_fps=self.target_fps,
                max_frames=self.max_frames_per_candidate,
            )
        frames = reader.get_batch(ids).asnumpy()
        return frames, ids, fps, len(reader) / fps

    def read_candidate(self, candidate: CandidateSegment) -> PreparedCandidate:
        """Decode and sample one candidate on CPU for optional prefetching."""

        frames, ids, fps, duration_s = self._read_candidate(candidate)
        return PreparedCandidate(candidate, frames, ids, fps, duration_s)

    def token_length(self, query: str) -> int:
        """Return the upstream multimodal token length used for safe batching."""

        if not query.strip():
            raise ValueError("query must not be empty")
        self._load()
        from eagle.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from eagle.mm_utils import tokenizer_image_token

        prompt = DEFAULT_IMAGE_TOKEN + query.strip() + "\n"
        return int(tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").numel())

    def rank_batch(
        self,
        queries: Sequence[str],
        candidates: Sequence[CandidateSegment],
        *,
        prepared_candidates: dict[CandidateSegment, PreparedCandidate] | None = None,
    ) -> list[ScoredCandidate]:
        """Score a small batch whose tokenized query lengths are identical.

        VideoITG's upstream grounding head returns one variable-length list of
        frame logits per batch item.  Exact token-length bucketing avoids the
        FlashAttention padding/index failure observed for mixed-length queries.
        The caller must keep the batch small enough for the model's measured
        single-GPU memory budget.
        """

        if len(queries) != len(candidates) or not candidates:
            raise ValueError("queries and candidates must be non-empty and have equal length")
        if any(not query.strip() for query in queries):
            raise ValueError("query must not be empty")
        self._load()
        import torch
        from eagle.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from eagle.mm_utils import tokenizer_image_token

        videos = []
        input_ids_list = []
        metadata = []
        for query, candidate in zip(queries, candidates):
            started = time.perf_counter()
            prepared = prepared_candidates.get(candidate) if prepared_candidates else None
            if prepared is not None:
                frames = prepared.frames
                ids = list(prepared.frame_indices)
                fps = prepared.fps
                duration_s = prepared.duration_s
                read_seconds = 0.0
                prefetched = 1
            else:
                read_started = started
                frames, ids, fps, duration_s = self._read_candidate(candidate)
                read_seconds = time.perf_counter() - read_started
                prefetched = 0
            if len(ids) > self.max_frames_per_candidate:
                keep = np.round(np.linspace(0, len(ids) - 1, self.max_frames_per_candidate)).astype(int).tolist()
                frames = frames[keep]
                ids = [ids[i] for i in keep]
            preprocess_started = time.perf_counter()
            videos.append(self._processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device))
            prompt = DEFAULT_IMAGE_TOKEN + query.strip() + "\n"
            input_ids_list.append(tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"))
            metadata.append({
                "candidate": candidate,
                "ids": ids,
                "fps": fps,
                "duration_s": duration_s,
                "read_seconds": read_seconds,
                "prefetched": prefetched,
                "preprocess_seconds": time.perf_counter() - preprocess_started,
                "started": started,
            })

        lengths = {int(item.numel()) for item in input_ids_list}
        if len(lengths) != 1:
            raise ValueError("rank_batch requires identical tokenized query lengths")
        input_ids = torch.stack(input_ids_list, dim=0).to(self.device)
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_id).to(self.device)
        model_started = time.perf_counter()
        with torch.inference_mode():
            output = self._model(input_ids, attention_mask=attention_mask, images=videos)
            raw_logits = output.logits
            if isinstance(raw_logits, list):
                score_tensors = [item.sigmoid().flatten() for item in raw_logits]
            elif raw_logits.ndim == 2:
                score_tensors = [raw_logits[i].sigmoid().flatten() for i in range(raw_logits.shape[0])]
            else:
                raise RuntimeError(f"unexpected batched VideoITG logits shape: {tuple(raw_logits.shape)}")
            score_arrays = [item.detach().float().cpu().numpy() for item in score_tensors]
        model_seconds = time.perf_counter() - model_started
        if len(score_arrays) != len(metadata):
            raise RuntimeError(f"VideoITG returned {len(score_arrays)} batch items for {len(metadata)} requests")

        scored = []
        for item, scores in zip(metadata, score_arrays):
            if len(scores) != len(item["ids"]):
                raise RuntimeError(f"VideoITG returned {len(scores)} scores for {len(item['ids'])} frames")
            postprocess_started = time.perf_counter()
            order = np.argsort(-scores)
            topk = order[: min(self.frame_score_topk, len(order))]
            frame_scores = [{"frame_index": int(item["ids"][i]), "score": float(scores[i])} for i in order]
            scored.append(
                ScoredCandidate(
                    candidate=item["candidate"],
                    segment_score=float(scores[topk].mean()),
                    frame_scores=frame_scores,
                    sampled_frame_indices=[int(i) for i in item["ids"]],
                    model_version=Path(self.model_path).name,
                    runtime={
                        "wall_seconds": time.perf_counter() - item["started"],
                        "read_seconds": item["read_seconds"],
                        "preprocess_seconds": item["preprocess_seconds"],
                        "model_seconds": model_seconds,
                        "postprocess_seconds": time.perf_counter() - postprocess_started,
                        "sampled_frames": len(item["ids"]),
                        "fps": item["fps"],
                        "duration_s": item["duration_s"],
                        "prefetched": item["prefetched"],
                        "batch_size": len(metadata),
                        "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
                    },
                )
            )
        del videos, input_ids, output, raw_logits, score_tensors, score_arrays
        if torch.cuda.is_available() and self.empty_cache_each_candidate:
            torch.cuda.empty_cache()
        return sorted(scored, key=lambda item: item.segment_score, reverse=True)

    def rank(
        self,
        query: str,
        candidates: Sequence[CandidateSegment],
        *,
        prepared_candidates: dict[CandidateSegment, PreparedCandidate] | None = None,
    ) -> list[ScoredCandidate]:
        """Score candidates one at a time and return descending segment score."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if not candidates:
            return []
        self._load()
        import torch
        from eagle.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from eagle.mm_utils import tokenizer_image_token

        scored: list[ScoredCandidate] = []
        for candidate in candidates:
            started = time.perf_counter()
            prepared = prepared_candidates.get(candidate) if prepared_candidates else None
            if prepared is not None:
                frames = prepared.frames
                ids = prepared.frame_indices
                fps = prepared.fps
                duration_s = prepared.duration_s
                read_seconds = 0.0
                prefetched = 1
            else:
                read_started = started
                frames, ids, fps, duration_s = self._read_candidate(candidate)
                read_seconds = time.perf_counter() - read_started
                prefetched = 0
            # Keep this guard close to the model call so callers cannot bypass the budget.
            if len(ids) > self.max_frames_per_candidate:
                keep = np.round(
                    np.linspace(0, len(ids) - 1, self.max_frames_per_candidate)
                ).astype(int).tolist()
                frames = frames[keep]
                ids = [ids[i] for i in keep]
            preprocess_started = time.perf_counter()
            video = self._processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            video = video.half().to(self.device)
            prompt = DEFAULT_IMAGE_TOKEN + query.strip() + "\n"
            input_ids = tokenizer_image_token(
                prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)
            pad_id = self._tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self._tokenizer.eos_token_id
            attention_mask = input_ids.ne(pad_id).to(self.device)
            preprocess_seconds = time.perf_counter() - preprocess_started
            model_started = time.perf_counter()
            with torch.inference_mode():
                output = self._model(input_ids, attention_mask=attention_mask, images=[video])
                scores = output.logits[0].sigmoid().flatten().detach().float().cpu().numpy()
            model_seconds = time.perf_counter() - model_started
            if len(scores) != len(ids):
                raise RuntimeError(f"VideoITG returned {len(scores)} scores for {len(ids)} frames")
            postprocess_started = time.perf_counter()
            order = np.argsort(-scores)
            topk = order[: min(self.frame_score_topk, len(order))]
            segment_score = float(scores[topk].mean())
            frame_scores = [
                {"frame_index": int(ids[i]), "score": float(scores[i])}
                for i in order
            ]
            postprocess_seconds = time.perf_counter() - postprocess_started
            peak = float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
            scored.append(
                ScoredCandidate(
                    candidate=candidate,
                    segment_score=segment_score,
                    frame_scores=frame_scores,
                    sampled_frame_indices=[int(i) for i in ids],
                    model_version=Path(self.model_path).name,
                    runtime={
                        "wall_seconds": time.perf_counter() - started,
                        "read_seconds": read_seconds,
                        "preprocess_seconds": preprocess_seconds,
                        "model_seconds": model_seconds,
                        "postprocess_seconds": postprocess_seconds,
                        "sampled_frames": len(ids),
                        "fps": fps,
                        "duration_s": duration_s,
                        "prefetched": prefetched,
                        "peak_cuda_gib": peak,
                    },
                )
            )
            del video, output, scores
            if torch.cuda.is_available() and self.empty_cache_each_candidate:
                torch.cuda.empty_cache()
        return sorted(scored, key=lambda item: item.segment_score, reverse=True)
