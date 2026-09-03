"""Video decoding and lightweight visual feature extraction for G2.

The encoder is lazy-loaded so importing the service does not allocate a model.
Only sampled frames are decoded; the resulting image embeddings are suitable
for versioned :class:`FeatureCache` storage and query-time retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def decode_uniform_frames(
    video_path: str | Path,
    *,
    sample_fps: float = 1.0,
    max_frames: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Decode uniformly sampled RGB frames and return frames, timestamps, duration."""

    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    fps = float(reader.get_avg_fps())
    if fps <= 0 or len(reader) == 0:
        raise ValueError("video has no decodable frames")
    duration = len(reader) / fps
    stride = max(1, int(round(fps / sample_fps)))
    indices = np.arange(0, len(reader), stride, dtype=np.int64)
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames, dtype=np.int64)
        indices = indices[positions]
    batch = reader.get_batch(indices.tolist()).asnumpy()
    return [np.asarray(frame) for frame in batch], indices.astype(np.float32) / fps, float(duration)


class SigLIPFeatureEncoder:
    """Lazy SigLIP image/text encoder with normalized projected embeddings."""

    version = "siglip-so400m-patch14-384-v1"
    model_name = "google/siglip-so400m-patch14-384"

    def __init__(self, model_path: str | Path, *, device: str = "cuda:0", batch_size: int = 8) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from videoitg_smart_clip.dependency_locks import TRANSFORMERS_IMPORT_LOCK

        with TRANSFORMERS_IMPORT_LOCK:
            import torch
            from transformers import AutoProcessor, SiglipModel
            self._processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
            if self.device.startswith("cuda"):
                kwargs.update({"dtype": torch.float16, "device_map": self.device})
            else:
                kwargs["dtype"] = torch.float32
            self._model = SiglipModel.from_pretrained(self.model_path, **kwargs).eval()

    def load(self) -> None:
        """Eagerly initialize the model before starting video decoding."""

        self._load()

    @staticmethod
    def _normalize(features):
        import torch

        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _feature_tensor(output):
        """Normalize the Transformers 4.x tensor and 5.x model-output APIs."""
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def encode(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Encode RGB frames once; output shape is ``[N, 1152]`` for this model."""

        if not frames:
            return np.zeros((0, 0), dtype=np.float32)
        from PIL import Image
        import torch

        self._load()
        rows = []
        for start in range(0, len(frames), self.batch_size):
            images = [Image.fromarray(np.asarray(frame).astype(np.uint8)) for frame in frames[start : start + self.batch_size]]
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self._normalize(self._feature_tensor(self._model.get_image_features(**inputs)))
            rows.append(features.float().cpu().numpy())
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)

    def decode_and_encode(
        self,
        video_path: str | Path,
        *,
        sample_fps: float = 1.0,
        max_frames: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode a video once and encode its sampled frames for cache storage."""

        frames, timestamps, _ = decode_uniform_frames(video_path, sample_fps=sample_fps, max_frames=max_frames)
        return self.encode(frames), timestamps

    def encode_query(self, query: str) -> np.ndarray:
        import torch

        self._load()
        inputs = self._processor(
            text=[query], return_tensors="pt", padding="max_length", truncation=True, max_length=64
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self._normalize(self._feature_tensor(self._model.get_text_features(**inputs)))
        return features.float().cpu().numpy()[0]
