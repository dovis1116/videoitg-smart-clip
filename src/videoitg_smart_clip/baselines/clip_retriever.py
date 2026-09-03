"""B0 lightweight CLIP image-text candidate retriever."""

from __future__ import annotations

import numpy as np


class ClipRetriever:
    def __init__(self, model_path: str, device: str = "cuda:0", batch_size: int = 16):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_path, local_files_only=True)
        self._model = CLIPModel.from_pretrained(self.model_path, local_files_only=True).eval().to(self.device)
        if self.device.startswith("cuda"):
            self._model = self._model.half()

    def rank(self, query: str, video_path: str, candidates: list[dict]) -> list[dict]:
        self._load()
        import torch
        from decord import VideoReader, cpu
        from PIL import Image

        reader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        fps = float(reader.get_avg_fps())
        text_inputs = self._processor(text=[query], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.inference_mode():
            text_features = self._model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scored = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            indices = [min(len(reader) - 1, max(0, int(((c["start_s"] + c["end_s"]) / 2) * fps))) for c in batch]
            images = [Image.fromarray(x) for x in reader.get_batch(indices).asnumpy()]
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                image_features = self._model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                scores = (image_features @ text_features.T).flatten().float().cpu().tolist()
            for candidate, score in zip(batch, scores):
                item = dict(candidate)
                item["score"] = float(score)
                scored.append(item)
        return sorted(scored, key=lambda x: x["score"], reverse=True)
