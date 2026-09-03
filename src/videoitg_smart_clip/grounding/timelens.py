"""TimeLens grounding adapter and explicit local-window timestamp mapping."""

from __future__ import annotations

import time
import re
import shutil
import subprocess
import tempfile
import importlib.util
from pathlib import Path
from typing import Sequence

from videoitg_smart_clip.pipeline.contracts import CandidateWindow, GroundingPrediction


class StubTimeLensGrounder:
    model_version = "TimeLens-8B-stub-v1"

    def predict(self, video_path: str | Path, query: str, candidate_windows: Sequence[CandidateWindow]) -> list[GroundingPrediction]:
        started = time.perf_counter()
        return [GroundingPrediction(w.candidate_id, w.start, w.end, w.score, (time.perf_counter() - started) * 1000.0, self.model_version, 0.5, 0.5) for w in candidate_windows]


class TimeLensGrounder:
    """Run the official TimeLens generation recipe on bounded video windows.

    TimeLens emits natural-language timestamps relative to the supplied clip.
    This adapter materializes each candidate window, parses the timestamps, and
    maps them back to the original video's absolute timeline.  No full-video
    input is ever sent to the 8B model.
    """

    model_version = "TimeLens-8B-pending"
    _TIMESTAMP = re.compile(r"(?P<start>\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(?P<end>\d+(?:\.\d+)?)\s*(?:seconds?|s)?", re.I)
    _PROMPT = "Please find the visual event described by the sentence '{}', determining its starting and ending times. The format should be: 'The event happens in <start time> - <end time> seconds'."

    def __init__(self, model_path: str | Path | None = None, *, device: str = "cuda:0", max_new_tokens: int = 128, total_pixels: int = 4 * 1024 * 1024, batch_size: int = 1) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.model_version = "TimeLens-8B" if self.model_path else "TimeLens-8B-pending"
        self.max_new_tokens = max_new_tokens
        self.total_pixels = total_pixels
        self.batch_size = max(1, int(batch_size))
        self.attention_implementation = "pending"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.model_path is None:
            raise RuntimeError("TimeLens-8B model_path is required")
        from videoitg_smart_clip.dependency_locks import TRANSFORMERS_IMPORT_LOCK

        with TRANSFORMERS_IMPORT_LOCK:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, padding_side="left", do_resize=False)
            kwargs = {"local_files_only": True, "dtype": torch.bfloat16 if self.device.startswith("cuda") else torch.float32, "low_cpu_mem_usage": True}
            if self.device.startswith("cuda"):
                # Keep local deployment runnable when flash-attn is not installed;
                # PyTorch SDPA is the lower-memory fallback and preserves the same
                # model contract (eager attention is a last-resort option).
                self.attention_implementation = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
                kwargs.update({"attn_implementation": self.attention_implementation, "device_map": self.device})
            else:
                self.attention_implementation = "eager"
            self._model = AutoModelForImageTextToText.from_pretrained(self.model_path, **kwargs).eval()

    @staticmethod
    def _ffmpeg() -> str:
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            try:
                import static_ffmpeg

                static_ffmpeg.add_paths()
                executable = shutil.which("ffmpeg")
                if executable:
                    return executable
            except Exception:
                pass
        raise RuntimeError("ffmpeg is required to materialize candidate windows")

    def _materialize_window(self, video_path: str | Path, window: CandidateWindow, directory: Path) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", window.candidate_id).strip("._") or "candidate"
        output = directory / f"{safe_id}.mp4"
        duration = max(0.1, window.end - window.start)
        # libx264/yuv420p requires even dimensions.  Existing source videos
        # include odd-sized frames (for example 500x375); normalize only the
        # temporary candidate clip so those rows do not silently fall back to
        # coarse retrieval because window materialization failed.
        command = [self._ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(window.start), "-i", str(video_path), "-t", str(duration), "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(output)]
        subprocess.run(command, check=True)
        return output

    @classmethod
    def parse_timestamps(cls, answer: str, window_duration: float) -> tuple[float, float]:
        match = cls._TIMESTAMP.search(answer)
        if match is None:
            raise ValueError(f"TimeLens output did not contain a timestamp range: {answer!r}")
        start = min(max(float(match.group("start")), 0.0), window_duration)
        end = min(max(float(match.group("end")), 0.0), window_duration)
        if end <= start:
            raise ValueError(f"TimeLens output has invalid timestamp range: {answer!r}")
        return start, end

    def _infer_window(self, clip_path: Path, query: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [{"type": "video", "video": str(clip_path), "min_pixels": 64 * 32 * 32, "total_pixels": self.total_pixels, "fps": 2}, {"type": "text", "text": self._PROMPT.format(query)}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        video_metadatas = None
        if videos:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        inputs = self._processor(text=[text], images=images, videos=videos, video_metadata=video_metadatas, padding=True, return_tensors="pt", **video_kwargs)
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _infer_batch(self, clips: Sequence[Path], query: str) -> list[str]:
        """Run one generation call for several materialized candidate clips."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [{"type": "video", "video": str(clip), "min_pixels": 64 * 32 * 32, "total_pixels": self.total_pixels, "fps": 2}, {"type": "text", "text": self._PROMPT.format(query)}]} for clip in clips]
        # ``apply_chat_template`` returns a single serialized conversation when
        # passed a list of messages.  Build one serialized conversation per
        # candidate so the processor creates a true batch rather than a single
        # request containing multiple videos.
        text = [
            self._processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        video_metadatas = None
        if videos:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        inputs = self._processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, padding=True, return_tensors="pt", **video_kwargs).to(self.device)
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def predict(self, video_path: str | Path, query: str, candidate_windows: Sequence[CandidateWindow]) -> list[GroundingPrediction]:
        self._load()
        started_all = time.perf_counter()
        predictions = []
        with tempfile.TemporaryDirectory(prefix="timelens_windows-") as temporary:
            directory = Path(temporary)
            windows = list(candidate_windows)
            for offset in range(0, len(windows), self.batch_size):
                group = windows[offset : offset + self.batch_size]
                clips = [self._materialize_window(video_path, window, directory) for window in group]
                started = time.perf_counter()
                answers = self._infer_batch(clips, query) if len(group) > 1 else [self._infer_window(clips[0], query)]
                batch_latency = (time.perf_counter() - started) * 1000.0
                if len(answers) != len(group):
                    raise RuntimeError(f"TimeLens returned {len(answers)} answers for {len(group)} candidate windows")
                for window, answer in zip(group, answers):
                    local_start, local_end = self.parse_timestamps(answer, window.end - window.start)
                    predictions.append(GroundingPrediction(window.candidate_id, window.start + local_start, window.start + local_end, 1.0, batch_latency / len(group), self.model_version, 1.0, 1.0, {"raw_answer": answer, "window_start": window.start, "window_end": window.end, "batch": len(group) > 1, "batch_size": len(group)}))
        return predictions
