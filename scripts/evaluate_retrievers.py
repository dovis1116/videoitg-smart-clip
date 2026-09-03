#!/usr/bin/env python3
"""Compare query-aware cached SigLIP retrieval with a uniform reference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoitg_smart_clip.evaluation.metrics import retrieval_recall, segment_iou
from videoitg_smart_clip.preprocessing import FeatureCache, SigLIPFeatureEncoder, cache_identity, decode_uniform_frames
from videoitg_smart_clip.retrieval import CachedCosineRetriever, UniformTemporalRetriever


def _ground_truth(row: dict) -> list[list[float]]:
    """Accept both the pilot ``clip_num`` and validation ``ground_truth`` schemas."""

    value = row.get("ground_truth_segments", row.get("ground_truth"))
    if value is not None:
        return [[float(segment[0]), float(segment[1])] for segment in value]
    return [[float(c) * 5.0, (float(c) + 1.0) * 5.0] for c in row.get("clip_num", [])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "records/phase3/videoitg100_media_pilot.jsonl")
    parser.add_argument("--model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"))
    parser.add_argument("--cache-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/features/g3_siglip"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=16)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    encoder = SigLIPFeatureEncoder(args.model, device="cuda:0", batch_size=8)
    encoder.load()
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    cache = FeatureCache(args.cache_root, feature_model=encoder.model_name)
    cached = CachedCosineRetriever(cache, query_encoder=encoder.encode_query, window_seconds=10.0)
    uniform = UniformTemporalRetriever(cache, window_seconds=10.0)
    details = []
    started = time.perf_counter()
    for row in rows:
        video = Path(row["video_path"])
        video_id = row.get("sample_id", video.stem).replace("/", "_")
        from decord import VideoReader, cpu

        reader = VideoReader(str(video), ctx=cpu(0), num_threads=2)
        duration = len(reader) / float(reader.get_avg_fps())
        key = cache_identity(video_id, encoder.version, {"decode": "decord", "segment": "uniform"}, {"fps": 1.0, "max_frames": args.max_frames})

        def extract(video=video):
            frames, timestamps, _ = decode_uniform_frames(video, sample_fps=1.0, max_frames=args.max_frames)
            return encoder.encode(frames), timestamps

        cached.index(video, video_id, key=key, extractor=extract, video_duration=duration)
        query = row["query"].split("\nAnswer the question", 1)[0].strip()
        windows = cached.retrieve(video_id, query, top_n=20)
        uniform.index(video, video_id, video_duration=duration)
        uniform_started = time.perf_counter()
        uniform_windows = uniform.retrieve(video_id, query, top_n=20)
        uniform_latency_ms = (time.perf_counter() - uniform_started) * 1000.0
        gt = _ground_truth(row)
        details.append({
            "sample_id": row.get("sample_id"),
            "video_id": video_id,
            "duration_s": duration,
            "cache_hit": bool(cached.last_metrics.get("cache_hit")),
            "feature_cache_size_bytes": cached.last_metrics.get("feature_cache_size"),
            "cached_cosine": {"recall_at_5": retrieval_recall([w.__dict__ for w in windows], gt, 5), "recall_at_10": retrieval_recall([w.__dict__ for w in windows], gt, 10), "recall_at_20": retrieval_recall([w.__dict__ for w in windows], gt, 20), "latency_ms": cached.last_metrics.get("retrieval_latency_ms"), "index_size_bytes": cached.last_metrics.get("feature_cache_size", 0)},
            "uniform": {"recall_at_5": retrieval_recall([w.__dict__ for w in uniform_windows], gt, 5), "recall_at_10": retrieval_recall([w.__dict__ for w in uniform_windows], gt, 10), "recall_at_20": retrieval_recall([w.__dict__ for w in uniform_windows], gt, 20), "latency_ms": uniform_latency_ms, "index_size_bytes": 0},
        })
    def mean(method: str, metric: str) -> float:
        return sum(float(item[method][metric]) for item in details) / max(1, len(details))

    result = {
        "run_id": f"g3_retriever_compare_{time.strftime('%Y%m%d_%H%M%S')}",
        "manifest": str(args.manifest),
        "sample_count": len(details),
        "actual_match_counts": {str(value): sum(1 for row in rows if row.get("actual_match") is value) for value in (True, False)},
        "feature_model_version": encoder.version,
        "max_frames": args.max_frames,
        "retrievers": {
            "cached_cosine": {"recall_at_5": mean("cached_cosine", "recall_at_5"), "recall_at_10": mean("cached_cosine", "recall_at_10"), "recall_at_20": mean("cached_cosine", "recall_at_20"), "mean_retrieval_latency_ms": sum(float(d["cached_cosine"]["latency_ms"] or 0) for d in details) / max(1, len(details)), "mean_index_size_bytes": mean("cached_cosine", "index_size_bytes")},
            "uniform": {"recall_at_5": mean("uniform", "recall_at_5"), "recall_at_10": mean("uniform", "recall_at_10"), "recall_at_20": mean("uniform", "recall_at_20"), "mean_retrieval_latency_ms": mean("uniform", "latency_ms"), "mean_index_size_bytes": mean("uniform", "index_size_bytes")},
        },
        "cache_events": cache.cache_events,
        "details": details,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "gpu_peak_memory_allocated_gib": (torch.cuda.max_memory_allocated(torch.device("cuda:0")) / 2**30 if torch.cuda.is_available() else None),
        "gpu_peak_memory_reserved_gib": (torch.cuda.max_memory_reserved(torch.device("cuda:0")) / 2**30 if torch.cuda.is_available() else None),
        "selection": "pending full validation set with negative coverage; present-only evidence is reported without formal No-Match calibration",
    }
    output = args.output or ROOT / "records/phase_g3" / f"{result['run_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
