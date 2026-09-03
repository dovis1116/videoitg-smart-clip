#!/usr/bin/env python3
"""Run a small fixed TimeLens quality pilot using existing cached features."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoitg_smart_clip.grounding import TimeLensGrounder
from videoitg_smart_clip.pipeline import CoarseToFinePipeline
from videoitg_smart_clip.preprocessing import FeatureCache, SigLIPFeatureEncoder
from videoitg_smart_clip.preprocessing.feature_cache import FeatureCacheKey
from videoitg_smart_clip.retrieval import CachedCosineRetriever


def find_cache(cache_root: Path, video_id: str) -> tuple[dict, Path]:
    for metadata_path in cache_root.glob("*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("video_id") == video_id:
            return metadata, metadata_path
    raise FileNotFoundError(f"no cached feature metadata for {video_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "records/phase3/videoitg100_media_pilot.jsonl")
    parser.add_argument("--cache-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/features/g3_siglip_full"))
    parser.add_argument("--feature-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"))
    parser.add_argument("--timelens-model", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "records/phase_g4/g4_timelens_pilot.jsonl")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()][: max(1, args.count)]
    encoder = SigLIPFeatureEncoder(args.feature_model, device="cuda:0", batch_size=8)
    encoder.load()
    cache = FeatureCache(args.cache_root, feature_model="google/siglip-so400m-patch14-384")
    retriever = CachedCosineRetriever(cache, query_encoder=encoder.encode_query, window_seconds=10.0)
    grounder = TimeLensGrounder(args.timelens_model, device="cuda:0", batch_size=args.batch_size)
    output_rows = []
    started_all = time.perf_counter()
    for row in rows:
        video_id = row["sample_id"]
        metadata, _ = find_cache(args.cache_root, video_id)
        key = FeatureCacheKey(video_id, metadata["feature_model_version"], metadata["preprocessing_config"], metadata["sampling_config"])
        retriever.index(row["video_path"], video_id, key=key, video_duration=float(metadata["video_duration"]), extractor=lambda: (_ for _ in ()).throw(AssertionError("pilot must reuse cached video features")))
        result = CoarseToFinePipeline(retriever, grounder, top_n=args.top_n, top_k=min(3, args.top_n)).search(row["video_path"], video_id, row["query"])
        predictions = []
        for candidate in result.get("predictions", []):
            predictions.append({"start_s": candidate["refined_start"], "end_s": candidate["refined_end"], "refined_start": candidate["refined_start"], "refined_end": candidate["refined_end"], "retrieval_score": candidate["retrieval_score"], "grounding_score": candidate["grounding_score"], "final_score": candidate["final_score"], "candidate_id": candidate["candidate_id"]})
        gt = [[float(index) * 5.0, (float(index) + 1.0) * 5.0] for index in row.get("clip_num", [])]
        output_rows.append({"sample_id": video_id, "video_path": row["video_path"], "query": row["query"], "ground_truth": gt, "coarse_windows": [{"start": window.start, "end": window.end, "score": window.score} for window in retriever.retrieve(video_id, row["query"], args.top_n)], "predictions": predictions, "candidates": result.get("candidates", []), "output_top_k": min(3, args.top_n), "status": result["status"], "actual_match": True, "degraded": result["degraded"], "elapsed_ms": sum(float(candidate.get("grounding_latency_ms", 0.0)) for candidate in result.get("predictions", []))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output_rows) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": f"g4_timelens_pilot_{time.strftime('%Y%m%d_%H%M%S')}", "sample_count": len(output_rows), "elapsed_ms": (time.perf_counter() - started_all) * 1000.0, "output": str(args.output), "scope": "fixed_target_present_pilot_using_cached_features"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
