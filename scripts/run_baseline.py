#!/usr/bin/env python
"""Run B0/B1/B2 on the target-present pilot with shared evaluation rules."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

def read_manifest(path: Path, limit: int | None, sample_ids: list[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("raw_video_present", True) and Path(row["video_path"]).is_file()]
    if sample_ids:
        wanted = set(sample_ids)
        rows = [row for row in rows if row.get("sample_id") in wanted]
    return rows[:limit] if limit else rows


def initialize_device(device: str) -> None:
    """Initialize CUDA before Decord; this avoids a known Decord/Torch lazy-init crash."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.set_device(device)


def b0(rows, args):
    from videoitg_smart_clip.baselines.common import uniform_candidates
    from videoitg_smart_clip.baselines.clip_retriever import ClipRetriever

    retriever = ClipRetriever(args.clip_model, args.device, args.clip_batch_size)
    outputs = []
    for row in rows:
        started = time.perf_counter()
        candidates = uniform_candidates(row["video_path"], args.segment_seconds, args.max_candidates)
        scored = retriever.rank(row["query"], row["video_path"], candidates)
        predictions = [
            {"candidate_id": c["candidate_id"], "start_s": c["start_s"], "end_s": c["end_s"], "score": c["score"]}
            for c in scored[: args.output_top_k]
        ]
        outputs.append(make_row(row, predictions, "B0", {"retrieval_seconds": time.perf_counter() - started, "candidate_count": len(candidates)}))
        print(f"B0 {len(outputs)}/{len(rows)} {row['sample_id']}")
    return outputs


def b2_retrieval_only(rows, args):
    """Keep the B2 coarse retrieval stage and expose all retrieved candidates."""
    from videoitg_smart_clip.baselines.common import uniform_candidates
    from videoitg_smart_clip.baselines.clip_retriever import ClipRetriever

    retriever = ClipRetriever(args.clip_model, args.device, args.clip_batch_size)
    outputs = []
    for row in rows:
        started = time.perf_counter()
        candidates = uniform_candidates(row["video_path"], args.segment_seconds, args.max_candidates)
        scored = retriever.rank(row["query"], row["video_path"], candidates)
        predictions = [
            {"candidate_id": c["candidate_id"], "start_s": c["start_s"], "end_s": c["end_s"], "score": c["score"]}
            for c in scored[: args.retrieval_top_n]
        ]
        outputs.append(make_row(row, predictions, "B2-Retrieval-only", {"retrieval_seconds": time.perf_counter() - started, "candidate_count": len(candidates)}, output_top_k=args.retrieval_top_n))
        print(f"B2R {len(outputs)}/{len(rows)} {row['sample_id']}")
    return outputs


def b1(rows, args):
    from videoitg_smart_clip.reranker import CandidateSegment, VideoITGReranker

    reranker = VideoITGReranker(args.videoitg_model, device=args.device, target_fps=args.target_fps, max_frames_per_candidate=args.max_frames, frame_score_topk=args.frame_score_topk)
    outputs = []
    for row in rows:
        started = time.perf_counter()
        full = CandidateSegment(row["video_path"], 0.0, None, candidate_id="full")
        scored = reranker.rank(row["query"], [full])[0]
        fps = float(scored.runtime["fps"])
        duration = float(scored.runtime["duration_s"])
        predictions = frame_score_predictions(scored.frame_scores, fps, duration, args.segment_seconds, args.output_top_k)
        runtime = dict(scored.runtime)
        runtime.update({"rerank_seconds": time.perf_counter() - started, "sampled_frames": len(scored.sampled_frame_indices)})
        result = make_row(row, predictions, "B1", runtime)
        if args.include_frame_scores:
            result["frame_scores"] = scored.frame_scores
            result["fps"] = fps
            result["duration_s"] = duration
        outputs.append(result)
        print(f"B1 {len(outputs)}/{len(rows)} {row['sample_id']}")
    return outputs


def b1_adaptive(rows, args):
    """Use 8-second intervals only when the 5-second top-1 score is low."""
    from videoitg_smart_clip.reranker import CandidateSegment, VideoITGReranker

    reranker = VideoITGReranker(args.videoitg_model, device=args.device, target_fps=args.target_fps, max_frames_per_candidate=args.max_frames, frame_score_topk=args.frame_score_topk)
    outputs = []
    for row in rows:
        started = time.perf_counter()
        full = CandidateSegment(row["video_path"], 0.0, None, candidate_id="full")
        scored = reranker.rank(row["query"], [full])[0]
        fps = float(scored.runtime["fps"])
        duration = float(scored.runtime["duration_s"])
        narrow = frame_score_predictions(scored.frame_scores, fps, duration, args.segment_seconds, args.output_top_k)
        wide = frame_score_predictions(scored.frame_scores, fps, duration, args.adaptive_wide_seconds, args.output_top_k)
        top1_score = float(narrow[0]["score"]) if narrow else float("-inf")
        use_wide = top1_score < args.adaptive_threshold
        predictions = wide if use_wide else narrow
        runtime = dict(scored.runtime)
        runtime.update({"rerank_seconds": time.perf_counter() - started, "sampled_frames": len(scored.sampled_frame_indices), "adaptive_threshold": args.adaptive_threshold, "adaptive_wide_seconds": args.adaptive_wide_seconds, "narrow_top1_score": top1_score, "selected_variant": "wide" if use_wide else "narrow"})
        outputs.append(make_row(row, predictions, "B1A", runtime))
        print(f"B1A {len(outputs)}/{len(rows)} {row['sample_id']} selected={'wide' if use_wide else 'narrow'}")
    return outputs


def b1_maxframes_adaptive(rows, args):
    """Use 32-frame scoring, widening output intervals only for low-confidence rows."""
    from videoitg_smart_clip.reranker import CandidateSegment, VideoITGReranker

    reranker = VideoITGReranker(args.videoitg_model, device=args.device, target_fps=args.target_fps, max_frames_per_candidate=args.max_frames, frame_score_topk=args.frame_score_topk)
    outputs = []
    for row in rows:
        started = time.perf_counter()
        full = CandidateSegment(row["video_path"], 0.0, None, candidate_id="full")
        scored = reranker.rank(row["query"], [full])[0]
        fps = float(scored.runtime["fps"])
        duration = float(scored.runtime["duration_s"])
        narrow = frame_score_predictions(scored.frame_scores, fps, duration, args.segment_seconds, args.output_top_k)
        wide = frame_score_predictions(scored.frame_scores, fps, duration, args.adaptive_wide_seconds, args.output_top_k)
        top1_score = float(narrow[0]["score"]) if narrow else float("-inf")
        use_wide = top1_score < args.adaptive_threshold
        predictions = wide if use_wide else narrow
        runtime = dict(scored.runtime)
        runtime.update({"rerank_seconds": time.perf_counter() - started, "sampled_frames": len(scored.sampled_frame_indices), "adaptive_threshold": args.adaptive_threshold, "adaptive_wide_seconds": args.adaptive_wide_seconds, "narrow_top1_score": top1_score, "selected_variant": "wide" if use_wide else "narrow"})
        outputs.append(make_row(row, predictions, "B1D", runtime))
        print(f"B1D {len(outputs)}/{len(rows)} {row['sample_id']} selected={'wide' if use_wide else 'narrow'}")
    return outputs


def b1_local_zoom(rows, args):
    """Refine a full-video peak with a second, denser local VideoITG call."""
    from videoitg_smart_clip.reranker import CandidateSegment, VideoITGReranker

    reranker = VideoITGReranker(args.videoitg_model, device=args.device, target_fps=args.target_fps, max_frames_per_candidate=args.max_frames, frame_score_topk=args.frame_score_topk)
    outputs = []
    for row in rows:
        started = time.perf_counter()
        full = CandidateSegment(row["video_path"], 0.0, None, candidate_id="full")
        global_scored = reranker.rank(row["query"], [full])[0]
        fps = float(global_scored.runtime["fps"])
        duration = float(global_scored.runtime["duration_s"])
        global_scores = list(global_scored.frame_scores)
        global_top = global_scores[0] if global_scores else None
        if global_top is None:
            predictions = []
            runtime = dict(global_scored.runtime)
            runtime.update({"rerank_seconds": time.perf_counter() - started, "sampled_frames": len(global_scored.sampled_frame_indices), "zoom_used": False})
        else:
            center = float(global_top["frame_index"]) / fps
            half = args.zoom_window_seconds / 2.0
            zoom_start = max(0.0, center - half)
            zoom_end = min(duration, zoom_start + args.zoom_window_seconds)
            zoom = CandidateSegment(row["video_path"], zoom_start, zoom_end, candidate_id="zoom")
            local_scored = reranker.rank(row["query"], [zoom])[0]
            # Keep the full-video evidence and add local dense evidence.  The
            # same frame id is scored only once, using the local observation.
            merged = {int(item["frame_index"]): item for item in global_scores}
            merged.update({int(item["frame_index"]): item for item in local_scored.frame_scores})
            merged_scores = sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True)
            predictions = frame_score_predictions(merged_scores, fps, duration, args.segment_seconds, args.output_top_k)
            runtime = dict(global_scored.runtime)
            runtime.update({
                "rerank_seconds": time.perf_counter() - started,
                "sampled_frames": len(global_scored.sampled_frame_indices) + len(local_scored.sampled_frame_indices),
                "zoom_used": True,
                "zoom_window_seconds": args.zoom_window_seconds,
                "zoom_start_s": zoom_start,
                "zoom_end_s": zoom_end,
                "global_top1_score": float(global_top["score"]),
                "global_sampled_frames": len(global_scored.sampled_frame_indices),
                "zoom_sampled_frames": len(local_scored.sampled_frame_indices),
            })
        outputs.append(make_row(row, predictions, "B1E", runtime))
        print(f"B1E {len(outputs)}/{len(rows)} {row['sample_id']}")
    return outputs


def b2(rows, args):
    from videoitg_smart_clip.baselines.common import uniform_candidates
    from videoitg_smart_clip.baselines.clip_retriever import ClipRetriever
    from videoitg_smart_clip.reranker import CandidateSegment, VideoITGReranker

    # Retrieval is completed and released before loading VideoITG to keep peak memory explicit.
    retriever = ClipRetriever(args.clip_model, args.device, args.clip_batch_size)
    retrieved = []
    for row in rows:
        candidates = uniform_candidates(row["video_path"], args.segment_seconds, args.max_candidates)
        scored = retriever.rank(row["query"], row["video_path"], candidates)
        retrieved.append((row, scored[: args.retrieval_top_n]))
        print(f"B2 retrieval {len(retrieved)}/{len(rows)} {row['sample_id']}")
    del retriever
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    reranker = VideoITGReranker(args.videoitg_model, device=args.device, target_fps=args.target_fps, max_frames_per_candidate=args.max_frames, frame_score_topk=args.frame_score_topk)
    outputs = []
    for row, candidates in retrieved:
        started = time.perf_counter()
        segments = [CandidateSegment(row["video_path"], c["start_s"], c["end_s"], c["candidate_id"]) for c in candidates]
        scored = reranker.rank(row["query"], segments)
        predictions = [
            {"candidate_id": s.candidate.candidate_id, "start_s": s.candidate.start_s, "end_s": s.candidate.end_s, "score": s.segment_score}
            for s in scored[: args.output_top_k]
        ]
        runtime = {"rerank_seconds": time.perf_counter() - started, "retrieval_candidate_count": len(candidates), "sampled_frames": sum(int(s.runtime.get("sampled_frames", 0)) for s in scored)}
        outputs.append(make_row(row, predictions, "B2", runtime))
        print(f"B2 rerank {len(outputs)}/{len(rows)} {row['sample_id']}")
    return outputs


def frame_score_predictions(frame_scores, fps, duration, segment_seconds, top_k):
    selected = []
    for item in frame_scores:
        center = float(item["frame_index"]) / fps
        start = max(0.0, center - segment_seconds / 2)
        end = min(duration, start + segment_seconds)
        if end <= start:
            continue
        candidate = {"candidate_id": f"frame_{int(item['frame_index'])}", "start_s": start, "end_s": end, "score": float(item["score"])}
        if any(max(start, x["start_s"]) < min(end, x["end_s"]) for x in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def make_row(row, predictions, baseline, runtime, output_top_k=3):
    from videoitg_smart_clip.evaluation.metrics import evaluate_sample, gt_segments_from_clip_num

    gt = gt_segments_from_clip_num(row["clip_num"])
    metrics = evaluate_sample(predictions, gt, output_top_k=output_top_k)
    return {
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "split": row.get("split", "pilot"),
        "query": row["query"],
        "baseline": baseline,
        "ground_truth_segments": gt,
        "predictions": predictions,
        "metrics": metrics,
        "runtime": runtime,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["B0", "B1", "B1A", "B1D", "B1E", "B2", "B2R"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--clip-model", default="/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/clip-vit-base-patch32")
    parser.add_argument("--videoitg-model", default="/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/VideoITG-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--frame-score-topk", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--retrieval-top-n", type=int, default=4)
    parser.add_argument("--output-top-k", type=int, default=3)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", help="run only the listed manifest sample_id; repeatable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive-threshold", type=float, default=0.6)
    parser.add_argument("--adaptive-wide-seconds", type=float, default=8.0)
    parser.add_argument("--zoom-window-seconds", type=float, default=10.0)
    parser.add_argument("--include-frame-scores", action="store_true", help="include all VideoITG frame scores in B1 output for offline temporal post-processing")
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("refusing to overwrite an existing baseline artifact")
    initialize_device(args.device)
    rows = read_manifest(args.manifest, args.limit, args.sample_id)
    if not rows:
        raise RuntimeError("manifest has no existing media rows")
    from videoitg_smart_clip.baselines.common import run_metadata
    from videoitg_smart_clip.evaluation.metrics import aggregate_metrics

    model_paths = {"clip": args.clip_model, "videoitg": args.videoitg_model}
    meta = run_metadata(model_paths, args.baseline, str(args.manifest), args.limit, args.seed)
    meta["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.baseline == "B0":
        results = b0(rows, args)
    elif args.baseline == "B1":
        results = b1(rows, args)
    elif args.baseline == "B1A":
        results = b1_adaptive(rows, args)
    elif args.baseline == "B1D":
        results = b1_maxframes_adaptive(rows, args)
    elif args.baseline == "B1E":
        results = b1_local_zoom(rows, args)
    elif args.baseline == "B2R":
        results = b2_retrieval_only(rows, args)
    else:
        results = b2(rows, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    summary = {"run": meta, "metrics": aggregate_metrics([x["metrics"] for x in results]), "output": str(args.output)}
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
