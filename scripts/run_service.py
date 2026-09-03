#!/usr/bin/env python
"""Run the restricted local service.

The default bind address is loopback. Use ``--backend stub`` for contract
smoke tests; ``--backend coarse_to_fine`` runs the new cache/retrieval/
grounding pipeline; ``--backend videoitg`` is retained as a baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_app(args):
    from videoitg_smart_clip.service.app import ServiceSettings, create_app
    from videoitg_smart_clip.service.runtime import BoundedTaskManager, CoarseToFineWorker, StubWorker, VideoITGWorker
    from videoitg_smart_clip.grounding import TimeLensGrounder
    from videoitg_smart_clip.pipeline.postprocess import BoundaryRefiner, CandidateRanker, NoMatchDecider, TemporalDeduplicator

    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    if not devices:
        raise SystemExit("--devices must contain at least one device")
    if args.backend == "videoitg" and not args.videoitg_model:
        raise SystemExit("--videoitg-model is required for --backend videoitg")
    if args.backend == "stub":
        workers = [StubWorker(delay_s=args.stub_delay_s) for _ in devices]
        model_version = "stub-v1"
    elif args.backend == "coarse_to_fine":
        feature_encoders = []
        if args.feature_model_path:
            from videoitg_smart_clip.preprocessing import SigLIPFeatureEncoder

            feature_encoders = [
                SigLIPFeatureEncoder(args.feature_model_path, device=device, batch_size=args.feature_batch_size)
                for device in devices
            ]
        grounders = []
        if args.timelens_model_path:
            grounders = [TimeLensGrounder(args.timelens_model_path, device=device, batch_size=args.timelens_batch_size, max_new_tokens=args.timelens_max_new_tokens, total_pixels=args.timelens_total_pixels) for device in devices]
        workers = [
            CoarseToFineWorker(
                args.feature_root,
                feature_encoder=(feature_encoders[index] if feature_encoders else None),
                grounder=(grounders[index] if grounders else None),
                boundary_refiner=BoundaryRefiner(
                    enabled=args.boundary_refinement_enabled,
                    expansion_seconds=args.boundary_expansion_seconds,
                    start_offset_seconds=args.boundary_start_offset_seconds,
                    end_offset_seconds=args.boundary_end_offset_seconds,
                    start_padding_seconds=args.boundary_start_padding_seconds,
                    end_padding_seconds=args.boundary_end_padding_seconds,
                ),
                ranker=CandidateRanker(weights={
                    "retrieval": args.ranking_retrieval_weight,
                    "grounding": args.ranking_grounding_weight,
                    "boundary": args.ranking_boundary_weight,
                    "completeness": args.ranking_completeness_weight,
                    "duplication": args.ranking_duplication_weight,
                }),
                deduplicator=TemporalDeduplicator(temporal_iou_threshold=args.dedup_temporal_iou_threshold),
                no_match=NoMatchDecider(
                    retrieval_threshold=args.no_match_retrieval_threshold,
                    grounding_threshold=args.no_match_grounding_threshold,
                    margin_threshold=args.no_match_margin_threshold,
                ),
                top_n=args.top_n,
                top_k=args.top_k,
                feature_sample_fps=args.feature_sample_fps,
                feature_max_frames=args.feature_max_frames,
            )
            for index, _ in enumerate(devices)
        ]
        model_version = workers[0].model_version
    else:
        workers = [
            VideoITGWorker(
                Path(args.videoitg_model),
                device,
                target_fps=2.0,
                max_frames=16,
            )
            for device in devices
        ]
        model_version = Path(args.videoitg_model).name
    root_values = list(args.allowed_video_root or [])
    upload_root_value = str(args.upload_root)
    if upload_root_value not in root_values:
        root_values.append(upload_root_value)
    roots = tuple(Path(item).expanduser() for item in root_values)
    settings = ServiceSettings(
        backend_name=args.backend,
        upload_root=Path(args.upload_root),
        allowed_video_roots=roots,
        max_upload_bytes=args.max_upload_bytes,
        max_video_duration_seconds=args.max_video_duration_seconds,
        deadline_ms=args.deadline_ms,
        async_timeout_ms=args.async_timeout_ms,
        policy_version="E0_fixed16",
        service_version="phase6-restricted-v1",
    )
    manager = BoundedTaskManager(
        workers,
        queue_size=args.queue_size,
        policy_version=settings.policy_version,
        estimated_service_ms=args.estimated_service_ms,
        state_path=args.state_path,
        task_timeout_s=args.async_timeout_ms / 1000.0,
    )
    app = create_app(manager, settings)
    app.state.configured_model_version = model_version
    if args.warmup_video:
        warmup_path = Path(args.warmup_video).expanduser().resolve(strict=True)
        for worker in workers:
            worker.run(warmup_path, args.warmup_query)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["stub", "coarse_to_fine", "videoitg"], default="coarse_to_fine")
    parser.add_argument("--videoitg-model", type=Path)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--deadline-ms", type=int, default=5000)
    parser.add_argument("--async-timeout-ms", type=int, default=120000)
    parser.add_argument("--estimated-service-ms", type=int, default=2000)
    parser.add_argument("--max-upload-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--max-video-duration-seconds", type=float, default=1800.0)
    parser.add_argument("--upload-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/raw/service_uploads"))
    parser.add_argument("--allowed-video-root", action="append")
    parser.add_argument("--stub-delay-s", type=float, default=0.01)
    parser.add_argument("--state-path", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/runtime/service_tasks.jsonl"))
    parser.add_argument("--feature-root", type=Path, default=Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/features"))
    parser.add_argument("--feature-model-path", type=Path, help="local SigLIP checkpoint; omit for deterministic smoke encoder")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--feature-sample-fps", type=float, default=1.0)
    parser.add_argument("--feature-max-frames", type=int, default=16)
    parser.add_argument("--timelens-model-path", type=Path, help="local TimeLens-8B checkpoint; omit to keep explicit Level-2 fallback")
    parser.add_argument("--timelens-batch-size", type=int, default=1)
    parser.add_argument("--timelens-max-new-tokens", type=int, default=128)
    parser.add_argument("--timelens-total-pixels", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--boundary-refinement-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary-expansion-seconds", type=float, default=0.0)
    parser.add_argument("--boundary-start-offset-seconds", type=float, default=0.0)
    parser.add_argument("--boundary-end-offset-seconds", type=float, default=0.0)
    parser.add_argument("--boundary-start-padding-seconds", type=float, default=0.0)
    parser.add_argument("--boundary-end-padding-seconds", type=float, default=0.0)
    parser.add_argument("--ranking-retrieval-weight", type=float, default=0.30)
    parser.add_argument("--ranking-grounding-weight", type=float, default=0.40)
    parser.add_argument("--ranking-boundary-weight", type=float, default=0.10)
    parser.add_argument("--ranking-completeness-weight", type=float, default=0.20)
    parser.add_argument("--ranking-duplication-weight", type=float, default=0.0)
    parser.add_argument("--dedup-temporal-iou-threshold", type=float, default=0.7)
    parser.add_argument("--no-match-retrieval-threshold", type=float, default=None)
    parser.add_argument("--no-match-grounding-threshold", type=float, default=None)
    parser.add_argument("--no-match-margin-threshold", type=float, default=None)
    parser.add_argument("--warmup-video", type=Path)
    parser.add_argument("--warmup-query", default="a video segment matching the query")
    args = parser.parse_args()
    if args.queue_size <= 0 or args.deadline_ms <= 0 or args.async_timeout_ms <= 0 or args.max_upload_bytes <= 0 or args.max_video_duration_seconds <= 0 or args.timelens_batch_size <= 0 or args.timelens_max_new_tokens <= 0 or args.timelens_total_pixels <= 0 or args.top_n <= 0 or args.top_k <= 0 or args.feature_sample_fps <= 0 or args.feature_max_frames <= 0 or args.dedup_temporal_iou_threshold < 0 or args.dedup_temporal_iou_threshold > 1:
        raise SystemExit("queue size, deadlines, upload size, video duration, TimeLens parameters, top-n and top-k must be positive")

    import uvicorn

    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
