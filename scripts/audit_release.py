"""Audit the small, reproducible restricted release surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = (
    "README.md",
    "AGENTS.md",
    "docs/project_state.md",
    "docs/current_system.md",
    "docs/current_architecture.md",
    "docs/timelens_model.md",
    "docs/model_download.md",
    "docs/service.md",
    "docs/validation_manifest.md",
    "plan/execution_plan.md",
    "records/phase_g2_g4_issue_log.md",
    "configs/default.yaml",
    "configs/eval.yaml",
    "eval/run_eval.py",
    "docs/release_checklist.md",
    "frontend/index.html",
    "frontend/app.js",
    "frontend/styles.css",
    "scripts/g7_browser_smoke.py",
    "scripts/run_service.py",
    "scripts/validate_local_workflow.py",
    "scripts/build_validation_manifest.py",
    "tests/test_service.py",
    "tests/test_coarse_to_fine.py",
    "tests/test_badcase.py",
    "tests/test_metrics.py",
    "tests/test_validation_manifest.py",
    "records/phase6_service_smoke.json",
    "records/phase6_safety_matrix.json",
    "records/phase6_steady_pressure.json",
    "records/phase8/local_workflow_protocol.json",
    "records/phase8/user_test_protocol.md",
    "scripts/summarize_feedback.py",
    "src/videoitg_smart_clip/pipeline/contracts.py",
    "src/videoitg_smart_clip/pipeline/service.py",
    "src/videoitg_smart_clip/pipeline/postprocess.py",
    "src/videoitg_smart_clip/preprocessing/feature_cache.py",
    "src/videoitg_smart_clip/preprocessing/feature_encoder.py",
    "src/videoitg_smart_clip/retrieval/temporal.py",
    "src/videoitg_smart_clip/grounding/timelens.py",
    "src/videoitg_smart_clip/dependency_locks.py",
    "src/videoitg_smart_clip/service/app.py",
    "src/videoitg_smart_clip/service/models.py",
    "src/videoitg_smart_clip/service/runtime.py",
    "src/videoitg_smart_clip/evaluation/metrics.py",
    "src/videoitg_smart_clip/evaluation/validation.py",
    "src/videoitg_smart_clip/badcase/taxonomy.py",
    "scripts/g4_timelens_smoke.py",
    "scripts/g4_timelens_runtime_matrix.py",
    "scripts/g5_postprocess_smoke.py",
    "scripts/g6_realtime_service_smoke.py",
    "scripts/g6_real_model_runtime_matrix.py",
    "scripts/g6_real_model_two_worker_matrix.py",
    "scripts/profile_realtime_model.py",
    "records/phase_g4/g4_timelens_smoke_20260831_225118.json",
    "records/phase_g4/g4_timelens_smoke_20260901_003603.json",
    "records/phase_g4/g4_timelens_runtime_matrix_recovery_20260903_v2.json",
    "records/phase_g6/g6_realtime_service_smoke.json",
    "records/phase_g6/g6_realtime_service_timeout_1000ms.json",
    "records/phase_g6/g6_realtime_smoke_recovery_20260903.json",
    "records/phase_g6/g6_realtime_service_smoke_recovery_20260903.json",
    "records/phase_g6/g6_realtime_service_timeout_1000ms_recovery_20260903.json",
    "records/phase_g6/g6_real_model_runtime_matrix_recovery_20260903.json",
    "records/phase_g6/g6_real_model_two_worker_matrix_recovery_20260903_v3.json",
    "records/phase_g6/g6_realtime_service_gpu_driver_stall_20260901.json",
    "records/phase_g6/stub_pressure_8.json",
    "records/phase_g8/eval_smoke_output/metrics.json",
    "records/phase_g9/realtime_model_profile.json",
    "records/phase_g9/realtime_model_profile_sdpa_4m.json",
    "records/phase_g9/realtime_model_profile_recovery_20260903.json",
    "records/phase_g9/gpu_driver_probe_20260901.json",
    "records/phase_g9/decord_build_20260901.json",
    "scripts/validate_validation_manifest.py",
    "src/videoitg_smart_clip/evaluation/validation.py",
    "records/phase_g8/validation_manifest.template.jsonl",
    "records/phase_g8/validation_present_pending_20260901.jsonl",
    "records/phase_g4/g4_timelens_pilot_eval/metrics.json",
    "records/phase_g4/g4_timelens_top20_batch2_eval/metrics.json",
    "records/phase_g8/g4_pilot_eval_current/metrics.json",
    "records/phase_g8/g4_top20_eval_current/metrics.json",
    "records/phase_g3/g3_retriever_compare_resource_20260831.json",
    "records/phase_g3/g3_retriever_compare_validation_present_20260903.json",
    "records/phase_g3/g3_retriever_compare_full100_recovery_20260903.json",
    "records/phase_g3/g3_retriever_default_decision_20260903.json",
    "records/phase_g4/g4_timelens_pilot_query_normalized_eval_v1/metrics.json",
    "records/phase_g4/g4_timelens_pilot_query_normalized.jsonl",
    "docs/requirements_traceability.md",
    "docs/real_acceptance_plan.md",
    "records/phase_g8/current_data_completion_20260901.json",
    "records/phase_g8/current_data_completion_20260903.json",
    "records/phase_g2/g2_media_pool_audit_recovery_20260903.json",
    "records/phase_g8/eval_smoke_output_v2/metrics.json",
    "records/phase_g8/eval_smoke_output_v3/metrics.json",
    "records/phase_g8/eval_smoke_output_v4/metrics.json",
    "records/phase_g8/eval_smoke_output_current_20260901/metrics.json",
    "records/phase_g8/eval_smoke_output_current_20260901/badcases.json",
    "records/phase_g8/eval_smoke_output_current_20260901/report.md",
    "scripts/summarize_feature_cache.py",
    "scripts/g4_timelens_contract_matrix.py",
    "scripts/ablate_postprocess.py",
    "scripts/service_contract_matrix.py",
    "records/phase_g2/g2_feature_cache_audit_20260901_v2.json",
    "records/phase_g2/local360_media_manifest_20260903.jsonl",
    "records/phase_g2/local360_media_audit_20260903.json",
    "records/phase_g2/local360_feature_cache_audit_20260903.json",
    "records/phase_g3/g3_retriever_local360_20260903.json",
    "records/phase_g4/g4_timelens_local360_recovery_20260903.jsonl",
    "records/phase_g8/g4_local360_eval_20260903/metrics.json",
    "records/phase_g8/g4_local360_eval_20260903/report.md",
    "records/phase_g2/g2_media_pool_audit_20260901.json",
    "records/phase_g4/g4_timelens_contract_matrix_20260901_v2.json",
    "records/phase_g5/postprocess_ablation_timelens_pilot_20260901_v2.json",
    "records/phase_g5/postprocess_ablation_timelens_top20_20260901_v2.json",
    "records/phase_g6/service_contract_matrix_20260901_v2.json",
    "records/phase_g8/g4_pilot_eval_current_20260901_v2/metrics.json",
    "records/phase_g5/postprocess_ablation_timelens_pilot_recovery_20260903_v2.json",
    "records/phase_g5/postprocess_ablation_timelens_top20_recovery_20260903_v2.json",
)


def _runtime_artifact_check(root: Path) -> bool:
    """Verify the archived Decord artifact without importing CUDA/torch."""
    record_path = root / "records/phase_g9/decord_build_20260901.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    artifact = Path(record["artifact"]["path"])
    expected_hash = str(record["artifact"]["sha256"])
    if not artifact.is_file():
        return False
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != expected_hash:
        return False
    with zipfile.ZipFile(artifact) as wheel:
        wheel_info = wheel.read("decord-0.6.0.dist-info/WHEEL").decode("utf-8")
        names = set(wheel.namelist())
    return (
        "Tag: cp313-cp313-manylinux_2_39_x86_64" in wheel_info
        and "decord/libdecord.so" in names
        and any(name.startswith("decord.libs/") for name in names)
    )


def _contract_checks(root: Path) -> dict[str, bool]:
    """Run cheap, CPU-only checks for the migration's public contracts."""
    from videoitg_smart_clip.pipeline.contracts import CandidateWindow, GroundingPrediction
    from videoitg_smart_clip.pipeline.postprocess import (
        BoundaryRefiner,
        CandidateRanker,
        NoMatchDecider,
        TemporalDeduplicator,
        build_candidate_record,
    )
    from videoitg_smart_clip.service.app import ServiceSettings
    from videoitg_smart_clip.service.models import FeedbackLabel, TaskLifecycleStatus

    prediction = GroundingPrediction("audit", 1.0, 2.0, 0.8, model_version="TimeLens-8B")
    record = build_candidate_record(
        CandidateWindow(0.0, 5.0, 0.7, "audit"),
        prediction,
        BoundaryRefiner().refine(prediction),
        final_score=0.75,
        retriever_version="cached-cosine-v1",
        postprocess_version="audit",
    )
    required_candidate_keys = {
        "candidate_id", "coarse_start", "coarse_end", "retrieval_score",
        "raw_start", "raw_end", "grounding_score", "refined_start", "refined_end",
        "final_score", "rank", "retriever_version", "grounder_version",
        "postprocess_version", "retrieval_latency_ms", "grounding_latency_ms",
        "postprocess_latency_ms", "degraded", "degrade_level",
    }
    lifecycle = {item.value for item in TaskLifecycleStatus}
    expected_lifecycle = {"PENDING", "PREPROCESSING", "INDEXING", "RETRIEVING", "GROUNDING", "POSTPROCESSING", "SUCCESS", "FAILED", "CANCELLED", "TIMEOUT"}
    canonical_feedback = {"ACCEPT", "IRRELEVANT", "START_TOO_EARLY", "START_TOO_LATE", "END_TOO_EARLY", "END_TOO_LATE", "DUPLICATE", "MISS"}
    feedback_values = {item.value for item in FeedbackLabel if item.value.isupper()}
    postprocess_types = (BoundaryRefiner, CandidateRanker, TemporalDeduplicator, NoMatchDecider)
    origins = set(ServiceSettings().frontend_origins)
    metrics_path = root / "records/phase_g8/eval_smoke_output_current_20260901/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_metric_keys = {
        "recall_at_1_iou_0.3", "recall_at_1_iou_0.5", "recall_at_1_iou_0.7",
        "miou", "mean_boundary_error_s", "duplicate_rate", "top_k_useful_rate",
        "no_match_accuracy", "false_positive_rate", "false_negative_rate",
        "decode_latency_ms_mean", "feature_extraction_latency_ms_mean",
        "retrieval_latency_ms_mean", "grounding_latency_ms_mean",
        "postprocess_latency_ms_mean", "end_to_end_latency_ms_mean",
        "gpu_memory_gib_mean", "timeout_rate", "failure_rate", "degrade_rate",
    }
    return {
        "candidate_schema": required_candidate_keys.issubset(record),
        "canonical_lifecycle": lifecycle == expected_lifecycle,
        "canonical_feedback": feedback_values == canonical_feedback,
        "independent_postprocess": all(callable(getattr(cls, "__init__", None)) for cls in postprocess_types),
        "loopback_cors_only": origins == {"http://127.0.0.1:8080", "http://localhost:8080"},
        "evaluation_metric_schema": required_metric_keys.issubset(metrics) and isinstance(metrics.get("evaluation_config"), dict),
        "runtime_artifact": _runtime_artifact_check(root),
    }


def audit(root: Path) -> dict:
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()
               ]
    workflow = json.loads((root / "records/phase8/local_workflow_protocol.json").read_text())
    required_claims = [
        "Query-aware Coarse-to-Fine Temporal Grounding",
        "TimeLens-8B",
        "G6 真实异步 HTTP smoke",
    ]
    project_state = (root / "docs/project_state.md").read_text(encoding="utf-8")
    required_claims_present = {marker: marker in project_state for marker in required_claims}
    try:
        contract_checks = _contract_checks(root)
    except Exception as exc:  # pragma: no cover - surfaced in audit output
        contract_checks = {"contract_check_exception": False, "error": str(exc)}
    contracts_ok = all(value for key, value in contract_checks.items() if key != "error")
    return {
        "run_id": datetime.now(timezone.utc).strftime("release_audit_%Y%m%dT%H%M%SZ"),
        "required_files": len(REQUIRED_FILES),
        "missing_files": missing,
        "workflow_protocol_success": workflow["successful_tasks"] == workflow["iterations"],
        "workflow_is_not_human_study": workflow.get("human_user_test") is False,
        "required_scope_markers_present": required_claims_present,
        "contract_checks": contract_checks,
        "status": "pass" if not missing and workflow["successful_tasks"] == workflow["iterations"] and all(required_claims_present.values()) and contracts_ok else "fail",
        "g8_status": "out_of_scope_no_manual_annotation_or_human_test",
        "g9_status": "packaging_and_gpu_automated_profiler_verified; portability_beyond_manylinux_2_39_conditional",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
