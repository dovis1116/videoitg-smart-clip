# New-route requirements traceability

更新时间：2026-09-03。状态只依据当前代码、测试和外置运行记录；`pending`
表示仍计划补证据，`out_of_scope` 表示当前明确不执行该类验收。

| Requirement | Implementation / evidence | Status |
|---|---|---|
| Query-aware Coarse-to-Fine route; TimeLens primary; VideoITG baseline | `docs/current_architecture.md`, `pipeline/service.py`, `grounding/timelens.py` | contract verified |
| Feature Cache four-field identity, sampling budget, hit/miss and concurrent first-writer safety | `preprocessing/feature_cache.py`, `service/runtime.py`, `records/phase_g2/g2_siglip_cache_20260831_222049.json` (`extractor_calls=1`), `records/phase_g2/g2_feature_cache_audit_full100_recovery_20260903.json`, service/cache regressions | two-Query no-reencode, per-query cache-hit, cross-instance file-lock, 100-row cache coverage/size and new extraction latency verified; scope remains target-present |
| Independent Retriever interface and two implementations | `pipeline/contracts.py`, `retrieval/temporal.py` | verified |
| Retriever Recall@5/10/20, latency, GPU memory, cache/index size | `records/phase_g3/g3_retriever_compare_full100_recovery_20260903.json`, `records/phase_g3/g3_retriever_default_decision_20260903.json` | 100-row target-present comparison verified; CachedCosine remains query-aware main default, Uniform is present-only reference best; difficult/negative validation unavailable |
| TimeLens candidate-window input, parser, absolute-time mapping, batch | `grounding/timelens.py`, `scripts/g4_timelens_runtime_matrix.py`, `records/phase_g4/g4_timelens_runtime_matrix_recovery_20260903_v2.json` | 9-case real serial/batch/window/budget matrix and decode failure handling verified; high-budget OOM not reproduced; quality remains diagnostic |
| Boundary raw/refined separation, independent toggle and directional offsets/padding | `pipeline/postprocess.py`, `configs/default.yaml`, `scripts/run_service.py`, `records/phase_g5/g5_postprocess_smoke_20260831_2259.json`, `records/phase_g5/postprocess_ablation_timelens_pilot_recovery_20260903_v2.json`, `records/phase_g5/postprocess_ablation_timelens_top20_recovery_20260903_v2.json` | runtime configuration and existing-prediction replay verified; full labeled quality ablation out_of_scope |
| Multi-score Ranking including boundary and duplication input | `CandidateRanker`, `pipeline/service.py`, `scripts/run_service.py`, candidate-schema tests | interface/runtime injection verified; validation weight selection pending |
| Temporal IoU Dedup and Duplicate Rate | `TemporalDeduplicator`, `evaluation/metrics.py`, pipeline diagnostic candidates | contract verified; real product rate pending |
| CONFIDENT/POSSIBLE/NO_MATCH and no forced Top-K | `NoMatchDecider`, `records/phase_g5/g5_postprocess_smoke_20260831_2259.json` | contract verified; synthetic offline threshold/negative calibration supported, real-world calibration unmeasured |
| Canonical async task states, independent async timeout and task/results/cancel APIs | `service/models.py`, `service/runtime.py`, `service/app.py`, `tests/test_service.py`, `records/phase_g6/g6_realtime_service_smoke_recovery_20260903.json`, `records/phase_g6/g6_realtime_service_timeout_1000ms_recovery_20260903.json`, `records/phase_g6/g6_real_model_runtime_matrix_recovery_20260903.json`, `records/phase_g6/g6_real_model_two_worker_matrix_recovery_20260903_v3.json` | lifecycle/watchdog/late-fallback, CPU Stub pressure, real failure/timeout/cancel, cross-GPU overlap and CUDA/thread cleanup verified; not a production throughput claim |
| Explicit Level 0/1/2 degradation | `pipeline/service.py`, `service/runtime.py`, service tests, `records/phase_g6/g6_realtime_service_timeout_1000ms.json` | synthetic paths and one real watchdog/fallback probe verified; representative timeout matrix pending |
| Candidate lossless schema, ID uniqueness and raw prediction preservation | `build_candidate_record`, `pipeline/service.py`, `candidates` + `deduplicated`/`pre_dedup_rank`, pipeline tests | schema, ID normalization and finite-value guards verified |
| Eight canonical feedback labels and model/user bounds | `service/app.py`, `frontend/app.js`, feedback tests (including completed TIMEOUT fallback and canonical response), `records/phase8/g7_browser_smoke_20260901.json` | API/CORS/headless UI and canonical response verified; real user验收 out_of_scope |
| Badcase taxonomy and classification-before-fix | `badcase/taxonomy.py`, `docs/hardcase_protocol.md` | taxonomy verified; human-labeled badcase set out_of_scope |
| Unified evaluation outputs, IoU=0.7/mIoU/Mean Boundary Error, product rates and engineering P50/P95 | `eval/run_eval.py`, `configs/eval.yaml`, `records/phase_g8/eval_smoke_output_current_20260901/{metrics.json,badcases.json,report.md}`, `records/phase_g8/g4_pilot_eval_current_20260901_v2/metrics.json`, `tests/test_metrics.py` | target-present/synthetic schema verified; single-class target-present pilots keep No-Match rates null; synthetic negative metrics are offline-only and unmeasured fields remain null |
| Validation-set safety for No-Match/Ranking calibration | `evaluation/validation.py`, `scripts/validate_validation_manifest.py`, `scripts/build_validation_manifest.py`, `docs/validation_manifest.md`, template manifest | validator and explicit synthetic four-category manifest verified; synthetic calibration is offline mechanism evidence, real-world calibration remains unmeasured |
| G0–G9 plan, issue log and profiler-first rule | `plan/execution_plan.md`, `records/phase_g2_g4_issue_log.md`, `records/phase_g9/realtime_model_profile_recovery_20260903.json`, `records/phase_g9/gpu_driver_probe_20260901.json` | process evidence and fixed-config real profiler/smoke verified after recovery; human gates out_of_scope |
| Runtime dependency reproducibility | `docs/current_system.md`, `docs/model_download.md`, `records/phase_g9/decord_build_20260901.json`, `records/phase_g2_g4_issue_log.md` | PyPI tag warning resolved with source-built cp313 wheel repaired by auditwheel and bundled FFmpeg runtime closure; `pip check`=0, import and real VideoReader smoke pass. Portability beyond manylinux_2_39 remains an explicit build-environment constraint |

## Release interpretation

The implementation and local contract surface are runnable, but the project is
not a quality-complete release. Human annotation, four-family negative-set
calibration, and real browser user tests are explicitly out of scope. The
current TimeLens pilot must remain a diagnostic result because Top-1 IoU>=0.3
is 0.20 and the adapter's `grounding_score=1.0` is only a parseability marker.

The existing-data-only completion record is
`records/phase_g8/current_data_completion_20260901.json`; the executable plan
for real quality and user acceptance is `docs/real_acceptance_plan.md`.
