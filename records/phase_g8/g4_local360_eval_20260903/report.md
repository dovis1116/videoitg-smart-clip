# TimeLens local360 automatic evaluation

## Scope

- 360 different videos, one query per video, selected deterministically from the previous project's local media and `video_itg_data.json`.
- 360/360 files decoded successfully and 360/360 `frame_num`/`clip_num` labels were within the audited bounds.
- Ground truth is derived from the previous project's 5-second `clip_num` field. This is target-present weak-label evidence, not human-confirmed continuous temporal annotation.
- Fixed configuration: SigLIP cache with 1 FPS and 16 frames; CachedCosine Retriever; TimeLens-8B; `SDPA`; `total_pixels=4,194,304`; `top_n=1`; `batch_size=1`.

## Results

| Metric | Result | 95% Wilson interval |
|---|---:|---:|
| R@1 IoU >= 0.3 | 110/360 = 0.3056 | 0.2602–0.3550 |
| R@1 IoU >= 0.5 | 86/360 = 0.2389 | 0.1977–0.2855 |
| R@1 IoU >= 0.7 | 28/360 = 0.0778 | 0.0544–0.1101 |

- Mean R@1 IoU: `0.1959`
- Mean boundary error: `13.36 s`
- Predictions: `360/360`
- Level-2 degradation: `0/360`
- Grounding latency: mean `1,198.70 ms`, P50 `1,112.80 ms`, P95 `1,880.74 ms`

## Stratified results

| Group | n | R@1 IoU >= 0.3 | mIoU | Boundary error |
|---|---:|---:|---:|---:|
| 2_3_m_nextqa | 10 | 0.1000 | 0.0600 | 45.48 s |
| 30_60_s_activitynetqa | 198 | 0.2626 | 0.1772 | 13.60 s |
| 30_60_s_nextqa | 152 | 0.3750 | 0.2292 | 10.93 s |
| deterministic dev | 30 | 0.4333 | 0.2794 | 11.15 s |
| deterministic test | 45 | 0.2667 | 0.1665 | 14.20 s |

The split rows are descriptive only: the selected media manifest is a local pilot, not a newly established official validation split.

## Interpretation

The 360-row run is materially stronger evidence than the previous five-row diagnostic pilot for reproducibility and failure coverage. It does not show that TimeLens quality has reached a production target: the measured R@1 IoU >= 0.3 is `0.3056`, and the labels are inherited weak labels concentrated in target-present data. No-Match, real negative categories, user preference, or formal production throughput can be inferred from this run.

Evidence files:

- Predictions: `records/phase_g4/g4_timelens_local360_recovery_20260903.jsonl`
- Metrics: `records/phase_g8/g4_local360_eval_20260903/metrics.json`
- Media audit: `records/phase_g2/local360_media_audit_20260903.json`
- Cache audit: `records/phase_g2/local360_feature_cache_audit_20260903.json`
- Retriever comparison: `records/phase_g3/g3_retriever_local360_20260903.json`
