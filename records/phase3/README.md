# Phase 3 badcase evidence

`b1_badcase_report.json` is the first metric-derived distribution from the 50-sample target-present pilot. It deliberately does not assign semantic labels such as wrong subject, duplicate, motion, or boundary error; those require manual review under `docs/hardcase_protocol.md`.

Current counts from B1 top-1 IoU:

- IoU ≥ 0.5: 25 samples (20 without a B2 rerank regression, 5 with one);
- 0.3 ≤ IoU < 0.5: 4 samples;
- IoU < 0.3: 21 samples (20 without and 1 with a B2 rerank regression).

The `+rerank_regression` suffix means B2-Retrieval-only Top-4 has a Top-k IoU≥0.3 hit while B2 Top-3 does not. It is a ranking diagnostic, not a semantic error label.

`manual_badcase_review.json` adds a provisional keyframe-based review of ten priority samples. It assigns only two `short_event_miss`, two `boundary_shift`, and six `other_or_uncertain`; full-video replay is still required before any sample enters an immutable regression pool.

The current `regression_pool.jsonl` contains only four `confirmed_keyframe` entries as a pilot regression-only pool. They remain marked `not_for_training=true` and `full_video_replay_status=pending`; this is not yet the final held-out regression gate.

G2 is passed for the 50-sample target-present pilot; see `records/phase2/g2_checklist.md`. G3 is conditionally passed for the sampled-2s clean97 review scope; continuous playback and unsupported no-match/ASR labels are not claimed.

`g3_checklist.md` records the current split: the final test-split pool has 18 rows, with eight sampled-2s manually reviewed rows and ten metric-only controls. G4 remains pending.

`segment8_candidate.json` records the first boundary-shift-oriented candidate. It is not promoted because gains are inconsistent across the 50-row pilot, four-row held-out pool, and four-row provisional train pool.

`heldout_split_regression_pool_final.jsonl` expands the test-split pool to all 7 rows: 4 provisional badcases plus 3 metric controls. Its B1 and 8-second candidate reports are `b1_heldout_final_regression_report.json` and `b1_seg8_heldout_final_regression_report.json`. Four badcase rows have a full-video 2-second scan status; controls are metric-only.

`videoitg100_media_pilot_clean.jsonl` is an expanded 97-row media-backed scope (71 train, 8 dev, 18 test) after excluding three first-frame near-duplicate candidates from the 100-row draw. The final 18-row test regression pool is `clean97_test_regression_pool_final.jsonl`; it contains eight sampled-2s manually reviewed rows (four `boundary_shift`, one `short_event_miss`, three `other_or_uncertain`) and ten metric-only controls. Every row is `not_for_training=true`; controls are deliberately not semantic labels. The review record is `clean97_test_manual_review.json`, with visual evidence under the external output paths recorded there. Reports for B1 5-second, B1 8-second, and offline B1A are `clean97_test_final_b1_regression_report.json`, `clean97_test_final_b1_seg8_regression_report.json`, and `clean97_test_final_b1a_regression_report.json`.

On this expanded test split, B1 8-second improves Top-k IoU≥0.5 from 0.556 (5-second) to 0.778, while B1A reaches 0.667 and switches only 5/97 rows. G3 is conditionally passed for the sampled-2s review scope; these results are still candidate-selection evidence, not a G4 promotion.

The adaptive B1A candidate is documented in `records/phase4/b1a_candidate.json`; its full run and repeat are identical, but it is not yet the default.

`group_metrics.json` shows B1's Recall@1@0.3 by source group: 0.20/10 on `2_3_m_nextqa`, 0.70/20 on `30_60_s_activitynetqa`, and 0.65/20 on `30_60_s_nextqa`. The overall B1 gain is therefore partly driven by the ActivityNet group; it is not evidence of uniform generalization.

`evaluate_regression_pool.py` is the one-command evaluator. The first B1 report is stored at `records/phase3/b1_regression_report.json`.

The four train-split provisional rows have now received sampled-2s full-video review and are documented separately in `records/phase4/train_badcase_manual_review.json`. They remain outside the held-out test pool and are not used for training yet; `records/phase4/targeted_badcase_pool_audit.json` records the evidence limit.
