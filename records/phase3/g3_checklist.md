# G3 checklist — badcase system and regression harness

结论：**G3 条件通过（sampled-2s full-video review）；连续视频播放不作声明，G4 仍 pending**。

| 条件 | 状态 | 证据 |
|---|---|---|
| 固定 regression pool 可一键运行 | pass | `scripts/evaluate_regression_pool.py`；最终池 `clean97_test_regression_pool_final.jsonl` 已对 B1/B1A/8 秒候选运行 |
| 报告可定位逐样本证据 | pass | `clean97_test_final_*_regression_report.json`；每条含 sample_id、split、类别和 IoU，证据索引见 `outputs/phase3_clean97_unreviewed_fullscan_20260714_1430/index.json` |
| 主类别定义可操作 | pass with scope limit | `boundary_shift`、`short_event_miss`、`other_or_uncertain` 均有人工 rationale；未声称 same-object、no-match、ASR 冲突等未覆盖类别 |
| held-out 样本数和语义复核 | pass with scope limit | 最终池 18 条：8 条完成人工 sampled-2s full-video scan（4 boundary-shift、1 short-event-miss、3 other/uncertain），10 条保留为 metric-only controls；连续视频播放不作声明 |

当前不进入训练：最终池全部 `not_for_training=true`。G3 只证明诊断与回归流程可复现，不等价于候选策略已通过 G4，也不支持超出该 scope 的泛化结论。
