# G2 checklist — target-present pilot

结论：**G2 条件通过（仅限当前 50 条 target-present media pilot）**。这不是全量 VideoITG-40K、无匹配或上线结论。

| 条件 | 状态 | 证据 |
|---|---|---|
| B0/B1/B2 可用同一命令/配置重跑 | pass | `scripts/run_baseline.py`；首次与重复 run 的 `summary.json` |
| 逐样本输出和汇总指标可追踪 | pass | `records/phase2/baseline_comparison.json`；B0/B1/B2 各 50 行 JSONL |
| 至少重复运行并排除明显非确定性 | pass | `records/phase2/repeatability_audit.json`；三组逐样本预测/指标差异均为 0 |
| B2 质量/成本相对位置实测 | pass | `docs/baselines.md`；B2 Recall@1@0.3=0.24、Top-k@0.3=0.54，平均候选重排约 6.24 s |
| 第一版 badcase 分布 | pass | `records/phase3/b1_badcase_report.json`；指标型分布 + provisional 关键帧复核 |

## 仍未通过的后续条件

- G3 已在 clean97 sampled-2s review scope 下条件通过；最终 18 条 test pool 见 `records/phase3/clean97_test_regression_pool_final.jsonl`，8 条人工复核、10 条 metric-only controls。
- 语义类别限定为 `short_event_miss`、`boundary_shift` 和 `other_or_uncertain`，未声称覆盖 wrong-subject、wrong-action、no-match 或 ASR 冲突；连续视频播放不作声明。
- 任何训练/阈值选择仍需使用 train/dev，不能使用最终 test pool；G4 尚未通过。
