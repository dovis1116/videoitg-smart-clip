# Plan changes

## 2026-09-03 — Offline-only acceptance scope clarified

- **Change:** 将项目说明和验收边界统一为离线原型；真实环境复现、真实用户验收、生产部署和生产吞吐不再是完成条件。
- **Added:** 允许使用显式标记的 `synthetic` 四类负样本测试 No-Match、Ranking 和降级逻辑；合成结果不得表述为真实世界指标。
- **Reason:** 与用户确认的交付目标一致，避免把部署/线上证据误当作离线工程完成条件。

## 2026-09-03 — Existing-data automatic evidence expanded to 360 videos

- **Evidence:** 前一项目 annotation 共 474,354 条；本地可匹配 360 个不同视频和 2,817 条 query，360/360 通过媒体、标签和 cache 审计，并完成 Retriever 与 TimeLens 固定配置自动评测。
- **Change:** 将当前数据收尾证据从 100 条媒体扩展为 360 个不同视频；新增 manifest、媒体/cache 审计、Retriever 对比和 TimeLens 评测报告，并将其纳入 release audit。
- **Boundary:** 结果仍是 target-present、`clip_num` weak-label 自动诊断；不新增人工标注，不据此推出正式质量、No-Match 或生产吞吐结论。

## 2026-07-14 — G1 scope correction

- **Evidence:** VideoITG-40K metadata has 474,354 records, but `motion` and `existence` are `No` for every record; no ASR field or independently verified 2–5 second boundary is available. The 50-video media pilot supports target-present temporal grounding only.
- **Change:** G1 is conditionally passed for a target-present temporal-grounding pilot. Phase 2 B0/B1/B2 baselines will report only this scope. No-match, ASR/visual conflict, and short-event boundary claims remain deferred to a manually supplemented Phase 4 dataset.
- **Reason:** This follows the execution plan's fallback rule for a public dataset that cannot support the intended task; it prevents unsupported negative-label or boundary metrics.

## 2026-07-14 — Clean-test review accounting correction

- **Evidence:** `clean97_test_regression_pool.jsonl` contains 18 rows: 4 previously reviewed rows, 4 metric-unreviewed candidates, and 10 metric controls. The earlier “11 rows awaiting semantic review” wording counted the controls and did not match the frozen pool.
- **Change:** The final pool separates 8 manually reviewed rows (uniform 2-second full-video scans) from 10 metric-only controls; only the 4 newly classified candidates were assigned semantic labels.
- **Reason:** Controls remain useful for regression coverage but must not be presented as semantic badcase labels; the distinction keeps G3 evidence auditable.

## 2026-07-14 — G5 traffic-contract split and restricted Phase 6

- **Evidence:** `records/phase5/multi_gpu_audit.json` shows 0/50 timeout for dual-GPU steady-1s but 46/50 timeout for the 50-request synchronous burst; Top-k@0.5 remains 0.68 in both. Prefetch, batch, and admission audits do not remove the burst capacity gap without unacceptable quality or coverage loss.
- **Change:** Replace the single G5 gate with G5-A steady interactive (passed only for the measured dual-GPU steady-1s scope) and G5-B synchronous burst (unsupported). Permit a restricted Phase 6 with steady synchronous service and bounded asynchronous burst tasks; do not claim synchronous burst support.
- **Reason:** The evidence identifies a traffic-contract/capacity mismatch rather than a remaining frame-budget tuning problem. The plan now matches the measured service capability and makes overload explicit.
# 2026-08-31 — TimeLens coarse-to-fine architecture migration

- **Change:** 项目主路线由 VideoITG-8B 全视频/候选精排替换为 Query-aware Coarse-to-Fine Temporal Grounding；TimeLens-8B 成为主 Grounder，VideoITG 保留为 baseline/参考。
- **Added:** 版本化 Feature Cache、独立 TemporalRetriever/TemporalGrounder contracts、Boundary/Ranking/Dedup/No-Match 模块、完整候选中间结果、生命周期状态和显式 Level 0/1/2 降级。
- **Evidence boundary:** 当前只完成 G0 审计和可运行接口/占位基础；TimeLens checkpoint/API、真实轻量编码器、阈值和新路线指标仍 pending，不用历史 VideoITG 指标替代。

## 2026-09-01 — TimeLens local-memory-safe default

- **Evidence:** 当前 Miniconda 环境无 `flash-attn`；TimeLens 使用 SDPA 时 14.68M `total_pixels` 在 24 GiB RTX 4090D 上触发 OOM，而 4,194,304 pixels 的单候选异步真实 smoke 成功，端到端约 20.7 s、`degraded=false`。
- **Change:** 将 TimeLens adapter、`configs/default.yaml`、`run_service.py` 和 profiler 默认视觉预算统一为 4,194,304；检测到 `flash-attn` 时仍可显式提高预算并单独复测。
- **Reason:** 保证当前本地环境默认可运行，避免把未经验证的显存假设写入默认路径；质量和吞吐仍需在验证集重新评估。

## 2026-09-01 — Existing-data-only completion boundary

- **Evidence:** 当前可复核数据包括 75 条回归通过、81/81 release audit、缓存双 Query smoke、100 条 Retriever target-present pilot、5 条 TimeLens diagnostic pilot 和 4 条合成评测输入；验证清单仍有 79 条 pending 正样本，四类负样本和 human browser test 缺失，宿主 GPU 驱动仍异常。
- **Change:** 本轮只完成现有数据可支持的 CPU-only 收尾与证据归档；真实效果、No-Match/Ranking 正式校准、真实 GPU 性能和真实用户验收转入 `docs/real_acceptance_plan.md`，并保持 pending/blocked。
- **Reason:** 防止把接口 smoke、stub workflow 或小规模 pilot 误报为生产质量和真实用户结论；后续按数据→单变量评测→真实用户验收顺序推进。

## 2026-09-01 — Remove manual annotation and human browser acceptance

- **Evidence:** 用户明确要求不进行人工标注和真实浏览器用户验收；当前 79 条清单为 pending 正样本，四类真实负样本不存在，自动 workflow 记录均为 `human_user_test=false`。
- **Change:** 将 G8/G7 的人工标注、真实用户验收和相关产品指标改为 `out_of_scope`；保留 target-present pilot、合成契约、接口回归和可复现 profiler/smoke。No-Match Accuracy/FPR/FNR、用户采用率和人工调整统计在无相应数据时标记 `not_measured`。
- **Reason:** 使项目范围与用户约束一致，避免用弱标签、stub 或 headless smoke 冒充人工质量结论。

## 2026-09-03 — Existing-data automated acceptance boundary clarified

- **Evidence:** 用户明确要求只在前一个项目已有数据上跑通并达到可由该数据支持的预期效果，不要求真实人工验证；当前工程已有 100 条媒体/cache/Retriever 统计、TimeLens 运行时矩阵和完整服务契约证据，但真实负样本与人工偏好数据不存在。
- **Change:** 将“前一个项目已有数据上的可复现主链路和自动化效果”明确为当前完成标准；移除真实人工标注、真实浏览器验收、用户采用率和人工边界调整作为完成条件。缺少数据支持的 No-Match/人工偏好类指标标记为 `out_of_scope` 或 `not_measured`。
- **Reason:** 使计划的验收条件与用户目标和可获得数据一致，同时保留证据边界，避免把未测指标写成已达到的效果。
