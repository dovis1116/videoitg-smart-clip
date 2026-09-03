# 离线自动验证方案

更新时间：2026-09-03

本项目是离线原型，不要求真实环境复现、真实浏览器用户验收、线上部署或生产吞吐。验证只使用前一项目数据、合成输入、构造负样本、代码契约和可复现的离线评测；已有真实模型运行记录仅作参考。

## 一、现有数据已完成范围

证据汇总在 `records/phase_g8/current_data_completion_20260903.json`：离线回归 79 passed、360 条媒体池存在性/解码/缓存/提取耗时与 Retriever 对比、360 条固定配置 TimeLens weak-label 自动评测、synthetic 四类负样本清单支持、后处理/No-Match 契约、已有运行时参考、统一评测 schema 和 Decord artifact 均已复核。

## 二、可继续执行的自动验证

1. **Target-present 离线评测**：直接使用已有 Query/区间记录，报告 Retriever Recall@5/10/20、TimeLens R@1 IoU、mIoU、边界误差；结果标记为 pilot 或 weak-label scope，不外推到完整数据集。
2. **后处理消融**：在已有预测或合成输入上分别开关 Boundary、Ranking、Dedup，检查 raw/refined 保留、重复率和排序稳定性；不把合成结果当真实质量。
3. **缓存和运行时回归**：重复 Query、跨实例锁、任务状态、取消、超时、Level 1/2 fallback、反馈字段和 JSONL 持久化均纳入自动测试。
4. **离线资源记录**：如有需要，记录 CPU/GPU、阶段耗时和缓存大小；不要求真实硬件复现，也不形成生产吞吐结论。
5. **Synthetic 负样本测试**：使用 `scripts/build_validation_manifest.py --synthetic` 构造 `event_absent`、`wrong_action`、`wrong_object`、`theme_unrelated`，可测试 No-Match/Ranking/降级逻辑；输出必须标记 `data_type=synthetic`。

## 三、明确不执行的项目

- 不人工确认现有 target-present 样本，也不把 weak label 写成连续边界真值。
- 不把 synthetic 负样本声称为真实负样本；不得用 `existence=No` 自动推断真实 NO_MATCH。
- 不执行真实参与者浏览器测试，不报告用户采用率、满意度或人工调整统计。

因此，synthetic 验证清单可以执行 `--require-complete --require-negative-categories`，但报告必须注明这是构造数据测试；真实世界 No-Match 指标保持 `not_measured`，不阻塞离线交付。

## 四、自动验收门槛

- 代码/接口：全量回归通过，release audit 无缺失文件，候选 schema、生命周期和反馈枚举全部通过。
- 目标存在场景：固定 pilot 上完整输出 Recall、IoU、边界误差和延迟，并保留数据范围。
- 无匹配场景：使用 synthetic 四类负样本验证状态机、阈值计算和空结果契约；指标标记 `data_type=synthetic`，不解释为真实世界 FPR/FNR。
- 性能：只报告实际采集到的阶段耗时、显存和超时；未运行字段写 `null`，不要求生产压测。
- 任一自动回归失败，先按 Badcase 分类，再只修改对应模块；不因接口测试通过而宣称质量达标。

## 五、现象—可能问题—排查—处理

| 现象 | 可能性 | 排查证据 | 处理方向 |
|---|---|---|---|
| Top-N 没有目标事件 | 采样过稀、Query 截断、特征不匹配、窗口不合适 | 看采样帧、Query token、Recall@N、窗口覆盖 | 先修 Retriever/采样 |
| 相似人物但动作错误 | 只学到外观、动作语义不足 | 查看 `wrong_event` 和动作相似样本 | 增加自动困难样本，单独评估 Retriever |
| 动作相似但对象错误 | 对象词截断、对象辨识弱 | 检查 Query 截断和对象词覆盖 | 修 Query 预处理或 Retriever |
| start/end 偏早或偏晚 | 窗口偏置、解析偏移、采样粒度不足 | 比较 raw/refined 和方向性误差 | 做 Boundary 开关/局部采样实验 |
| 短事件漏检 | 采样间隔大、窗口过长 | 检查事件帧是否被采到 | 提高局部采样或缩小窗口 |
| 长事件被截断 | 候选上限小、边界扩展受限 | 比较 coarse_end 与预测长度 | 调整窗口/边界策略 |
| 结果重复 | Dedup 未生效、候选窗口重复 | 检查 tIoU、`pre_dedup_rank` 和 penalty | 修 Dedup/Ranking |
| 无匹配仍返回结果 | 真实负样本缺失或 synthetic 构造规则不充分 | 检查四类 synthetic 输入、阈值和候选分数 | 先修 No-Match 逻辑；结果标记 `synthetic`，不冒充真实指标 |
| 有事件却 NO_MATCH | 漏召回或阈值过高 | 分开检查 Recall 和阈值输出 | 只做 target-present 诊断，不下正式结论 |
| 第二个 Query 仍完整编码 | cache key 漂移或锁失效 | 查看 hit/miss、四字段 key、extractor_calls | 修 Cache 身份/锁 |
| 任务长期 INDEXING/GROUNDING | GPU 驱动、OOM、解码阻塞、worker 死锁 | 查看 stage、error_code、已有进程快照 | 分类后执行显式 fallback |
| TIMEOUT 覆盖终态 | watchdog/worker 竞态 | 检查状态时间线和 degrade_reason | 保留 TIMEOUT，迟到结果只写 Level-2 |

## 六、下一步顺序

1. 重新运行全量回归和 release audit，确保本轮新增脚本、锁和证据文件同步。
2. 继续运行现有 pilot、合成后处理、缓存和服务回归，不新增人工数据依赖。
3. 更新 `metrics.json`、`badcases.json`、`report.md` 时继续明确 `not_measured` 项。

项目最终交付声明限定为“可离线复现的长视频片段检索工程原型及自动化验证结果”，包含前一项目 weak-label 诊断和 synthetic 机制测试，不包含真实用户体验、真实环境兼容性或生产部署结论。
