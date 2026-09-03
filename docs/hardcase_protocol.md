# 困难集协议（迁移后）

## 原则

历史 VideoITG-40K 的 `motion` 和 `existence` 字段不能直接解释为“无匹配”或“静态事件”。本项目不执行人工标注；离线测试允许按明确规则构造 synthetic 候选，但必须保留构造来源和标签，不能冒充真实质量指标。Badcase 仍必须先分类再决定修改 Retriever、Grounder、Boundary、Ranking 或阈值。

## 分类

| code | 复核标准 | 自动候选信号 | 当前状态 |
|---|---|---|---|
| `wrong_event` | 主体相同但动作不符，或动作相似但主体/对象不符 | 查询/答案包含主体、动作和对象词 | 弱标签诊断 |
| `short_event` | 目标事件持续 2–5 秒但未被完整召回 | 单个 5 秒 clip 或起止边界紧邻 | 当前标签分辨率不足 |
| `duplicate` | Top-K 结果重叠过高 | 评测预测阶段生成 | Phase 2 |
| `false_positive` | 视频中没有目标但系统返回候选 | 合成 `event_absent` 或显式负样本 | synthetic offline |
| `no_match_error` | CONFIDENT/POSSIBLE/NO_MATCH 状态判断错误 | 分数阈值、margin、多候选一致性 | synthetic offline |
| `boundary_shift` | 语义事件存在但边界偏移 | 相邻 clip 多选 | 自动指标可测 |
| `asr_visual_conflict` | ASR/文字与画面冲突 | 当前元数据未提供 ASR | 不可用，需补数据 |

统一回归分类还包括：`retrieval_miss`、`wrong_event`、`boundary_shift`、`short_event`、`long_event`、`duplicate`、`false_positive`、`no_match_error`、`ranking_error`、`latency_timeout`、`video_decode_error`。

## 当前 pilot 产物

`records/phase1/hardcase_candidates.jsonl` 只保存候选标签和证据字段，自动标签均视为 `weak_label`。它不能参与正式训练或测试集指标，只能用于错误分布诊断。

当前数据集不提供可靠的 2–5 秒边界、真实无匹配真值或 ASR，因此真实世界泛化类别不作结论；离线报告可以使用 synthetic 四类负样本测试 No-Match、Ranking 和降级逻辑，但必须与 target-present weak-label 结果分开统计。
