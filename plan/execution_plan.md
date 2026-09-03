# Execution plan — Query-aware Coarse-to-Fine Temporal Grounding

## 1. Final objective

实现一个面向长视频的离线自然语言片段检索原型：输入本地长视频和 Query，视频侧只解析/编码一次并缓存；Query-aware Retriever 召回 Top-N 候选窗口；TimeLens-8B 在候选窗口内预测绝对 start/end；独立后处理完成边界修正、排序、时间去重和 No-Match 判断；前端支持预览、边界调整和反馈，反馈进入 Badcase 分析与离线回归测试。本项目不要求真实环境复现、真实用户验收、线上部署或生产吞吐。

项目不负责自动拼接、字幕、转场、音乐、特效或最终成片生成。

## 1.1 验收边界（用户确认）

本项目不要求真实人工标注、真实浏览器用户验收、用户采用率统计、人工边界调整统计、真实硬件复现或生产压力测试。项目完成与否以“离线代码、前一项目已有数据、合成输入和可复现评测跑通主链路”为准：结果必须可复现、字段和状态契约完整、失败/超时/降级可观测，自动化指标必须标明真实数据或构造数据来源。

No-Match、Ranking 和降级逻辑允许使用明确构造的四类负样本：`event_absent`、`wrong_action`、`wrong_object`、`theme_unrelated`。构造数据可以用于离线机制测试和相对对比，但必须标记为 `synthetic`，不能表述为真实世界准确率。没有真实负样本时，真实场景泛化指标仍标记为 `not_measured`，但不再阻塞离线项目完成。

## 2. Architecture decision

```text
Long Video + Query
 -> Decode/Segment/Sample/Feature Cache
 -> Query-aware Retriever (Top-N)
 -> TimeLens-8B Grounder
 -> Boundary Refinement
 -> Candidate Ranking
 -> Temporal Deduplication
 -> No-Match
 -> Top-K Preview/Adjustment/Feedback
```

- TimeLens-8B：主 Temporal Grounding 模型，只接收候选窗口。
- VideoITG-8B：仅 baseline 和粗召回参考，不承担最终片段预测。
- Retriever、Grounder、Postprocess 必须通过稳定接口解耦。
- 完整长视频不得直接输入 8B Grounder。

## 3. Acceptance contracts

### Feature Cache

缓存键必须同时包含 `video_id`、`feature_model_version`、`preprocessing_config`、`sampling_config`；保存视频时长、模型、版本、配置、路径、创建时间。必须记录 hit/miss；同一视频第二个 Query 不得重新进行完整视频侧编码。

### Retriever

```python
class TemporalRetriever:
    def index(self, video_path, video_id): ...
    def retrieve(self, video_id, query, top_n): ...
```

输出 `{start, end, score, candidate_id}`。至少比较两种轻量方案，报告 Recall@5/10/20、Retrieval Latency、GPU Memory、Feature Cache Size，并在验证集按质量—资源规则选择默认实现。

### Grounder

```python
class TemporalGrounder:
    def predict(self, video_path, query, candidate_windows): ...
```

TimeLens 输出 `raw_start`、`raw_end`、`grounding_score`、`inference_latency`、`model_version`；所有时间为原视频绝对时间。业务代码不得依赖具体模型实现。

### Postprocess

Boundary Refinement、Ranking、Dedup、No-Match 独立配置并做开/关消融。原始边界不得被覆盖，候选记录必须保存 `coarse_*`、`raw_*`、`refined_*`、各项分数、排名、版本、阶段耗时和降级字段。

Ranking 至少使用 retrieval score、grounding score、boundary confidence、completeness 和 duplication penalty；权重只能由验证集确定。Dedup 默认 `temporal_iou_threshold: 0.7`，单独报告 Duplicate Rate。

No-Match 输出 `CONFIDENT`、`POSSIBLE` 或 `NO_MATCH`，禁止强制 Top-K。离线阈值测试可使用显式标记的 synthetic 验证集，并报告 No-Match Accuracy、False Positive Rate、False Negative Rate；报告必须区分合成测试结果与真实数据结果。负样本至少覆盖无事件、错误动作、错误对象和主题无关四类。

### Service and feedback

任务状态：`PENDING`、`PREPROCESSING`、`INDEXING`、`RETRIEVING`、`GROUNDING`、`POSTPROCESSING`、`SUCCESS`、`FAILED`、`CANCELLED`、`TIMEOUT`。接口支持创建、查询、结果、取消。任务记录保存 `task_id/video_id/query/status/progress/current_stage/model_version/timestamps/error_code/degraded`。

降级：Level 0 完整链路；Level 1 Boundary 异常返回 raw grounding；Level 2 Grounder 超时/异常返回 coarse windows。必须显式保存 `degraded/degrade_level/degrade_reason`。

反馈类型：`ACCEPT`、`IRRELEVANT`、`START_TOO_EARLY`、`START_TOO_LATE`、`END_TOO_EARLY`、`END_TOO_LATE`、`DUPLICATE`、`MISS`。人工调整同时保存 model_start/model_end 和 user_start/user_end。

## 4. G0–G9 phases

### G0 — 现有系统审计

只审计，不大规模重构。确认入口、VideoITG 调用、frame score→segment、Boundary、Dedup、Ranking、No-Match、Cache、Async、Feedback、评测集和性能瓶颈。输出 `docs/current_system.md` 与 `docs/current_architecture.md`。

### G1 — 统一模型接口

落地 `TemporalRetriever` 与 `TemporalGrounder`；VideoITG 和 TimeLens 均通过适配器接入。

### G2 — 预处理与 Feature Cache

实现 decode、segment、sampling、feature extraction/storage、版本校验和 hit/miss 日志；用同一视频两个 Query 验证无重复完整编码。

### G3 — Query-aware Retriever

比较至少两种轻量 Retriever，固定数据版本和候选定义，报告 Recall@5/10/20、延迟、显存、索引/缓存大小。

### G4 — TimeLens-8B

完成候选窗口输入、Query 输入、start/end 解析、batch inference、绝对时间映射和错误处理；已有 checkpoint/API 运行记录为可选参考，离线 adapter 契约未通过前状态必须为 pending。

### G5 — 后处理

分别实现 Boundary Refinement、Ranking、Dedup、No-Match；每个模块有独立配置和消融，保留 raw/refined 全量字段。

### G6 — 异步任务与降级

实现 progress、stage、cancel、timeout、fallback、error state 和 `/tasks/{task_id}/results`。

### G7 — 前端与反馈

实现候选预览、切换、时间轴调整、结构化反馈和任务状态展示。

### G8 — 离线评测与回归

统一入口 `python eval/run_eval.py --config configs/eval.yaml`，输出 `metrics.json`、`badcases.json`、`report.md`。Badcase 必须先分类：`retrieval_miss`、`wrong_event`、`boundary_shift`、`short_event`、`long_event`、`duplicate`、`false_positive`、`no_match_error`、`ranking_error`、`latency_timeout`、`video_decode_error`。

### G9 — 可选性能记录与打包

如需要性能信息，再 profiler Decode、Feature、Retriever、Grounder、Postprocess、Serialization；性能和打包只作为可选记录，不作为离线项目完成门槛。最终仍需完成版本、配置、指标和说明文档审计。

## 5. Experiment rules

一次实验只验证一个主变量；阈值/模型/配置/数据/代码版本必须记录；Retriever、Grounder、Boundary、Ranking 不得在同一实验同时修改；指标未提升的方案不得因开发完成而默认保留；任何行为修改必须运行普通集、困难集和历史回归池。

## 6. Current migration status

G0 文档和稳定接口/后处理基础模块已建立；Feature Cache、两个 Retriever、TimeLens adapter、完整候选结构、合成负样本支持和独立 No-Match/降级编排已落地。360 条 target-present 数据、合成后处理/负样本契约、统一评测入口、失败/超时/取消状态机和已有运行时参考均已归档。真实 GPU、真实服务压力、跨环境打包和真实用户体验不属于本离线项目完成条件；真实 TimeLens 质量仍只按已有 weak-label 数据报告，不外推正式生产质量。

## 7. Current-data-only boundary and next acceptance gate

本轮只推进现有数据可以证明的内容：79 条离线回归测试、发布审计、360 条媒体/缓存审计、360 条现有媒体池的 cache miss 提取耗时、Retriever 对比、360 条固定配置 TimeLens 自动评测、synthetic 四类负样本清单和 No-Match/Ranking 契约、9-case TimeLens 参数/失败矩阵、已有预测后处理 replay、adapter/服务契约矩阵和评测 schema。汇总证据见 `records/phase_g8/current_data_completion_20260903.json` 和 `records/phase_g8/g4_local360_eval_20260903/report.md`。这些结果用于判断离线原型的可运行性和自动化效果，不外推到真实部署或生产质量结论。

人工标注、真实浏览器用户验收、真实环境复现和生产压力测试不属于当前范围，也不是本项目完成条件。按 `docs/real_acceptance_plan.md` 执行离线验证：使用前一项目 target-present 数据、明确标记的 synthetic 四类负样本、合成输入和历史回归；No-Match 指标可以在 synthetic 数据上测试，但必须标注 synthetic，不代表真实世界泛化。已有真实模型运行记录仅作参考，不要求重新执行。
