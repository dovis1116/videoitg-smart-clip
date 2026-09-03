# Query-aware Coarse-to-Fine Temporal Grounding

## Final project definition

本项目实现面向长视频的自然语言片段检索系统：视频先解析、分段并缓存视频侧特征；Query-aware Retriever 快速召回 Top-N 候选窗口；TimeLens-8B 在候选窗口内执行细粒度 Temporal Grounding；之后由独立的 Boundary Refinement、Ranking、Temporal Deduplication 和 No-Match 模块生成最终结果。用户可预览、调整边界并提交反馈，反馈用于 Badcase 分析、离线评测和回归测试。

系统不负责自动拼接多个片段、生成最终成片、字幕、转场、音乐或特效。

本项目是离线原型：输入、预测、合成数据和评测结果均以本地文件复现为主，不要求真实用户、线上部署或生产吞吐。No-Match/Ranking 可使用显式标记的 synthetic 四类负样本进行机制测试，但 synthetic 指标不等同于真实世界质量。

## End-to-end flow

```text
Long Video + Natural Language Query
  -> Decode / Segment / Sample
  -> Feature Cache (video-side, versioned)
  -> Query-aware Coarse Retriever
  -> Top-N candidate windows
  -> TimeLens-8B Temporal Grounder
  -> Boundary Refinement
  -> Candidate Ranking
  -> Temporal Deduplication
  -> No-Match decision
  -> Top-K candidates
  -> Preview / manual adjustment / feedback
  -> Badcase analysis / regression
```

TimeLens-8B 是主 Grounder；VideoITG-8B 仅为 baseline 和粗召回参考，不承担业务主路径的最终时间片段预测。

## Module boundaries

### Feature Cache

缓存键必须同时包含 `video_id`、`feature_model_version`、`preprocessing_config` 和 `sampling_config`。缓存元数据还保存视频时长、模型名称、特征路径和创建时间。每次访问记录 `cache_hit`/`cache_miss`；重复 Query 不得重新执行完整视频侧编码。

### Temporal Retriever

```python
class TemporalRetriever:
    def index(self, video_path, video_id): ...
    def retrieve(self, video_id, query, top_n): ...
```

Retriever 优先优化 Recall@5/10/20、延迟、GPU memory 和缓存大小，不负责精确事件边界。当前实现提供 `CachedCosineRetriever` 与 `UniformTemporalRetriever` 两种可比较方案。
`CachedCosineRetriever` 仅在查询编码前去除固定数据集答题格式后缀；原始 Query 仍完整传给 TimeLens Grounder。

### Temporal Grounder

```python
class TemporalGrounder:
    def predict(self, video_path, query, candidate_windows): ...
```

`TimeLensGrounder` 只接收候选窗口和 Query；所有输出边界必须映射回原始视频绝对时间轴。TimeLens checkpoint/API 的本地窗口与真 batch smoke 已验证；未配置 checkpoint 或运行时异常时，适配器会显式失败，业务层按 Level 2 降级。

### Independent post-processing

Boundary、Ranking、Dedup 和 No-Match 是独立模块。Boundary 支持 start/end 独立偏移和局部扩展，默认参数为 0，需由验证集选择；模型 raw 输出永不被 refined 结果覆盖；候选记录必须同时保存 `coarse_*`、`raw_*`、`refined_*`、各项分数、版本、阶段耗时和降级字段。

## Candidate record

```json
{
  "candidate_id": "video:r0",
  "coarse_start": 120.0,
  "coarse_end": 180.0,
  "retrieval_score": 0.87,
  "raw_start": 133.2,
  "raw_end": 141.8,
  "grounding_score": 0.91,
  "refined_start": 132.8,
  "refined_end": 142.4,
  "final_score": 0.89,
  "duplication_penalty": 0.0,
  "deduplicated": false,
  "pre_dedup_rank": 1,
  "rank": 1,
  "retriever_version": "cached-cosine-v1",
  "grounder_version": "TimeLens-8B",
  "postprocess_version": "boundary-refinement-v1+ranking-v1+temporal-dedup-v1+no-match-v1",
  "retrieval_latency_ms": 0,
  "grounding_latency_ms": 0,
  "postprocess_latency_ms": 0,
  "degraded": false,
  "degrade_level": 0
}
```

## No-Match and degradation

结果状态为 `CONFIDENT`、`POSSIBLE` 或 `NO_MATCH`。阈值由验证集确定，依据 retrieval score、grounding score、final score、Top-1/Top-2 margin 和候选一致性综合判断。

离线 No-Match 指标中的 `actual_match=true` 表示事件存在，`false` 表示清单声明的 absent 样本；absent 样本可以是 synthetic 构造数据。评测入口据此计算拒识 Accuracy、FPR 和 FNR，但报告必须标明数据类型。

- Level 0：Retriever → TimeLens → Boundary → Ranking/Dedup。
- Level 1：Boundary 异常时返回 raw grounding 边界。
- Level 2：TimeLens 超时/异常时返回 coarse candidate windows。

当状态为 `NO_MATCH` 时，用户侧 `predictions` 为空，避免强制展示 Top-K；完整候选中间结果保留在 `candidates` 字段供诊断和阈值校准。

降级必须显式记录 `degraded`、`degrade_level` 和 `degrade_reason`，禁止静默降级。

## Service lifecycle

任务状态统一文档名为 `PENDING`、`PREPROCESSING`、`INDEXING`、`RETRIEVING`、`GROUNDING`、`POSTPROCESSING`、`SUCCESS`、`FAILED`、`CANCELLED`、`TIMEOUT`。兼容旧 `/v1/*` 路径，同时提供 `/tasks`、`/tasks/{task_id}`、`/tasks/{task_id}/results` 和 `DELETE /tasks/{task_id}`。
