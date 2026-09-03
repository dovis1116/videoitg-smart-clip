# Current architecture after migration

当前业务目标是 Query-aware Coarse-to-Fine Temporal Grounding，接口层已经与具体模型解耦：

```text
video -> decode/segment/sample -> FeatureCache -> TemporalRetriever.index
query + cached features -> TemporalRetriever.retrieve(Top-N)
candidate windows -> TemporalGrounder.predict(TimeLens)
raw predictions -> BoundaryRefiner -> CandidateRanker
             -> TemporalDeduplicator -> NoMatchDecider -> Top-K
             -> frontend preview/adjustment -> feedback -> badcase regression
```

`VideoITGWorker` 和旧 `reranker/` 只保留为 baseline compatibility path。新的业务层只能依赖 `pipeline.contracts.TemporalRetriever` 与 `TemporalGrounder`；不得在业务逻辑中直接导入 VideoITG 或 TimeLens。

当前已落地的可运行基础模块：

- `preprocessing.feature_cache.FeatureCache`
- `preprocessing.feature_encoder.SigLIPFeatureEncoder` 与 `decode_uniform_frames`（G2 已完成最小真实视频 smoke；服务默认 1 FPS、最多 16 帧，采样预算进入缓存身份）
- `retrieval.temporal.CachedCosineRetriever` / `UniformTemporalRetriever`
- Pipeline 会为 Retriever 缺失或重复的 `candidate_id` 生成稳定唯一 ID，并拒绝非有限时间/分数，避免候选覆盖和排序污染。
- Retriever 仅对文本 embedding 做固定数据集答题后缀清洗；Grounder 保留原始 Query
- `grounding.timelens.TimeLensGrounder`（单/双候选串行与真 batch smoke 已通过；未配置 checkpoint 时显式失败）
- `pipeline.postprocess` 中 Boundary、Ranking、Dedup、No-Match
- `pipeline.service.CoarseToFinePipeline`
- `evaluation.validation` 与 `scripts/validate_validation_manifest.py`：离线 No-Match/Ranking 校验；支持 synthetic 四类负样本覆盖，真实世界校准不在范围内

这些模块组成离线原型主链路；TimeLens 质量结果按前一项目 weak-label 数据解释，synthetic 负样本只用于机制测试。真实环境、用户验收和生产部署不属于当前完成条件。
