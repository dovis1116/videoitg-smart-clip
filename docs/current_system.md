# Current system audit (G0)

本文件记录 2026-08-31 架构迁移前后的可核验状态，避免把目标设计误写成已验证能力。

## Existing entry points

- 视频入口：`POST /v1/tasks/upload`、`POST /v1/tasks`，兼容路径为 `/tasks`。
- Query 入口：`RerankRequest.query`，前端来自查询输入框。
- 旧模型入口：`service/runtime.py::VideoITGWorker` → `reranker/videoitg_adapter.py`。
- 旧帧分数到片段：`frame_score_predictions`，基于帧中心生成非重叠时间区间。
- Boundary：已有 `boundary_head.py` 和 `boundaries/`，新独立 contract 位于 `pipeline/postprocess.py`。
- Dedup：旧逻辑在帧区间生成阶段；新 Temporal IoU 去重位于 `TemporalDeduplicator`。
- Ranking：旧路径主要按模型帧分数排序；新 `CandidateRanker` 接收召回、grounding、完整度和重复惩罚。
- No-Match：旧服务没有正式拒识状态；新 `NoMatchDecider` 已提供，阈值仍为 pending validation。
- Cache：迁移前没有统一的视频侧特征缓存；新 `FeatureCache` 已实现版本化落盘和 hit/miss 标记。G2 已用本地 SigLIP 在真实视频上完成最小双 Query smoke；完整视频采样预算和更大验证集仍待评测。
- Async task：已有有界队列、状态持久化、取消和恢复；新响应增加 lifecycle stage/progress/error/degraded 字段及 results endpoint。
- Feedback：已有 `/v1/feedback`；新反馈可关联 `task_id`、`candidate_id`、`video_id`、`query`、`model_version` 和 `final_score`，不覆盖模型 raw 预测。
- Validation manifest：`evaluation.validation` 与 `scripts/validate_validation_manifest.py` 已提供字段、GT/`actual_match` 一致性和四类负样本覆盖校验；`scripts/build_validation_manifest.py --synthetic` 可生成带 `label_status=synthetic` 的构造负样本，供离线 No-Match/Ranking 逻辑测试。
- Runtime dependency：上游 PyPI wheel 的 `cp36` tag 与当前 Python 不兼容。已有 Decord cp313 构建和真实读帧记录作为可选参考；真实环境、跨系统兼容和 GPU 复现不属于离线项目完成条件。证据见 `records/phase_g9/decord_build_20260901.json`。

## Current model truth

VideoITG-8B 已有 baseline 证据；TimeLens-8B 的 adapter、候选窗口输入、输出解析和绝对时间映射均有代码契约与已有运行记录。真实 checkpoint/GPU smoke 是可选参考；离线项目不宣称生产可用，也不要求复现真实硬件。

## Offline validation focus

1. TimeLens 在前一项目 weak-label 数据上的离线基线与失败分类。
2. Feature Cache 的重复 Query 命中率和 Retriever Recall@5/10/20。
3. 用四类 synthetic 负样本测试 No-Match、Ranking、Dedup 和降级逻辑。
4. Boundary、Ranking、Dedup 各自消融及离线阶段耗时。
5. 真实标注、真实环境、用户采用率和生产压力不在当前验证范围。
