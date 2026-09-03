# Scripts

后续脚本按阶段添加，并优先提供统一入口：环境核验、数据审计、baseline、Badcase 报告、训练、profiling、服务启动和 smoke test。每个脚本必须支持 `--help`，且不得把大型输出默认写入仓库。

当前脚本：

- `phase0_smoke.py`：单视频 VideoITG 32 帧 smoke。
- `audit_videoitg40k.py`：全量元数据 schema、重复和 video-level split 审计。
- `select_media_pilot.py`：只选择已落地媒体的 pilot。
- `audit_pilot_media.py`：解码、1 FPS frame、5 秒 clip 索引检查。
- `screen_pilot_duplicates.py`：精确 hash 和首帧 hash 重复初筛。
- `phase1_adapter_smoke.py`：候选片段 adapter 真实模型调用。
- `run_baseline.py`：在同一 target-present pilot、5 秒 GT 和候选协议下运行 B0/B1/B1A/B1D/B1E/B2/B2R（B1A 为 16 帧分数阈值自适应区间，B1D 为 32 帧分数阈值自适应区间，B1E 为 top-1 附近局部 zoom，B2R 为 B2-Retrieval-only），并输出逐样本 JSONL 与汇总指标。
- `replay_latency_policy.py`：使用已完成 pilot 的逐样本运行时做离线队列/策略回放；明确标记为 simulation-only，不代表真实服务吞吐。
- `score_hard_negative_pool.py`：对已人工复核的 train-side temporal hard negatives 与 GT 区间做 VideoITG 分数对照；结果仅作诊断，强制保留 `not_for_training=true`。
- `materialize_frame_hard_negatives.py`：从已确认 train badcase 的 B1 全视频错误峰中物化 frame-level 诊断候选；不赋予语义 no-target 标签，也不放开训练。
- `build_badcase_report.py`：从 B1/B2R/B2 逐样本输出生成仅基于 IoU 的 badcase 分布；不自动赋予语义错误标签。
- `export_badcase_evidence.py`：为优先 badcase 导出 GT/预测关键帧 contact sheet 和复核元数据。
- `build_regression_pool.py`：只把明确标记为 `confirmed_keyframe` 的样本物化为 pilot regression pool，并强制标记为不可训练。
- `evaluate_regression_pool.py`：对固定 regression pool 一键计算总体和按 badcase 类别的回归指标。
- `export_fullscan_evidence.py`：按固定时间间隔导出指定视频的全视频 contact sheet，供人工语义复核；不生成训练标签。
- `build_split_regression_pool.py`：把指定 video-level split 的 provisional badcase 与 metric controls 合并为不可训练的回归池。
- `summarize_group_metrics.py`：按 source group 和 video-level split 汇总 B0/B1/B2，检查优势是否由小组驱动。
- `build_adaptive_variant.py`：按 train-selected 的 Top-1 分数阈值，在 5 秒/8 秒 B1 结果间生成离线 B1A 自适应候选。
- `filter_media_manifest.py`：根据 duplicate-screen 结果排除待人工确认的 near-duplicate 视频，生成干净评测 manifest。
- `build_temporal_hard_negative_candidates.py`：从已确认 train-side badcase 生成相邻同视频 hard-negative 候选；默认全部 `not_for_training=true`。
- `materialize_reviewed_hard_negatives.py`：只物化已人工复核的相邻区间候选；不自动放开训练权限。
- `select_additional_heldout.py`：从未进入既有 manifest 且媒体实际存在的指定 hash split 中选择独立 held-out；不混入训练或阈值调参。
- `compare_budget_runs.py`：对同一 sample_id 集合的固定帧数运行做配对质量、来源组和延迟审计。
- `select_manifest_split.py`：按 video-level split 从既有 manifest 生成固定子集，不改变原始 manifest。
- `evaluate_boundary_postprocess.py`：只在 train/dev frame-score dump 上网格评估轻量时序后处理候选，禁止把 test 用于调参。
- `apply_boundary_postprocess.py`：将已在 train/dev 预注册的时序后处理参数应用到冻结预测，并输出可回滚的 prediction JSONL。
- `train_boundary_head.py`：只用 train frame scores 和 GT 区间拟合 CPU ridge boundary head，输出可审计 JSON 参数。
- `apply_boundary_head.py`：把冻结 boundary head 应用于 B1 frame-score dump，保持 5 秒候选和输出格式不变。
- `benchmark_budget_policy.py`：在单 GPU 上真实执行固定 E0 与队列积压 retrieval-only 降级策略；记录服务耗时、队列等待、超时和质量，区别于 simulation-only 回放。
- `benchmark_budget_policy.py --retain-cuda-cache`：可选地跳过每个候选后的 CUDA allocator 清理，用于运行时微优化对照；该开关不改变质量，不能替代 G5 真实质量/延迟审计。
- `benchmark_budget_policy.py --prefetch-read`：在固定 16 帧路径上，用单个 CPU worker 预读下一条 full-video frame batch 并与当前 GPU 精排重叠；质量保持不变，但仅为 opt-in pipeline 优化，不能替代 G5 服务/尾延迟验收。
- `benchmark_multi_gpu_workers.py`：用共享请求队列和每 GPU 一个 VideoITG worker 做本地多卡容量实验；包含一条 lookahead CPU 预取，但明确不是 Phase 6 服务实现，burst/steady 结果必须分开报告。
- `VideoITGReranker.rank_batch`：仅供实验的 exact-token-length 小 batch 接口；当前实测不快于两次单请求，不能作为默认路径。
- `run_service.py`：启动受限本地服务；`steady` 请求走同步 deadline，burst 走有界异步任务队列。默认 loopback；VideoITG backend 仅用于历史 baseline/兼容 smoke，新主链路通过 TemporalRetriever/TemporalGrounder contracts 接入。
- `run_service.py` 的 `--timelens-max-new-tokens`、`--timelens-total-pixels`：显式配置 TimeLens 生成预算和视觉 token 预算，避免模型参数隐式写死。
  当前默认 `total_pixels=4,194,304`，适配无 `flash-attn` 的 24 GiB GPU；adapter 检测到 `flash-attn` 时可单独提高并复测。
- `run_service.py` 的 `--async-timeout-ms`：为异步 `/tasks` 任务设置独立超时；同步 `--deadline-ms` 不会覆盖该预算。
- `g2_feature_cache_smoke.py`：使用真实 SigLIP 在视频上做一次视频侧编码并执行双 Query cache-hit smoke。
- `evaluate_retrievers.py`：在固定媒体 manifest 上比较 CachedCosine 与 Uniform 的 Recall@5/10/20、延迟和缓存大小；结果必须注明 target-present/pending 范围，不能自动替代 query-aware 默认方案。
- `summarize_feature_cache.py`：不重新编码地审计既有缓存的覆盖、采样帧数、特征维度、视频时长、磁盘大小和已持久化的提取耗时。
- `g4_timelens_smoke.py`：对一个或多个候选窗口执行真实 TimeLens 推理，验证时间戳解析和绝对时间映射。
- `g4_timelens_runtime_matrix.py`：以独立子进程运行合法窗口、batch、视觉预算、生成预算和损坏媒体 case，并关联真实异步 TIMEOUT；高预算成功只表示本次未复现 OOM，不代表容量上限。
- `g4_timelens_contract_matrix.py`：用假推理验证 TimeLens adapter 的 batch、局部时间映射、截断和错误返回契约；不代表真实模型质量/性能。
- `g4_timelens_pilot.py`：复用已缓存视频特征，在固定 target-present pilot 上评估 TimeLens Top-k grounding 与 coarse Recall；结果仅作质量诊断，不自动晋级模型或阈值。
- `g5_postprocess_smoke.py`：对 Boundary 开/关、Ranking、Temporal Dedup 和 No-Match 做独立合成契约 smoke；输出不代表真实质量或阈值校准。
- `ablate_postprocess.py`：在既有预测 JSONL 上重放 Boundary/Ranking/Dedup 开关；缺失 raw 字段时显式标记 preview-bound proxy。
- `g6_realtime_smoke.py`：在外置短视频上执行真实 SigLIP→CachedCosine→TimeLens→后处理单请求链路并保存阶段/降级记录；不代表并发吞吐或质量 Gate。
- `g6_realtime_service_smoke.py`：通过 FastAPI `/tasks` 异步接口执行同一真实链路并轮询 canonical 状态；不代表并发吞吐或质量 Gate。
  传入 `--async-timeout-ms` 可生成真实超时契约记录，非默认输出写入独立文件。
- `g6_real_model_runtime_matrix.py`：在单个真实 TimeLens Worker 上验证队列取消、任务终态、显存释放和线程清理；不代表多卡吞吐。
- `g6_real_model_two_worker_matrix.py`：在两张 GPU 上验证两个真实 TimeLens Worker 的请求重叠、失败/降级状态和 shutdown 资源释放；不代表生产压测。
- `profile_pipeline.py`：在 stub grounder 下对新编排的 Retriever/Grounding/Postprocess 阶段做 cProfile 基线；不能替代真实模型 profiler。
- `profile_realtime_model.py`：在单候选真实 TimeLens 请求上采集 `torch.profiler` CUDA/CPU 热点和阶段耗时；输出是优化前基线，不代表吞吐或质量。
- `eval/run_eval.py`：统一生成 `metrics.json`、`badcases.json`、`report.md`；若输入含 `actual_match` 和延迟字段，同时输出 No-Match 与 P50/P95 工程指标。
- `validate_validation_manifest.py`：校验 G8 JSONL 清单的字段、正负样本一致性、四类负样本覆盖和 pending 状态；当前项目不执行人工标注，严格负样本校准参数仅在未来扩大范围时使用。
- `run_service.py`：启动本地服务；Boundary、Ranking、Dedup 和 No-Match 均通过独立 CLI 参数注入，便于单模块消融和回滚。
- `pressure_steady_service.py`：向 loopback 服务回放固定间隔 steady 请求，记录 HTTP 成功、504/429、wall P95 和每条任务状态；不是 burst 或生产压测。
- `smoke_oom_recovery.py`：在隔离 CUDA 子进程中验证 OOM 捕获、`empty_cache` 和小张量恢复；不加载模型，不作为容量或生产稳定性结论。
- `service_contract_matrix.py`：执行 CPU stub 的成功、幂等、失败、超时降级、取消、队列满、OOM 后恢复、并发和线程清理矩阵；不代表真实模型吞吐或 GPU 资源释放。
- B1 的 full-video candidate 使用 `end_s=None`，由 adapter 单次读取视频并回传 fps/duration，避免重复打开 metadata reader；该优化不改变预测协议。
