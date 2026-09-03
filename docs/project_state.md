# Project State

Last updated: 2026-09-03

## Current Phase

**Offline prototype — Query-aware Coarse-to-Fine Temporal Grounding（当前离线自动化范围完成）**

本次路线替换后，TimeLens-8B 是主 Temporal Grounder；VideoITG-8B 降级为 baseline/参考。项目完成标准是离线数据、合成输入、接口契约和自动评测可复现；真实环境、真实用户和生产部署不属于完成条件。

## 2026-09-03 automated closure

- 已有 GPU/真实模型 profiler、单请求链路和异步 HTTP 记录作为可选运行时参考；固定 `SDPA + total_pixels=4,194,304` 的记录不构成真实环境复现或生产性能门槛。证据见 `records/phase_g9/realtime_model_profile_recovery_20260903.json`、`records/phase_g6/g6_realtime_smoke_recovery_20260903.json`、`records/phase_g6/g6_realtime_service_smoke_recovery_20260903.json`、`records/phase_g6/g6_realtime_service_timeout_1000ms_recovery_20260903.json`。
- TimeLens 真实失败/资源矩阵已补齐当前可自动验证范围：损坏媒体显式 failed；单 Worker 队列取消后首任务成功、取消任务不启动、线程和 CUDA 显存释放；双 GPU 两个真实 Worker 重叠约 19.3 秒，均成功且不降级，shutdown 后两卡各约 0.009 GiB allocated。记录见 `records/phase_g4/g4_timelens_decode_failure_recovery_20260903.json`、`records/phase_g6/g6_real_model_runtime_matrix_recovery_20260903.json`、`records/phase_g6/g6_real_model_two_worker_matrix_recovery_20260903_v3.json`。
- 真实 TimeLens 窗口矩阵已覆盖双窗口串行和真 batch；当前两次高视觉预算探针成功但没有复现 OOM，因此不能把它们写成 OOM 修复或容量结论。新增 `scripts/g4_timelens_runtime_matrix.py` 以独立子进程完成 9 个合法窗口、batch、视觉/生成预算和损坏媒体 case，并关联真实异步 TIMEOUT；修正越界窗口 harness 后 matrix 通过，失败和降级均保留原始记录。记录见 `records/phase_g4/g4_timelens_runtime_matrix_recovery_20260903_v2.json`。
- 现有前一项目数据已扩展到 360 个不同视频（360/360 存在、可解码且 frame/clip 标签通过审计），完成 cache miss 特征提取和 Retriever 对比：360/360 cache 有效、0 缺失，NPZ 16,259,649 bytes，提取耗时 mean/P50/P95=798.37/468.66/2429.58 ms；CachedCosine/Uniform Recall@5/10/20 分别为 0.7500/0.9389/0.9917 与 0.9861/0.9889/0.9944。Uniform 仅在该 target-present 观测集上更高且更快，主路线仍保留 query-aware CachedCosine；记录见 `records/phase_g2/local360_media_audit_20260903.json`、`records/phase_g2/local360_feature_cache_audit_20260903.json`、`records/phase_g3/g3_retriever_local360_20260903.json`。
- 在相同 360 个视频上完成固定配置 TimeLens 自动评测：360/360 有预测、0/360 Level-2 降级；R@1 IoU>=0.3/0.5/0.7=0.3056/0.2389/0.0778，mIoU=0.1959，平均边界误差=13.36s。该结果显著扩大了原 5 条诊断样本的运行证据，但仍是 `clip_num` 推导的 target-present weak-label 结果，不是正式质量 Gate；详见 `records/phase_g8/g4_local360_eval_20260903/report.md`。
- 本轮新增代码经过离线全量回归：79 passed、1 warning；最终发布审计为 117/117 required files、missing=0，契约检查全部通过。记录见 `records/phase9/release_audit_20260903_local360.json`。

已新增/完成的迁移基础：

- `docs/current_system.md`、`docs/current_architecture.md`：G0 审计输出。
- `preprocessing/feature_cache.py`：四字段版本键、元数据、NPZ 特征存储和 cache hit/miss。
- `configs/default.yaml`：默认视频侧编码器已同步为本地 `google/siglip-so400m-patch14-384`，缓存版本键为 `siglip-so400m-patch14-384-v1`；完整采样/验证集仍 pending。
- `retrieval/temporal.py`：`TemporalRetriever` 兼容的 cached-cosine 与 uniform 参考实现。
- `grounding/timelens.py`：`TemporalGrounder` 兼容的 TimeLens adapter；未配置 checkpoint 时显式失败。
- `pipeline/postprocess.py`：Boundary、Ranking、Temporal IoU Dedup、No-Match 和完整候选 schema。
- `pipeline/service.py`：粗到细编排和 Level 2 显式降级。
- 服务新增 `/tasks`、`/tasks/{id}/results` 兼容路径，以及 stage/progress/error/degraded 字段。
- 新增 `tests/test_coarse_to_fine.py` 与 `tests/test_badcase.py`，覆盖 Feature Cache 单次编码、采样预算/缓存身份、跨实例并发锁、raw/refined 边界保留、Boundary 时长夹断与方向性参数/非法区间、Temporal Dedup、TimeLens 解析/显式失败、空召回短路、Retriever ID 规范化、非有限时间/分数拒绝、Ranking、No-Match 分数语义/阈值校准（含空 predictions 负样本）、用户表面、Badcase taxonomy、验证集契约、synthetic 负样本标记、服务生命周期、异步超时/ watchdog、反馈字段（含成对边界校验）、canonical 反馈响应、Duplicate Rate、Retriever Query 清洗、同一视频第二 Query cache-hit/不重复编码、评测 YAML 加载、空输入完整指标 schema 和 lossless 边界评测、奇数尺寸候选物化；当前离线回归测试为 79 passed（1 warning）。
- 已下载官方 `TencentARC/TimeLens-8B` 到外置模型目录（4 个 safetensors 分片及索引，约 17 GB）；隔离环境的 Transformers 4.57.1 与当前 Miniconda Transformers 5.8.1 均完成 `AutoConfig`/`AutoProcessor` 本地加载核验，当前运行时额外安装 Decord/static-ffmpeg 并通过真实候选窗口推理 smoke。
- 当前 Miniconda 解释器下再次核验 TimeLens：4/4 safetensors 分片齐全，`AutoConfig.model_type=qwen3_vl`，`AutoProcessor=Qwen3VLProcessor`；系统 Python 的依赖探测不作为推理证据。
- 已新增 `preprocessing/feature_encoder.py`：本地 SigLIP 视觉特征编码和统一视频采样；在真实视频上完成双 Query cache smoke，首次编码后第二个 Query 命中缓存，`extractor_calls=1`。记录见 `records/phase_g2/g2_siglip_cache_20260831_222049.json`；该结果是最小 smoke，不代表完整验证集指标。
- 已新增 `scripts/evaluate_retrievers.py` 并完成 5 条媒体 pilot 的初始 CachedCosine/Uniform 对比及跨进程 cache-hit 复跑；结果见 `records/phase_g3/g3_retriever_compare_20260831_222401.json`、`records/phase_g3/g3_retriever_compare_20260831_222420.json`。随后扩展为 100 条 pilot，详见下条记录。
- 已完成 G3 100 条媒体 pilot（1 FPS、最多 16 帧、10 秒窗口）：CachedCosine Recall@5/10/20=0.76/0.94/0.99，Uniform=0.97/0.98/1.00；CachedCosine 平均检索约 19.6 ms，GPU peak allocated/reserved=1.65/1.76 GiB。记录见 `records/phase_g3/g3_retriever_compare_20260831_224512.json`，仅 target-present pilot，尚不据此宣称默认最优。
- 已补齐 Retriever 资源字段并在 5 条缓存编码样本复跑：CachedCosine 平均检索 16.43 ms、平均索引 44.7 KB；Uniform 平均检索 0.12 ms、索引 0 B；Recall@20 均为 1.0。记录见 `records/phase_g3/g3_retriever_compare_resource_20260831.json`，仍属于小规模 target-present 资源审计，不改变完整验证集选型 pending。
- 已完成 TimeLens 单候选、两个窗口串行和两个窗口真 batch 真实推理 smoke：候选结果均完成绝对时间映射并保存原始答案；真 batch 总耗时 27.4s，GPU 峰值 allocated 18.47 GiB、reserved 19.08 GiB。当前 SDPA/4M pixels 单窗口 smoke 也通过（约 22.6s，峰值 allocated 18.65 GiB）；记录见 `records/phase_g4/g4_timelens_smoke_20260831_223326.json`、`records/phase_g4/g4_timelens_smoke_20260831_223541.json`、`records/phase_g4/g4_timelens_smoke_20260831_224423.json`、`records/phase_g4/g4_timelens_smoke_20260831_225118.json`、`records/phase_g4/g4_timelens_smoke_20260901_003603.json`。质量校准、失败处理和端到端性能仍待验证。
- 已完成固定 5 条 target-present TimeLens pilot（复用缓存特征）：Top-1 R@1 IoU≥0.3=0.20；同一样本 Top-20/batch-2 的 coarse Recall@10=1.0，但 Top-3 IoU 命中=0，端到端约 91.5s。该结果说明当前 adapter 的可解析输出不等于定位质量，G4 质量 Gate 保持 pending；记录见 `records/phase_g4/g4_timelens_pilot.jsonl`、`records/phase_g4/g4_timelens_pilot_eval/`、`records/phase_g4/g4_timelens_top20_batch2.jsonl`。
- 已完成一次仅修改 Retriever Query 清洗的 G4 对照实验：剥离数据集附加答题指令后，5 条 pilot 的 Top-1 IoU 命中仍为 0.20，但平均 boundary error 从约 37.2s 降至 20.1s；未达到质量晋级条件，保留该清洗并继续冻结 Grounder。记录见 `records/phase_g4/g4_timelens_pilot_query_normalized.jsonl` 与 `records/phase_g4/g4_timelens_pilot_query_normalized_eval_v1/`。
- 已修正 G5 Ranking：`CandidateRanker` 现在显式纳入 boundary confidence 和 duplication penalty 输入，默认权重配置同步到 `configs/default.yaml`；排序单测和全量回归通过（当前 48 passed）。权重仍需验证集校准。
- 已完成 G5 后处理合成契约 smoke：Boundary 开/关消融均保留 raw/refined 字段，Temporal IoU Dedup 删除 1/3 重复候选，Ranking 候选记录保留 `duplication_penalty`，No-Match 输出 `CONFIDENT/POSSIBLE/NO_MATCH` 并计算 Accuracy/FPR/FNR。记录见 `records/phase_g5/g5_postprocess_smoke.json`、`records/phase_g5/g5_postprocess_smoke_20260831_2259.json`；该记录仅验证模块接口和字段，不代表真实质量或阈值校准。
- Pipeline 现在在 `candidates` 中保留 Dedup 前的全部候选，并用 `deduplicated` 标记被移除行；用户侧 `predictions` 仍只返回去重后的 Top-K，评测入口据此计算真实 Duplicate Rate。
- 全量诊断候选现在同时保留 `rank` 与 `pre_dedup_rank`，避免去重后的复制行出现空排名。
- 已完成 G6 单请求真实新链路 smoke：10 秒外置短视频经 SigLIP→CachedCosine→TimeLens→Boundary/Ranking/Dedup/No-Match 成功，`cache_hit=1`、`grounder_version=TimeLens-8B`，端到端约 23.1s；阶段回调记录 INDEXING/RETRIEVING/GROUNDING/POSTPROCESSING。记录见 `records/phase_g6/g6_realtime_smoke.json`；不代表并发吞吐或质量 Gate。
- 已完成 G6 真实异步 HTTP smoke：通过 `/tasks` 提交同一 10 秒视频，轮询到 `GROUNDING→SUCCESS` 并读取 `/tasks/{id}/results`，端到端约 27.1s；记录见 `records/phase_g6/g6_realtime_service_smoke.json`。仅验证单请求任务状态机，不代表并发吞吐或质量 Gate。
- 已修正当前环境的 TimeLens 本地部署兼容性：缺少 `flash-attn` 时自动使用 PyTorch SDPA；14.68M pixels 在 SDPA 下 OOM，降至 4,194,304 pixels 后真实异步 HTTP smoke 成功（约 20.7s、`degraded=false`、TimeLens-8B、绝对时间输出）。默认配置、服务参数和 profiler 已统一该预算；记录与失败探针见 `records/phase_g6/g6_realtime_service_smoke.json`、`records/phase_g6/g6_realtime_service_timeout_1000ms.json`。
- 已修正 G6 deadline 状态：同步请求超时现在持久化为 `TIMEOUT`/`error_code=TIMEOUT`，不会继续以 `running` 对外暴露；超时回归测试已更新并通过。
- 已补齐异步任务独立超时预算：`async_timeout_ms` 默认 120000，与同步 deadline 分离；watchdog 到预算即对外持久化 `TIMEOUT`，迟到 Grounder 结果再补写 coarse fallback，并新增回归覆盖。
- 已完成真实 1 秒异步超时探针：任务约 1 秒进入 `TIMEOUT`，约 25 秒后迟到 Grounder 结果补写 Level-2 coarse candidate，`/tasks/{id}/results` 可读；记录见 `records/phase_g6/g6_realtime_service_timeout_1000ms.json`。该结果说明 watchdog 正确工作，但 GPU 调用仍不可强杀，预算与最终资源释放时间需分开监控。
- 历史上一次 120 秒真实异步 smoke 曾在 `INDEXING` 阶段因宿主 CUDA 驱动无响应而中断：`nvidia-smi -L` 与 `torch.cuda.is_available()` 均在探测窗口内无返回。已停止重试并通过服务重启恢复将任务显式落为 `FAILED`/`SERVICE_RESTART_RECOVERY`，不覆盖此前成功 smoke；GPU 恢复后的真实异步成功/超时和双 Worker 记录见本文件 2026-09-03 收尾段。详细现象、排查和处理见 `records/phase_g2_g4_issue_log.md`。
- 后续健康检查发现 VS Code 的 `dolvin.vscode-nvidia-gpu-monitor-0.0.1` 在驱动失灵时每 2 秒无超时地累积 `nvidia-smi` 子进程（一次观察为 854 个），因此已停止新增 GPU 探测；该宿主环境问题及建议处理记录在 `records/phase_g9/gpu_driver_probe_20260901.json`，不归因于项目模型代码。
- 已补齐异步超时 Level-2 结果可见性：若 Grounder 在预算后返回候选，worker 将其转换为显式粗窗口降级结果；`/tasks/{id}/results` 对带 fallback 的 `TIMEOUT` 任务开放只读结果读取，并保留 `degrade_reason`。
- 已补齐服务媒体时长校验：`ffprobe`/static-ffmpeg 可探测时长超过 `max_video_duration_seconds` 时入口返回 413；配置和单测已接通。
- 已扩展统一评测入口：对带 `actual_match` 的输入计算 No-Match Accuracy/FPR/FNR，对可选延迟/显存字段计算 mean/P50/P95，并给无 IoU 命中样本保留 `badcase_type`（缺省为 `unclassified`）。合成入口验证输出见 `records/phase_g8/eval_smoke_output/`，不代表真实数据集质量。
- 统一评测现已补齐 R@1 IoU=0.7、mIoU、Mean Boundary Error、Top-K Useful Rate、用户采用率和人工边界调整秒数（有反馈字段时）；在同一合成输入上生成 `records/phase_g8/eval_smoke_output_v2/metrics.json`、`eval_smoke_output_v3/metrics.json`、`eval_smoke_output_v4/metrics.json`，仅作指标契约验证。
- 已用当前 TimeLens 五条 pilot 与一条 Top-20 记录复跑统一评测入口，确认真实输出字段可直接消费；五条 pilot 的 R@1 IoU≥0.3=0.20、mIoU≈0.111，Top-20 单样本的 Retriever Recall@20=1.0 但 Top-K grounding 命中为 0。结果见 `records/phase_g8/g4_pilot_eval_current/metrics.json` 和 `records/phase_g8/g4_top20_eval_current/metrics.json`，仍不是完整质量 Gate。
- 已新增 `badcase.taxonomy.BadcaseType`，统一 11 类 Badcase 标签；评测入口只接受枚举中的语义分类，未知标签保留为 `unclassified`，不强行归因到 Grounding。
- 已新增验证集 No-Match 网格校准函数；合成评测输入可输出校准阈值和指标，结果必须标记为 synthetic offline evidence，不解释为真实世界校准。
- 已新增 `evaluation.validation` 与 `scripts/validate_validation_manifest.py`，并提供 `records/phase_g8/validation_manifest.template.jsonl`、`scripts/build_validation_manifest.py` 和 79 条正样本 scaffold；校验器支持 `actual_match`、GT 一致性、四类 synthetic 负样本和 complete 标记检查。真实 No-Match 泛化仍不测，但 synthetic No-Match/Ranking 逻辑测试属于当前离线范围。
- 已修正 No-Match 用户表面：判定为 `NO_MATCH` 时不再强制返回 `predictions`，但所有 lossless 候选保留在 `candidates` 供诊断和校准；新增回归覆盖。
- 已更新本地前端任务卡片，展示 canonical stage 与 0–100% progress；候选预览、切换、边界调整和 8 类反馈流程保持不变。
- 已修正反馈持久化：反馈事件现在同时保存 `task_id/video_id/query/candidate_id/model_version/final_score`、`query_hash`、模型边界和用户边界；仍不写入视频内容或视频路径。
- 已新增逐条需求追踪表 `docs/requirements_traceability.md`，将每个新路线要求映射到实现、证据和 pending 状态，作为最终发布审计入口。
- 反馈兼容层现在将旧 lowercase 标签规范化为八类 canonical uppercase 后落盘；No-Match 时也可从诊断 `candidates` 关联候选边界，模型/用户边界保持独立。
- 已新增 G9 `scripts/profile_pipeline.py` 与 `scripts/profile_realtime_model.py`；前者记录编排 baseline，后者在真实 TimeLens 单候选上记录 CUDA/CPU 热点。真实 profile 端到端约 24.8s（Indexing 5.95s、Retrieval 21.7ms、Grounding 16.04s、Postprocess 2.44s，CUDA 峰值 19.26 GiB），记录见 `records/phase_g9/realtime_model_profile.json`；尚未据此修改模型参数。
- 已将 TimeLens `max_new_tokens` 与 `total_pixels` 纳入默认配置及 `run_service.py` 显式参数；默认值与已验证 smoke 路径保持一致，后续实验可单变量调整并记录。
- 当前 SDPA/4M pixels 的真实 profiler 结果为端到端 23.6s，CUDA 峰值 20.29 GiB；与旧 FlashAttention profile 分开记录，不宣称性能提升。记录见 `records/phase_g9/realtime_model_profile_sdpa_4m.json`。
- 已修正同一视频多 Query 的缓存可观测性：后续 Query 重新执行缓存键校验而不执行 extractor，因而既保持 `extractor_calls=1` 又记录 `cache_event=cache_hit`；新增服务回归覆盖。
- 已修正长视频采样预算未进入服务 Worker 的问题：`feature_sample_fps`/`feature_max_frames` 现在由 Worker 和 `run_service.py` 显式配置，写入采样缓存身份并传给 SigLIP `decode_and_encode`；默认保持 1 FPS、最多 16 帧。
- 已补齐 Feature Cache 跨 Worker 一致性：同一缓存键使用 POSIX 文件锁二次校验，多个 Worker/进程并发首次查询时只允许一个 extractor 写入，其余请求命中已写入特征。
- 已优化空召回路径：Retriever 没有候选窗口时 Pipeline 直接返回 `NO_MATCH`，不加载或调用 TimeLens，避免无事件查询触发无意义的 8B 初始化。
- 已统一反馈 API 输出：lowercase 输入仍兼容，但响应与持久化事件均返回 canonical 八类大写标签。
- 已收紧服务边界：`ServiceSettings` 对注入的 `frontend_origins` 执行 loopback-only 校验，未就绪文案使用 backend-neutral 的 model workers；同时补齐评测空输入的完整指标 schema（未测项为 `null`）。
- 已补齐 Retriever 窗口边界防护：Pipeline 为缺失/重复 candidate ID 生成稳定唯一 ID，CandidateWindow/GroundingPrediction 对非有限时间、分数和延迟显式拒绝，避免污染排序与 No-Match。
- 已完成 CPU Stub 异步压力矩阵：允许根目录内短视频 8 请求得到 6×200、2×429 overloaded、0 超时，wall P95 约 0.248s；首次误用仓库 fixture 的 400 已按路径安全预期记录，未放宽校验。证据见 `records/phase_g6/stub_pressure_8.json`，不外推为真实模型吞吐。
- 已修正 G8 No-Match 校准遗漏：显式 `predictions=[]` 的负样本现在从 `candidates`/`coarse_windows` 提取召回分数并进入阈值搜索；当前 4 条合成输入均参与校准，产物见 `records/phase_g8/eval_smoke_output_current_20260901/`，不代表正式验证集。
- 已修正统一评测入口忽略 YAML 配置的问题：`eval/run_eval.py` 现在读取 `output_top_k`、`retriever_top_n`、`duplicate_iou_threshold` 及数据/切分版本，并将生效配置写入 `metrics.json`。
- 已固定 G8 工程指标 schema：Decode/Feature/各阶段延迟、显存、Timeout/Failure/Degrade、用户采用率和人工调整秒数在缺少测量时显式输出 `null`，避免把“未测”误报为 0；同时提供 `failure_rate`/`degrade_rate` 规范别名。
- 已增强 `scripts/audit_release.py`：现在覆盖新路线 92 个关键文件（新增现有媒体/缓存审计、TimeLens adapter 合约矩阵、已有预测后处理消融和 CPU stub 服务矩阵），并执行候选 schema、生命周期/反馈枚举、独立后处理、loopback CORS、评测指标 schema 及 Decord wheel SHA256/WHEEL tag/运行库目录契约检查；`records/phase9/release_audit_20260901_final.json` 审计通过，人工标注和真实用户验收均为 out_of_scope。
- 已完成现有 100 条媒体池的 CPU 解码/标签审计：100/100 文件存在、可解码、1 FPS frame/5 秒 clip 标签均在边界内；记录见 `records/phase_g2/g2_media_pool_audit_20260901.json`。
- 已完成现有 100 份 SigLIP 缓存的只读审计：100/100 元数据和 NPZ 有效、无 manifest 缺失，16 帧/条，NPZ 总大小 4,513,344 bytes，平均 45,133.44 bytes；旧缓存未持久化特征提取耗时，因此该字段为 null。新缓存已开始持久化 `extraction_latency_ms`；记录见 `records/phase_g2/g2_feature_cache_audit_20260901_v2.json`。
- 已完成已有 TimeLens pilot/Top-20 预测的 Boundary、Ranking、Dedup replay 消融；结果分别无差异，且旧 pilot 的 raw 字段缺失已标记为 preview-bound proxy；记录见 `records/phase_g5/postprocess_ablation_timelens_pilot_20260901_v2.json` 与 `records/phase_g5/postprocess_ablation_timelens_top20_20260901_v2.json`。
- 已完成 CPU-only TimeLens adapter 合约矩阵和服务运行时矩阵；前者验证 batch/绝对时间映射/clamp/错误返回，后者验证成功、幂等、失败、TIMEOUT Level-2、取消、队列满、OOM 后恢复、双 worker 并发和线程清理；记录见 `records/phase_g4/g4_timelens_contract_matrix_20260901_v2.json`、`records/phase_g6/service_contract_matrix_20260901_v2.json`。两者均不替代真实模型质量、GPU 并发或显存证据。
- 已修正统一评测的单类别 No-Match 误报：仅 target-present pilot 不再输出 `no_match_accuracy/FPR/FNR` 数值，保持 null；回归见 `tests/test_metrics.py`，复跑见 `records/phase_g8/g4_pilot_eval_current_20260901_v2/metrics.json`。
- 已修正 OOM 故障路径：服务 worker 不再在异常处理线程中冷启动导入 torch，避免 CUDA cache 清理造成任务挂起；OOM 复现约 0.4s 完成且后续任务可继续执行。前端 TIMEOUT 也会继续轮询 results 端点，迟到 Level-2 fallback 可见；真实浏览器交互不在当前范围。
- 已修正降级反馈闭环：带迟到 Level-2 fallback 的 TIMEOUT 任务现在可以提交反馈，未完成或无结果的 TIMEOUT 仍被拒绝；模型边界与用户边界继续分开保存。
- 已补齐任务持久化审计字段：状态 JSONL 和 TaskResponse 现在保存 `model_version` 与持续更新的 `updated_at`，重启恢复后仍可追踪模型版本和最后更新时间。
- 已修正反馈模型边界语义：`model_start/model_end` 优先保存原始 `raw_start/raw_end`，Boundary Refinement 和用户调整分别保留，不覆盖模型原始预测。
- 已补齐运行时配置注入：`run_service.py` 可独立配置/关闭 Boundary Refinement、设置 Ranking 权重、Dedup IoU 阈值和 No-Match 阈值，`CoarseToFineWorker` 通过依赖注入保持模块解耦。
- 依赖审计发现上游 `decord 0.6.0` 的 PyPI wheel tag 为 `cp36`；已用 FFmpeg 6.1.1 + Decord 源码编译，并通过 `auditwheel` 打包完整 runtime 依赖。当前 Miniconda 使用 cp313 wheel，`pip check`、import、`ldd` 和 `VideoReader(imax.mp4)` 均通过；wheel 标签为 manylinux_2_39，低于该基线的部署需在对应构建镜像重打包。证据见 `records/phase_g9/decord_build_20260901.json`。
- 已补齐后处理配置数值校验：Ranking 权重必须为有限非负值，No-Match retrieval/grounding 阈值限制在 `[0,1]`，margin 阈值必须非负，非法启动配置会显式失败。
- 已修正运行中取消语义：DELETE 设置取消后，worker 自然结束也不会再覆盖为 SUCCESS，而是落盘 `CANCELLED` 并丢弃迟到结果。
- 本轮依赖、段错误、设备错配、长 Query、OOM 超时竞态及其修复流程统一记录于 `records/phase_g2_g4_issue_log.md`。

当前只完成：

- 创建项目目录和外置数据目录；
- 固定项目范围、两条亮点和分阶段 Gate；
- 创建新会话恢复用 `AGENTS.md`；
- 创建详细执行计划、架构说明和初始配置。
- 核验官方代码仓库 `NVlabs/VideoITG` 和 checkpoint `nvidia/VideoITG-8B` 的入口、文件结构与许可提示。
- 已下载 VideoITG 官方代码、4 个模型 safetensors shard、配置/tokenizer 文件。
- 已下载 `VideoITG-40K/video_itg_data.json`，结构校验通过，共 474,354 条记录。
- 已下载官方 `assets/imax.mp4` 作为 smoke-test 原始视频；`file` 识别为 ISO MP4。
- 已在隔离环境 `envs/videoitg312` 安装并验证最小运行时依赖，包括 PyTorch 2.6.0+cu124、Transformers 4.47.1、Decord、PyAV、以及上游 loader 强制需要的 `flash-attn==2.7.4.post1`。
- 已缓存 checkpoint 所需的外部 SigLIP `google/siglip-so400m-patch14-384`。
- 已用 `scripts/phase0_smoke.py` 完成 32 帧单视频端到端推理，保存 `outputs/phase0_smoke_20260714_0015/result.json`；32 帧峰值 CUDA 分配 16.56 GiB、总耗时 33.99 s。
- 已验证官方默认 512 帧在单张 24 GiB RTX 4090D 上 OOM；详见 `docs/upstream_audit.md` 和 `outputs/phase0_smoke_20260714_0000/smoke.log`。
- 已用 `imageio-ffmpeg` 提供的静态 FFmpeg 和 PyAV 补齐 `imax.mp4` 探测：H.264、1280×720、23.976 FPS、9,514 帧、396.81 s。
- 已在 `envs/videoitg312` 安装 `static-ffmpeg`，并验证环境内静态 `ffprobe` 可独立探测 `imax.mp4`（396.829 s、23,297,725 bytes）。
- 已审计 VideoITG-40K 全量元数据：474,354 条记录、42,519 个唯一视频路径，schema/重复/非负索引检查通过；结果见 `docs/data_audit.md` 和 `records/phase1/videoitg40k_audit.json`。
- 已生成按 video-level hash 隔离的内部 train/dev/test split 和 50 视频真实媒体 pilot（2_3_m_nextqa 10 条、30_60_s_nextqa 20 条、30_60_s_activitynetqa 20 条）；50/50 解码与 1 FPS/5 秒索引检查通过，10 条多帧 spot check 与问答相容，三位置近重复初筛无候选。
- 已实现 `VideoITGReranker` 候选片段 adapter，并在 `imax.mp4` 两个 12 秒候选上完成真实模型调用；结果见 `outputs/phase1_adapter_smoke_20260714_0100/result.json`。
- 已为 adapter 和评测指标添加最小单测，`PYTHONPATH=src pytest` 结果为 4 passed；G1 条件逐项记录见 `records/phase1/g1_checklist.md`。
- 已建立困难集分类协议和 50 条 review-only 候选；所有困难标签仍为 `pending_manual`，未用于训练或指标。
- 已记录范围修正：`records/plan_changes.md`；Phase 2 只报告 target-present temporal grounding，无匹配、ASR/视觉冲突和短事件边界延期到人工补标。
- 已实现统一评测入口 `scripts/run_baseline.py`，固定 5 秒候选、`clip_num`→5 秒 GT、Top-k=3、VideoITG `max_frames=16`；B0/B1/B2 smoke 和 50 条完整 pilot 均已完成。
- 已补跑 B2-Retrieval-only（B2R）Top-4 候选覆盖基线；B2R Top-k 命中 IoU≥0.3=0.66，而 B2 Top-3=0.54，暂时表明当前重排排序存在候选覆盖损失。
- B0/B1/B2/B2R 逐样本与汇总证据保存在外置输出目录，比较记录见 `records/phase2/baseline_comparison.json` 和 `docs/baselines.md`；pilot 上 B1 的 Recall@1 IoU≥0.3 为 0.58，B0 为 0.28，B2 为 0.24。
- 已生成 `records/phase3/b1_badcase_report.json`；B1 Top-1 IoU 分布为 IoU≥0.5 共 25 条、0.3–0.5 共 4 条、<0.3 共 21 条，其中 6 条出现 B2 相对 B2R 的候选命中回退。该报告只做指标分层，尚未赋予语义 badcase 标签。
- 已为 10 条优先样本导出 GT/B1/B2R/B2 密集关键帧和查询证据，并生成 `records/phase3/manual_badcase_review.json`；暂定 2 条 short-event miss、2 条 boundary shift、6 条 other/uncertain。该复核仍是 provisional，连续视频回放尚未完成。
- 已生成 `records/phase3/regression_pool.jsonl` 和 `records/phase3/b1_regression_report.json`；4 条样本仅作为 pilot regression-only、明确 `not_for_training=true`，可用 `scripts/evaluate_regression_pool.py` 一键复评。
- 已从 pilot `test` split 生成 `records/phase3/heldout_regression_pool.jsonl` 和 `records/phase3/b1_heldout_regression_report.json`；4 条均可评估，但因样本少且标签 provisional，G3 仅条件通过，清单见 `records/phase3/g3_checklist.md`。
- 已完成 B1 8 秒输出区间候选的完整 pilot 和两套 regression pool 评估；结果见 `records/phase3/segment8_candidate.json`，由于不同池表现不一致，尚未替换 5 秒默认。
- 已将 test split 扩展为 7 条 held-out regression pool（4 条 provisional badcase + 3 条 metric control），最终清单为 `records/phase3/heldout_split_regression_pool_final.jsonl`；B1 与 8 秒候选均完成一键评估，结果见 `records/phase3/b1_heldout_final_regression_report.json` 和 `records/phase3/b1_seg8_heldout_final_regression_report.json`。
- 已按 source group/split 汇总 B0/B1/B2，见 `records/phase3/group_metrics.json`；B1 的 Recall@1@0.3 在 ActivityNet 组为 0.70、30–60s NExTQA 为 0.65、2–3m NExTQA 为 0.20，优势并非各组均匀。
- 已实现并重复运行 B1A 自适应区间候选（5 秒→低分时 8 秒，阈值 0.6），两次 50 条逐样本输出完全一致；结果见 `records/phase4/b1a_candidate.json`。pilot Top-k@0.5=0.72、held-out test=0.714，已作为 Phase 4 candidate，但尚未提升为默认。
- 已在排除 3 个 near-duplicate 候选后的 97 条 clean manifest 上完成 B1 5 秒、B1 8 秒和 B1A 比较；18 条 test split 上 B1A Top-k@0.5=0.667（B1=0.556），扩展证据见 `records/phase4/clean97_candidate_comparison.json`。
- 已完成 clean97 test 的语义复核收口：最终 18 条回归池中 8 条完成人工 sampled-2s full-video scan（4 条 boundary_shift、1 条 short_event_miss、3 条 other_or_uncertain），10 条保留为 metric-only controls；最终池和复核记录见 `records/phase3/clean97_test_regression_pool_final.jsonl`、`records/phase3/clean97_test_manual_review.json`。G3 在该明确 scope 下条件通过，连续视频播放不作声明。
- 已完成 G4 候选审计，见 `records/phase4/g4_candidate_audit.json`：B1A 在 clean97 test Top-k@0.5 比 B1 高 0.111，但 confirmed boundary-shift 类别无提升、short-event 仍为 0；8 秒策略在 test 高 0.222，却在普通 50 条 pilot 低 0.02。因此 G4 未通过，B1 继续作为默认，B1A/8 秒保留为候选。
- 已完成 train-side 4 条 provisional badcase 的 sampled-2s full-video 复核（2 条 short_event_miss、2 条 boundary_shift）；记录见 `records/phase4/train_badcase_manual_review.json`。它们只冻结为 Phase 4 候选池，尚未用于训练，详见 `records/phase4/targeted_badcase_pool_audit.json`。
- 已从上述样本生成 6 个相邻同视频 temporal hard-negative 候选，并复核物化其中 5 个（3 个 adjacent non-target、2 个 boundary-ranking hard negative）；1 个 same-event 候选被排除。全部仍 `not_for_training=true`，记录见 `records/phase4/temporal_hard_negative_pool.jsonl`。
- 已直接对上述 5 个候选与 GT 区间做 VideoITG segment-score 对照：5/5 负例分数低于 GT（差值约 −0.007 至 −0.049），因此它们尚未形成可训练的 ranking inversion，继续保持 `not_for_training=true`。记录见 `records/phase4/temporal_hard_negative_score_audit.json`。
- 已从 4 条人工确认 train badcase 物化 B1 全视频高分错误峰为 frame-level 诊断池（2 条 short-event、2 条 boundary-shift）；候选分数最高达 0.979，但仍全部 `not_for_training=true`，不赋予 no-target 语义。记录见 `records/phase4/frame_hard_negative_pool.jsonl`。
- 已实跑并重复 B1 `max_frames=32`：50 条 pilot 逐样本完全一致，普通 Top-k@0.5=0.76（16 帧=0.68）；97 条 clean scope Top-k@0.5=0.722（16 帧=0.649）；最终 test 的 boundary-shift Top-k@0.5=1.0（16 帧=0.25），但 10 条 controls 从 0.9 降到 0.8，平均精排时间增加约 29%。审计见 `records/phase4/mf32_candidate_audit.json`，暂不提升默认。
- 已完成 B1C 直接候选（`max_frames=32`、8 秒输出区间）的 pilot、clean97、held-out、最终 18 条回归池和 50 条重复运行；预测逐样本完全一致。它改善普通 pilot（Top-k@0.5=0.74）和 clean97 overall（0.711），但历史 7 条 held-out 不变（0.571），最终 controls 从 0.9 降至 0.8，平均精排时间约增加 29%，因此 G4 仍未通过。直接运行审计见 `records/phase4/b1c_candidate_audit.json`；离线区间重映射结果不作为 gate 证据。
- 已完成 B1D pilot（32 帧、5 秒/低分时 8 秒）：仅 2/50 条选择 8 秒，Top-k@0.5 与固定 32 帧/5 秒相同（0.76），没有可见质量收益，未扩展到 clean97 或 held-out。审计见 `records/phase4/b1d_candidate_audit.json`。
- 已直接验证 32 帧/6 秒中间区间：pilot Top-k@0.5=0.70，低于 32 帧/5 秒的 0.76，已丢弃，记录见 `records/phase4/mf32_seg6_candidate_audit.json`。
- 已完成 48 帧/5 秒的 10 条显存与质量 profile：单卡运行未 OOM，但 Top-k@0.3=0.70、Top-k@0.5=0.70，未优于同 10 条的 32 帧候选，且边界误差更高，未扩展到完整 pilot；记录见 `records/phase4/mf48_profile10_audit.json`。
- 已完成 B1E 局部 zoom 原型：全视频定位后对 top-1 附近 10 秒窗口再调用一次 VideoITG；10 条 pilot profile 无增益，5 条确认 train/test badcase 均未恢复，且每条样本需要两次模型调用，已丢弃。审计见 `records/phase4/b1e_candidate_audit.json`。
- 已完成 B1 `max_frames=64` 的 pilot、重复运行、clean97 和固定回归池评估：pilot Top-k@0.5=0.78、clean97 overall=0.732；最终 18 条中 boundary-shift=0.5、short-event=1.0、controls=1.0。历史 7 条 held-out aggregate 不变为 0.571，平均重排时间约增加 70%，因此它是当前最强候选但仍不替换 B1 默认。审计见 `records/phase4/mf64_candidate_audit.json`。
- 已新增 27 条未进入 clean97、按 video-level hash 落在 test 的媒体 held-out，并完成 B1 16/64 帧配对评估：Top-k@0.5 从 0.778 提升到 0.815，但 paired gains/losses 同时存在，ActivityNet 与 NExTQA 方向相反，平均延迟增加约 41%；因此没有形成稳定的 G4 晋级证据。记录见 `records/phase4/heldout_extra27_manifest.jsonl`、`records/phase4/heldout_extra27_mf64_audit.json`。
- 已完成该 27 条 held-out 的 sampled-2s 全视频人工复核：7 条 boundary_shift、18 条 control_correct、2 条 other_or_uncertain；64 帧在 boundary_shift 类别 Top-k@0.5 与 16 帧均为 0.5714，没有改善目标错误类别。最终不可训练回归池和复核记录见 `records/phase4/heldout_extra27_regression_pool.jsonl`、`records/phase4/heldout_extra27_manual_review.json`。
- 已在同一 27 条人工复核池直接测试 16 帧/8 秒区间：整体 Top-k@0.5 从 0.778 降至 0.741，boundary_shift 从 0.571 降至 0.429，已丢弃。记录见 `records/phase4/heldout_extra27_seg8_audit.json`。
- 已按 train/dev 预注册检查 32→64 帧 confidence fallback：71 条 train 中 64 帧相对 32 帧的 Top-k@0.5 改善与退化各 9 条，8 条 dev 没有变化，当前可观测 top-1 分数/间隔不足以支持稳定门控规则；未用 test 调阈值。记录见 `records/phase4/mf64_policy_audit.json`。
- 已完成一轮不重新训练基础模型的 boundary-aware 时序后处理原型：在 train/dev 上预注册的“下一采样帧分数 beta=0.02”候选使 dev Top-k@0.5 从 0.75 升至 0.875、train 不退化；但在未参与选择的 final18 与 additional heldout27 上总体指标及人工确认的 boundary/control 类别均完全不变，候选已淘汰。记录见 `records/phase4/boundary_postprocess_train_dev_grid.json`、`records/phase4/boundary_final18_asym_beta002_report.json`、`records/phase4/boundary_extra27_asym_beta002_report.json`。
- 已完成 boundary head v1：只用 train 的 71 条 frame-score/GT 训练 CPU ridge 校准器，冻结到 final18 与 additional heldout27 后，boundary_shift Top-k@0.5 分别从 0.25→0.75、0.571→0.857；普通 50 条 pilot Top-k@0.5 保持 0.68，controls 不退化，单条后处理约 0.29 ms。该候选仅在 boundary_shift scope 晋级；唯一 short_event_miss 仍为 0，未宣称短事件修复。记录见 `records/phase4/boundary_head_v1_train_only.json`、`records/phase4/boundary_head_v1_final18_report.json`、`records/phase4/boundary_head_v1_extra27_report.json`。
- 已完成 24 帧质量候选的 pilot、clean97、final18 子池和 additional heldout27 回归：普通 pilot Top-k@0.5 保持 0.68；clean97 overall 0.649→0.701，final18 0.556→0.722 且 confirmed short_event_miss 0→1、controls 0.9→0.9；additional heldout27 overall 0.778→0.852，boundary_shift 0.571→0.714、controls 0.833→0.889。该候选改善质量但精排延迟增加约 10–15%，因此 G4 在质量 scope 条件通过，固定16仍是默认，不能用于宣称 G5 通过。审计见 `records/phase4/mf24_candidate_audit.json`。
- 已淘汰 target_fps=4/16 帧采样相位候选：在 2 条 confirmed short-event 和 2 条 boundary-shift train badcase 上均未产生 Top-k@0.5 修复，因此未扩展完整 pilot。记录见 `records/phase4/targetfps4_candidate_audit.json`。
- 已用 boundary-head 特征检查 16→64 帧 short-event fallback：train 中 64 帧相对 16 帧 Top-k@0.5 改善 10、退化 5，dev 8 条全部不变，无法校准选择性门控；记录见 `records/phase4/mf16_boundary_head_policy_audit.json`，不使用 test 调阈值。
- 已完成 Phase 5 离线 latency preflight：16/32/64 帧真实 pilot 的 P50/P95 重排时间分别约 2.04/3.59 s、2.64/4.79 s、3.25/6.97 s；建立了显式可回滚的 `BudgetPolicy` 纯函数和默认 fixed 配置，但尚未做并发/队列回放，G5 尚未开始。记录见 `records/phase5/latency_profile_preflight.json`。
- 已完成基于上述真实逐样本运行时的 12 场景离线请求回放（1/2/4 worker、burst/steady、E0/E2）；结果仅为 queueing simulation。未校准的 E2 在 burst 下出现更多高帧选择和超时，不能作为策略结论，也不代表真实服务吞吐。记录见 `records/phase5/latency_policy_replay_simulation.json`。
- 已补做 50 条单 GPU 真实请求基准，覆盖 E0 fixed16 与 E3 queue-bypass 的 burst/1 秒 steady 两种到达模式：E3 的 service P95 约降低 21%，但 49/50 条降为 retrieval-only，Top-k@0.5 从 0.68 降到 0.54，且 50/50 请求仍超出 5 秒 deadline；因此 G5 仍未通过，E0 保留为默认。记录见 `records/phase5/real_budget_benchmark_audit.json`、`records/phase5/g5_candidate_audit.json`。
- 已补测固定 8/12 帧 E1 候选：12 帧 P95 与 16 帧基本相同且 Top-k@0.5 下降 0.02；8 帧 P95 下降约 25% 但 Top-k@0.5 下降 0.22，均淘汰。记录见 `records/phase5/e1_frame_budget_audit.json`。
- 已完成 E0 的 CUDA allocator 微优化对照：同一 GPU、同一 50 条 burst、预热和 16 帧预算下，跳过每条请求后的 `empty_cache` 保持 Top-k@0.5=0.68，但 service P95 为 3522.7 ms，略高于默认清理的 3519.0 ms；因此不构成 G5 延迟候选。开关和结果见 `scripts/benchmark_budget_policy.py --retain-cuda-cache`、`records/phase5/no_empty_cache_audit.json`。
- 已完成 full-video single-reader 优化：让 adapter 在一次 `VideoReader` 中同时获得 fps/duration，去掉 B1/benchmark 调用前的重复 metadata reader；5 条 smoke 预测逐样本等价，50 条 burst 的 Top-k@0.5 均为 0.68，mean/P50 service 分别下降约 1.4%/1.6%，但 P95 上升约 1.1% 且仍约 3.63 s，因此只保留为实现清理，不视为 G5 通过。审计见 `records/phase5/single_reader_audit.json`。
- 已完成 CPU full-video frame prefetch 对照：在固定 16 帧预算下，将下一请求的解码与当前 GPU 精排重叠；50 条 burst/1 秒 steady 的 Top-k@0.5 均保持 0.68，service P95 分别降至约 2.02/1.95 s（相对旧 E0 下降约 42.8%/45.7%），但仍有 48/50 和 44/50 条超过 5 秒 deadline，队列尾延迟未被消除。因此仅保留为 opt-in pipeline 优化，G5 仍不通过。审计见 `records/phase5/prefetch_audit.json`。
- 已补做 prefetch 的容量边界探针：20 条、1.5 秒到达间隔下 P95 total 约 4.38 s、0/20 超时；该结果只说明当前单卡在较低到达率下可清空队列，不改变 50 条 burst/steady-1s 的 G5 结论。
- 已完成两张本地 GPU 的共享队列容量实验：每卡一个 fixed16 worker，并加入一条 lookahead CPU 预取；steady-1s 的 50 条请求 0/50 超时、Top-k@0.5=0.68，但 burst 仍有 46/50 超时、P95 total 约 39.9 s。因此多卡结果只证明 steady-load 容量改善，不能作为 burst-safe G5 或服务实现证据。审计见 `records/phase5/multi_gpu_audit.json`。
- 已做真实两请求同卡 microbatch smoke：变长 query 会触发上游 FlashAttention index 错误；相同 token 长度的 2×16 帧 batch 可运行（峰值 18.84 GiB、逐帧差异 <0.001），但耗时 18.22 s，显著慢于两次单请求 4.61 s；因此只保留受限 rank_batch 正确性实验，淘汰为 burst 延迟方案。记录见 `records/phase5/batch_smoke_audit.json`。
- 已对双 GPU burst 做显式 admission 分析：5 秒内仅 4/50 条完成，若拒绝其余请求则覆盖率损失 92%；因此不能把 admission 作为质量保持的 G5 通过方案，记录见 `records/phase5/admission_audit.json`。
- 已按真实容量证据拆分 G5：G5-A 为双 GPU、50 条 steady-1s、5 秒 deadline，实测 0/50 timeout、P95 total 约 4.39 s、Top-k@0.5=0.68，在该限定 scope 下通过；G5-B 为 50 条同步 burst，46/50 timeout，明确 unsupported。计划变更见 `records/plan_changes.md`。
- 已实现受限 Phase 6 服务骨架：FastAPI loopback 网关、steady 同步 endpoint、burst 有界异步任务、幂等 request_id、状态查询、取消、429 overload、上传/路径校验、health/readiness、模型/策略版本字段与优先队列；stub 独立进程 smoke 通过。记录见 `docs/service.md`、`records/phase6_service_smoke.json`。
- 已完成真实 VideoITG-8B 双 GPU service smoke：cuda:0/cuda:1 均加载成功，healthz 显示 2 个 loaded workers，单条 steady 请求 200 成功，端到端约 2.985 s（model 1.380 s、读取 1.328 s、峰值 16.45 GiB），进程干净关闭；这不是 steady 压测或生产验证。
- 已将 admission-time CPU frame prefetch 接入服务，并完成真实双 GPU steady-1s 压测：50/50 返回 200、0 timeout、wall P95 约 3.11 s；记录见 `records/phase6_steady_pressure.json`。该结果不外推到同步 burst。
- 已完成 22 条服务契约/安全单测；路径、格式、大小、幂等、取消、overload、后端失败、deadline timeout、synthetic OOM recovery、未加载 readiness 和终态/中断重启恢复均有证据。安全矩阵见 `records/phase6_safety_matrix.json`。
- 已补齐 Phase 7 最小本地前端：上传、异步任务状态轮询、本地视频预览、候选区间选择、边界调整字段和结构化反馈；前端说明见 `frontend/README.md`，交付清单见 `docs/release_checklist.md`。
- 已新增 `/v1/feedback` 版本化反馈接口，反馈 JSONL 记录 task/video/candidate/query/model/final score、标签和可选边界调整，不写入视频内容或视频路径；反馈字段回归已通过。
- 已完成 8 次自动化工作流协议验证（上传→任务→轮询→反馈），记录见 `records/phase8/local_workflow_protocol.json`；该记录明确 `human_user_test=false`，不作为 G8 用户验证或 VideoITG 质量证据。
- 代码修复后的本地 Stub 工作流复核完成 2/2（任务→轮询→反馈），记录见 `records/phase8/local_workflow_protocol_latest_20260901.json`；仍明确 `human_user_test=false`，仅用于接口回归。
- 已补齐静态前端（8080）到 API（8000）的显式 loopback CORS，并完成 Playwright headless 上传→异步轮询→候选选择→边界调整→反馈 smoke；记录见 `records/phase8/g7_browser_smoke_20260901.json`。该自动化记录仅作为接口回归，真实用户验收不在项目范围。
- 已保留 20 次操作协议作为自动化接口回归参考，见 `records/phase8/user_test_protocol.md`；真实参与者验收不在当前范围。
- 已完成 22 次用户测试协议操作（10 条查询×2 轮 + 2 条额外），覆盖全部 8 个反馈标签；记录见 `records/phase8/user_test_execution.json`；该记录使用 stub backend 通过 REST API 执行，`human_user_test=false`，不作为 G8 用户验证或 VideoITG 质量证据。
- 已完成 clean97 的 B0 cheap-signal 检查：CLIP Top-1 分数/Top-1−Top-2 margin 在 train/dev 上无法形成选择性 16→64 帧门控，已丢弃该策略，未用 test 调阈值。记录见 `records/phase5/b0_cheap_signal_audit.json`。
- 已完成 B1 小规模 profiling 以及 32 帧 full-pilot/clean97 验证，结果见 `records/phase2/b1_profiling.json`；当前仍冻结 `max_frames=16`、5 秒输出区间，32 帧作为质量更高但约 29% 更慢且 controls 有退化的 Phase 4 候选。
- 已完成 B0/B1/B2 各 50 条的重复运行/逐样本方差审计，结果见 `records/phase2/repeatability_audit.json`；三组预测与指标均完全一致，G2 的重复性条件已满足。

当前仍未完成或无法完成的边界：

- TimeLens 离线质量基线：360 个不同视频上的 `R@1 IoU>=0.3=0.3056`、mIoU=0.1959 已形成当前 weak-label 自动基线；由于标签来自 `clip_num` 且 `top_n=1`，不能外推为真实世界质量，但这不阻塞离线原型完成。
- 合成负样本与 No-Match：允许使用 `event_absent`、`wrong_action`、`wrong_object`、`theme_unrelated` 四类 synthetic 数据测试状态、阈值、Ranking 和降级逻辑；合成指标必须标注 `synthetic`，不解释为真实世界 FPR/FNR。
- 真实用户采用率、人工边界调整、真实环境复现、生产吞吐和跨环境打包：均不属于本离线项目完成条件；已有真实运行时和 Decord 记录只作为可选参考。

## Current Gate

离线原型的 G0–G9 自动化范围已完成：G2 SigLIP cache、G3 Retriever 对比、G4 TimeLens 适配与评测、G5 后处理/合成负样本契约、G6 任务状态与降级、G8 统一离线评测均有独立记录。当前离线全量回归 79 passed（1 warning）；最新发布审计 117/117、missing=0 且契约检查全部通过。360 条真实数据结果和 synthetic 负样本结果均按各自数据类型解释，不外推为生产结论。

G1 已条件通过；新路线 G2–G9 的离线自动化证据已完成，包括 360 条 target-present 媒体池 Retriever/cache/TimeLens 统计、synthetic 四类负样本支持、后处理消融、任务状态机、失败/超时降级和统一评测。真实 profiler、真实异步 HTTP、双 GPU 和 Decord artifact 作为可选参考保留。旧 G2–G7 的 VideoITG 结果不能迁移为 TimeLens 主链路结论；ASR/视觉冲突仍不在当前 scope。

## Next Actions

1. 使用 360 条 target-present weak-label 数据和四类 synthetic 负样本继续做离线回归、消融和阈值逻辑测试。
2. 真实负样本、真实用户、生产吞吐和更老系统兼容性不在当前离线项目计划内；只有项目范围改变时才另开验证。

## Current-data completion boundary

2026-09-03 已完成离线项目可支持的自动化收尾：全量回归 79 passed、release audit 117/117、360 条媒体解码/标签审计、360 条新缓存覆盖/提取耗时统计、Retriever 对比、360 条固定配置 TimeLens 自动评测、synthetic 四类负样本契约、已有 TimeLens 运行时参考、后处理 replay 和 adapter/服务契约矩阵均已归档。机器可读汇总见 `records/phase_g8/current_data_completion_20260903.json`。

人工标注和真实浏览器用户验收已从范围中移除。自动离线验证、synthetic 负样本规则及“现象→可能问题→排查→处理”矩阵见 `docs/real_acceptance_plan.md`。当前 360 条媒体池是 target-present weak-label 诊断；四类 synthetic 负样本可用于离线逻辑测试，真实世界 No-Match 泛化不测。GPU profiler/真实运行时 smoke 仅为可选参考。

## New conversation handoff

本项目当前交付范围是“可离线复现的长视频片段检索工程原型 + 自动化验证结果”。主链路为 SigLIP Feature Cache → Query-aware Retriever → TimeLens-8B → Boundary/Ranking/Dedup/No-Match → 异步任务与反馈接口；VideoITG-8B 仅作 baseline。人工标注、真实浏览器用户验收、真实环境复现和生产部署已明确 out_of_scope；四类 synthetic 负样本属于离线测试范围。

已验证：离线全量回归 79 passed（1 warning）；release audit 117/117 pass；360 条媒体审计、cache/Retriever 统计及固定配置 TimeLens 自动评测；synthetic 四类负样本清单支持；双 Query cache smoke 的 `extractor_calls=1`；TimeLens 9-case 参数矩阵、profiler、单/双窗口、真 batch、decode failure、async timeout fallback、单 Worker cancel/cleanup、双 GPU Worker overlap/cleanup；后处理 replay；Decord cp313 runtime artifact。主要证据索引见 `records/phase_g8/current_data_completion_20260903.json` 和本轮 `records/phase_g2/`、`records/phase_g3/`、`records/phase_g4/`、`records/phase_g6/`、`records/phase_g9/` 记录。

当前仍有限制：360 条 TimeLens 结果是 target-present weak-label、`top_n=1` 的离线基线；synthetic 负样本只能说明逻辑在构造场景下的行为，不能证明真实世界泛化。项目不要求生产吞吐或真实环境复现；已有 GPU 记录仅作参考，仍禁止重复调用会阻塞的 `nvidia-smi`/CUDA 探测。

新会话启动顺序：先读 `AGENTS.md`、本文件和 `plan/execution_plan.md`；只执行自动离线/接口/运行时验证；每次只改一个主要变量；结果写入新的 `run_id`，同步问题日志和状态文档。

## Active Runs

None.

## Verified Metrics

Download validation: passed (TimeLens-8B four-shard index and local config/processor).
New-route evidence: G2 SigLIP two-Query cache smoke passed (`extractor_calls=1`); G3 360-row target-present CachedCosine Recall@5/10/20=0.7500/0.9389/0.9917 vs Uniform=0.9861/0.9889/0.9944; G4 360-row fixed-config TimeLens run R@1 IoU>=0.3/0.5/0.7=0.3056/0.2389/0.0778, mIoU=0.1959, mean boundary error=13.36s, 0/360 degraded; one/two-window serial and true-batch smoke passed; G6 real single-request chain passed (23.1s, cache hit). These are smoke/weak-label pilot measurements, not production or formal quality claims.
Model inference metrics: 32-frame smoke passed; adapter 2-candidate smoke passed; 512-frame single-GPU OOM. Phase 2 pilot: B0/B1/B2 each 50 samples completed; B1 Recall@1 IoU≥0.3=0.58, B0=0.28, B2=0.24. B1 profiling, repeatability, 7-row held-out regression, 97-row clean-scope candidate evaluation, and 18-row final sampled-2s regression review are complete; G2 passed on the pilot and G3 is conditionally passed within the documented scope.

## Known Risks

- VideoITG 上游已在当前环境完成 32 帧推理，但官方 512 帧默认输入在单卡上 OOM。
- VideoITG-40K 目前只落地了 3 个小分片和 50 条媒体 pilot，42,154 个唯一媒体路径仍缺失；不能据此代表完整数据集进行效果评测或训练。
- VideoITG-40K/LLaVA-Video 数据卡允许研究/教育用途；商业再分发范围未获授权。
- 原始模型是否适合作为候选片段精排器尚需接口与效果验证。
- 公开时序定位数据能否满足“长视频、多片段、无匹配、短事件”覆盖尚需数据审计。
- 3×4090 是可用资源假设，不等于模型一定能按计划训练或服务；需以实测显存和吞吐为准。
- 当前 8 条人工复核 held-out 样本使用 sampled-2s full-video scan，不能等同连续播放；10 条 controls 没有语义标签。不得报告超出该 scope 的 G3/G4 泛化结论。
