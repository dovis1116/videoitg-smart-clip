# Baselines and retriever comparison protocol

本文件记录迁移后的 baseline 角色。VideoITG-8B 只作为历史/兼容 baseline；主方案是 Query-aware Retriever + TimeLens-8B。当前已有的 VideoITG 数字仅覆盖已下载并通过媒体审计的 50 条样本，不代表全量数据，也不能替代 TimeLens 结果。

## 统一输入与标签

- manifest：`records/phase1/videoitg40k_media_pilot.jsonl`，50 条真实视频；运行前要求 `video_path` 存在。
- 查询：manifest 中的 `query` 字段。
- GT：上游 `clip_num` 按 5 秒一个 clip 转换为 `[start_s, end_s]`；这是对当前上游 1 FPS/5 秒索引格式的显式评测约定，不是人工重新标注的精确边界。
- 候选：统一使用 5 秒非重叠区间，最多 64 个；Top-k 输出为 3。
- 指标：Recall@1 和 Top-k 命中分别在 IoU 0.3/0.5 下统计，同时报告 Top-k 最大 IoU 与 Top-k 边界误差均值。
- 运行：Python 3.12.3、PyTorch 2.6.0+cu124、VideoITG commit `50a60a822c0e362bfd8747c45ba34e66e9c9d650`、单张 `cuda:0`、VideoITG `max_frames=16`。所有结果均保留逐样本 JSONL 和汇总 JSON。

## Baseline 定义

| 基线 | 定义 | 主要成本 |
|---|---|---|
| B0 | 本地 `openai/clip-vit-base-patch32` 对每个 5 秒候选的中心帧与查询做图文相似度，直接取 Top-k。 | 只做轻量图文召回 |
| B1（历史） | 用 VideoITG 对整段视频抽帧并得到帧分数，再将高分帧映射到 5 秒区间。 | 历史 baseline，不是主路径 |
| B2（历史参考） | 先用轻量模型取候选，再逐候选调用 VideoITG。 | 用于比较旧级联，不是 TimeLens |
| B2-Retrieval-only（CLI `B2R`） | 与 B2 相同的 CLIP 初筛，但保留全部 Top-4 候选，不调用 VideoITG。 | 用于测量候选召回上限 |

实现入口：`scripts/run_baseline.py`。示例：

```bash
PYTHONPATH=src:external/VideoITG \
HF_HOME=/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface \
HF_HUB_CACHE=/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface \
HF_HUB_OFFLINE=1 \
/home/hdd-2t/zjy_dataset/videoitg_smart_clip/envs/videoitg312/bin/python \
scripts/run_baseline.py --baseline B1 \
  --manifest records/phase1/videoitg40k_media_pilot.jsonl \
  --output /home/hdd-2t/zjy_dataset/videoitg_smart_clip/outputs/phase2_B1_full/predictions.jsonl \
  --summary /home/hdd-2t/zjy_dataset/videoitg_smart_clip/outputs/phase2_B1_full/summary.json \
  --videoitg-model /home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/VideoITG-8B \
  --device cuda:0 --max-frames 16
```

## 50 条 pilot 结果

结果明细见 `records/phase2/baseline_comparison.json`。对应输出目录均位于外置数据盘：

- B0：`outputs/phase2_B0_full_20260714_0145/`
- B1：`outputs/phase2_B1_full_20260714_0200/`
- B2：`outputs/phase2_B2_full_20260714_0050/`
- B2-Retrieval-only：`outputs/phase2_B2R_full_20260714_0110/`

| 基线 | Recall@1 IoU≥0.3 | Recall@1 IoU≥0.5 | Top-k 命中 IoU≥0.3 | Top-k 命中 IoU≥0.5 | Top-k 最大 IoU | 边界误差 (s) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.28 | 0.28 | 0.52 | 0.52 | 0.380 | 11.44 |
| B1 | **0.58** | **0.50** | **0.80** | **0.68** | **0.537** | **4.24** |
| B2 | 0.24 | 0.22 | 0.54 | 0.52 | 0.377 | 10.75 |
| B2-Retrieval-only (Top-4) | 0.28 | 0.28 | 0.66 | 0.64 | 0.487 | 7.55 |

这些历史数字只用于回归和路线对照。新主链路必须在同一 split 上重新报告 Retriever Recall@5/10/20、TimeLens grounding 指标、后处理指标和端到端性能。

## Retriever comparison

至少比较：

1. `CachedCosineRetriever`：对视频侧缓存特征与 Query embedding 做相似度排序。
2. `UniformTemporalRetriever`：均匀时间窗口参考上限/资源基线。

默认选择规则为“验证集 Recall@20 最高且满足延迟、GPU memory、Feature Cache Size 预算”；所有阈值和选择结果必须写入独立 run 记录，当前状态为 `pending`。

### B1 小规模 profiling

`records/phase2/b1_profiling.json` 保存了同一前 10 条 pilot 的抽帧密度、输出区间和离线 Top-k replay；随后又补充了 50 条 full-pilot 和 97 条 clean97 的 32/64 帧实跑。32 帧在 full pilot 的 Top-k@0.5=0.76（16 帧=0.68），clean97 为 0.722（16 帧=0.649）；64 帧进一步达到 pilot=0.78、clean97=0.732，并在最终回归池恢复短事件且保留 controls，但平均精排时间约增加 70%，历史 7 条 held-out aggregate 不变。因此默认仍冻结为 16 帧、5 秒，64 帧作为当前最强但未晋级的 Phase 4 候选。

为检查 held-out 证据是否过窄，又从未进入 clean97 且按 video-level hash 落在 test 的媒体中新增 27 条，完成固定 16/64 帧配对运行和 sampled-2s 全视频人工复核。64 帧的总体 Top-k@0.5 为 0.815（16 帧=0.778），但人工确认的 7 条 boundary_shift 类别仍为 0.571（两种帧数相同），ActivityNet-QA 组提升 0.20、NExTQA 组下降 0.167，平均重排时间增加约 41%；这仍不足以支持统一的 G4 晋级或动态门控。逐样本审计见 `records/phase4/heldout_extra27_mf64_audit.json`，复核池见 `records/phase4/heldout_extra27_regression_pool.jsonl`。

### Boundary head v1（离线后处理）

在不增加 VideoITG 调用、不改变 16 帧/5 秒候选协议的前提下，使用 train 71 条 frame-score/GT 样本拟合 CPU ridge boundary head。冻结参数后，final18 的 boundary_shift Top-k@0.5 从 0.25 提升到 0.75，additional heldout27 从 0.571 提升到 0.857；普通 50 条 pilot 的 Top-k@0.5 保持 0.68，单条 CPU 后处理约 0.29 ms。该结果只支持 boundary_shift scope 的条件晋级；short_event_miss 仍为 0，不能宣称短事件修复。完整审计见 `records/phase4/g4_candidate_audit.json`。

### B1 24 帧质量候选

保持 5 秒输出区间和同一 VideoITG 模型，仅把 full-video 抽帧上限从 16 提高到 24。普通 50 条 pilot 的 Top-k@0.5 保持 0.68，clean97 overall 从 0.649 提升到 0.701；final18 的 boundary_shift 从 0.25→0.50、confirmed short_event_miss 从 0→1、controls 保持 0.90；独立 additional heldout27 的 overall 从 0.778→0.852，boundary_shift 从 0.571→0.714，controls 从 0.833→0.889。该候选的 pilot 平均精排约 2.78 s、P95 约 4.17 s，较固定16更慢，因此只作为 G4 质量优先候选，固定16仍是默认，不能作为 G5 延迟通过证据。详见 `records/phase4/mf24_candidate_audit.json`。

## 成本与复现注意

- B0 每条样本平均检索耗时约 1.00 s；B1 平均 VideoITG 重排耗时约 2.43 s，峰值 CUDA 分配约 16.45 GiB；B2 平均候选重排耗时约 6.24 s，固定重排 4 个候选。
- 运行前必须先调用 `torch.cuda.set_device` 再导入 Decord；否则在本环境中曾复现 torch CUDA lazy-init/Decord segfault。脚本已内置该顺序。
- 上游 checkpoint 加载时会提示部分 SigLIP 视觉塔权重未用于 `EagleQwenG` 初始化；这是当前官方适配器结构的已观察加载提示，真实推理 smoke 和上述三组基线均已完成。
- 任何新增 baseline 必须复用同一 manifest、GT 转换、候选粒度和输出 schema，并保留唯一输出目录，禁止覆盖已有证据。
- 重复性审计见 `records/phase2/repeatability_audit.json`：B0/B1/B2 各自重跑 50 条后逐样本预测与指标差异均为 0；B2 两次平均重排耗时绝对差约 0.021 秒。
