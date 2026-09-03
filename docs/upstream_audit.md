# VideoITG 上游核验（历史 baseline）

本文件记录迁移前 VideoITG-8B 的可复现实验事实，不代表它仍是主 Grounder。TimeLens-8B 需要在 G4 单独核验 checkpoint、输入输出和性能。

更新时间：2026-07-14

## 核验对象

- 官方代码：本地 checkout `external/VideoITG`，commit `50a60a822c0e362bfd8747c45ba34e66e9c9d650`。
- 主模型：本地 `models/VideoITG-8B`，4 个 safetensors shard、index、config 和 tokenizer 均存在。
- 视觉塔：`config.json` 指向 `google/siglip-so400m-patch14-384`；该模型不在 VideoITG-8B 目录内，已单独缓存到 `cache/huggingface`，占用约 3.3 GiB。
- 许可：代码目录包含 Apache 2.0 `LICENSE`；模型目录包含 NVIDIA License，条款明确限制为非商业研究/评估用途。简历项目不能据此宣称商业可用。

## 环境证据

运行环境：`envs/videoitg312`，Python 3.12.3；PyTorch `2.6.0+cu124`，CUDA runtime `12.4`；GPU 为 3× NVIDIA GeForce RTX 4090D 24,564 MiB，驱动 `595.71.05`。

官方代码的 SigLIP loader 强制传入 `attn_implementation='flash_attention_2'`。上游 requirements 没有列出 `flash-attn`，因此额外安装并验证 `flash-attn==2.7.4.post1`；安装过程成功。

## Smoke test

可复现脚本：`scripts/phase0_smoke.py`。输入为官方 `imax.mp4`，查询为上游 `infer.py` 中的人工可核验多选题。运行记录：

- 成功配置：单卡 `CUDA_VISIBLE_DEVICES=0`、本地模型和本地 Hugging Face cache、`HF_HUB_OFFLINE=1`。
- 32 帧：成功完成模型加载、视频采样、SigLIP 编码和 VideoITG 前向，输出 8 个带分数的帧索引。
- 结果文件：`outputs/phase0_smoke_20260714_0015/result.json`。
- 观测：视频 9,514 帧、23.976 FPS；采样 32 帧；模型加载 29.29 s；预处理和推理总耗时 33.99 s；峰值 CUDA 分配 16.56 GiB。
- 512 帧：在同一单卡上 OOM。日志显示模型已占用约 19.54 GiB，视觉编码额外申请 2.25 GiB 时失败；记录于 `outputs/phase0_smoke_20260714_0000/smoke.log`。

因此 G0 的“单样本可复现推理”已通过，但“官方默认 512 帧单卡可运行”未通过。当前证据支持先采用候选局部/分块视觉编码，而不是把 512 帧整段直接送入单卡。

## 最薄 adapter 接口

当前实现为 `src/videoitg_smart_clip/reranker/videoitg_adapter.py`，并已在两个 12 秒候选上运行；结果在 `outputs/phase1_adapter_smoke_20260714_0100/result.json`，运行元数据在 `records/phase1/adapter_smoke_run_meta.json`。adapter 将候选片段转换为：

```text
rank(query: str, candidates: list[Candidate]) -> list[ScoredCandidate]
Candidate(video_path, start_s, end_s, frame_indices)
ScoredCandidate(..., frame_scores, segment_score, model_version, runtime)
```

实现上应按候选片段分批，每批限制 `max_frames`，保留原始帧索引到视频时间的映射，并记录 `peak_cuda_gib`、wall time 和降级状态。32 帧成功、512 帧 OOM 是当前 batch-size 选择的实测边界，不应写成普遍硬阈值。

## G0 判断

G0 可判定为“有条件通过”：模型和上游接口已跑通，且有可复现实验记录；但默认长序列输入不适合单卡，需要在 Phase 1 先完成分块/候选局部方案和数据覆盖审计后再扩大训练或服务范围。
