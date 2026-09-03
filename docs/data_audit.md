# VideoITG-40K 数据审计（Phase 1）

更新时间：2026-07-14

## 来源与范围

本地元数据来自 Hugging Face `NVEagle/VideoITG-40K`；当前路线中仅用于 VideoITG baseline/格式审计，不作为 TimeLens 主路径证据。仓库 revision `112bdd23fc9b804d6e4afe5758d28b5129f933ca`。数据卡标注 `apache-2.0`，主文件为 `video_itg_data.json`；本地文件包含 474,354 条记录、42,519 个视频路径。逐次 API 查询和文件清单保存在 `records/phase1/hf_repo_metadata.json`。

元数据中的所有 `video` 路径都以 `LLaVA-Video-178K/` 开头，媒体源仓库是 `lmms-lab/LLaVA-Video-178K`。该源仓库 revision 为 `6d8c562dc26d70042a0d9704d1cae58c94b89098`，包含多个按时长/来源分组的 tar 分片；完整媒体仓库过大，当前没有下载。源数据卡虽写 Apache 2.0，但同时明确限制为 academic research and education；原文已保存到 `records/phase1/source_cards/README.md`，因此本项目只能按研究/教育用途处理，不能写成商业数据可用。

## 结构校验结果

- 必需字段 `id/video/question/answer/frame_num/clip_num/motion/existence` 全部存在。
- 空文本、空路径、负帧号、负片段号和非法类型：0 条。
- `id` 唯一：474,354/474,354；`(video, question)` 无重复。
- 视频路径唯一：42,519；单视频最多 15 条标注。
- `frame_num` 长度 1–86，中位数 4；`clip_num` 长度 1–39，中位数 2。
- `motion` 全部为 `No`，`existence` 全部为 `No`。这意味着该元数据快照不能单独支撑“运动/无匹配”分层评估，不能把这些字段当作已验证的正负标签。
- 按视频路径 hash 生成 train/dev/test（约 80/10/10），视频级交集为空；这只是可复现的内部 split，不是官方 benchmark split。

完整机器审计结果：`records/phase1/videoitg40k_audit.json`。

## 媒体可用性与 pilot

当前只下载了三个小分片：`2_3_m_nextqa_videos_1.tar.gz`（约 184 MB）、`30_60_s_nextqa_videos_2.tar.gz`（约 1.35 GB）和 `30_60_s_activitynetqa_videos_1.tar.gz`（约 3.37 GB）。全量 42,519 个唯一媒体路径中已有 365 个可匹配，其余 42,154 个仍缺失；因此尚未声称 G1 通过。已生成 50 视频的真实媒体 pilot manifest：

`records/phase1/videoitg40k_media_pilot.jsonl`

它包含代表查询、帧/clip 标注、来源分组和许可证待核验字段。50/50 路径存在、50/50 可解码、50/50 的帧标注在范围内；机器结果见 `records/phase1/media_pilot_audit.json`，人工抽样记录见 `records/phase1/media_pilot_manual_audit.md`。该 pilot 只覆盖三个来源组，不能代表完整数据集。

## 当前数据决策

1. 不下载完整 699GB 级媒体仓库。
2. 先按 pilot 中的 source group 选择一个或少量 tar 分片，下载到 `/home/hdd-2t/zjy_dataset/videoitg_smart_clip/raw/` 并做路径匹配。
3. 在真实媒体到位前，不进入训练、不报告 Recall/mAP，也不把 VideoITG-40K 宣称为本项目长视频业务数据集；其标注时长主要为 30 秒–3 分钟，而项目目标是 5–30 分钟长视频。
