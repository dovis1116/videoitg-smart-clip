# Pilot media spot audit

日期：2026-07-14  
样本：当前已下载媒体中的 50 条 pilot 记录（`2_3_m_nextqa` 10 条、`30_60_s_nextqa` 20 条、`30_60_s_activitynetqa` 20 条）

## 已检查

- 50/50 目标路径存在；50/50 使用 Decord 成功打开。
- 按上游 `clip_embedding.py` 的 1 FPS 采样规则校验，50/50 的 `frame_num` 均在特征帧范围内；按 5 秒 clip 规则校验，50/50 的 `clip_num` 均在区间内。
- 对三个来源组各抽取若干第一标注帧进行人工目视检查，均能正常渲染，未见损坏或全黑帧；接触表见 `/home/hdd-2t/zjy_dataset/videoitg_smart_clip/outputs/phase1_media_audit/media_pilot_contact_sheet.jpg`。
- 该 50 条 pilot 的真实视频时长与来源组命名一致（30–60 秒组和 2–3 分钟组），但尚未进行所有视频的精确时长分布审计。

## 语义 spot check

对三个来源组共 10 条样本，围绕标注的 1 FPS `frame_num[0]` 前后各取数帧，人工查看序列；10/10 条都呈现出与问题/答案**相容的视觉证据**（例如直升机下降、孩子与气球、狮子行走、秋千被推动，以及 ActivityNet 的人物动作）。这不是独立重标注，也不能推导全量准确率；仍需扩大抽样并记录明确错误率。

序列接触表：`/home/hdd-2t/zjy_dataset/videoitg_smart_clip/outputs/phase1_media_audit/semantic_spot_frames.jpg`。
逐条记录：`records/phase1/manual_spot_review.json`。

## 未声称的部分

本次检查不等于全量标注正确率，也没有得到逐视频可商业再分发许可；当前数据只能按来源卡的 academic research and education 限制使用。
