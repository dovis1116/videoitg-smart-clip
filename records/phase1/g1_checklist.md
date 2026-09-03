# Gate G1 checklist

更新时间：2026-07-14

| 条件 | 证据 | 状态 |
|---|---|---|
| schema validator 全部通过 | `videoitg40k_audit.json`, 474,354 条，0 validation errors | 通过 |
| 数据来源可追踪 | `hf_repo_metadata.json`，两个仓库 revision 已固定 | 通过 |
| video-level split 无交集 | `videoitg40k_audit.json`，三组 overlap 均为空 | 通过 |
| 30–50 条真实媒体 pilot | `videoitg40k_media_pilot.jsonl`，50 条 | 通过 |
| pilot 视频可解码 | `media_pilot_audit.json`，50/50 | 通过 |
| 1 FPS frame / 5 s clip 索引合法 | `media_pilot_audit.json`，50/50 | 通过 |
| 重复/近重复初筛 | `media_pilot_duplicate_screen.json`，精确重复 0，首帧/中帧/标注帧三位置 hash 近重复 0 | 初筛通过 |
| 数据许可与用途约束可追踪 | `source_cards/README.md` 固定了 Apache 2.0 与 academic research and education 限制；商业再分发未获授权 | 研究用途通过 |
| 随机人工语义一致性与错误率 | `media_pilot_manual_audit.md`：10 条多帧 spot check 均与问答相容；尚未形成独立重标注错误率 | target-present pilot 通过 |
| 困难集六类覆盖 | `docs/hardcase_protocol.md` 与 `hardcase_candidates.jsonl` 已建立；无匹配/ASR/短事件边界不由当前元数据证明，按范围修正延期 | 延期，不阻塞 target-present pilot |

结论：G1 **conditional pass**（仅限 target-present temporal grounding pilot）。无匹配、ASR/视觉冲突和短事件边界不在本 Gate 的可报告范围内。
