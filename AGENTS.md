# VideoITG Smart Clip Agent Instructions

## Primary Objective

将本项目做成一个证据完整、可离线复现、可被面试深入追问的长视频自然语言片段检索原型；主路线为 Query-aware Coarse-to-Fine Temporal Grounding。本项目不以真实部署、线上服务或生产吞吐为目标。

核心成功标准不是模块数量，而是形成一条可复现证据链：

```text
基线 -> 构造困难集与 Badcase 归因 -> 定向优化 -> 离线回归评测
     -> 接口/状态机 -> 合成故障与降级 -> 结果归档
```

## Scope Lock

只保留两条亮点：

1. Badcase 驱动的构造难负样本、无匹配拒识和离线回归闭环。
2. 延迟预算感知的动态 Top-K、动态抽帧和负载降级。

除非用户明确批准且现有证据显示必要，否则不要扩展到：

- 参考图像条件化；
- 视频问答或视频生成；
- DPO/RLHF；
- 新的大模型结构；
- Kubernetes、多机分布式或虚构高并发；
- 完整重训 VideoITG、TimeLens 或其他大模型。

## Truthfulness and Evidence

- 所有“提升、降低、上线、稳定、可用、支持”等结论必须有对应日志、指标文件、测试或可运行服务作为证据。
- 计划值统一标为 `target`，未运行实验统一标为 `pending`，不得用预计数字替代实测结果。
- 若记录服务代码，只能称为“离线接口原型”或“本地服务 smoke”；不得称为个人生产部署、线上服务或大规模线上 A/B。
- 字节/剪映业务或岗位要求属于外部、易变化事实；需要引用时必须重新核验官方公开来源。
- TimeLens-8B 的接口和适配性必须有代码契约或已有证据支撑；真实 checkpoint、GPU、依赖和性能记录属于可选参考，不是离线项目完成条件。VideoITG-8B 仅作为 baseline，不能把历史证据转移到主路径。
- 保留失败实验和退化样本。任何局部提升都必须同时报告普通集、困难集和性能指标的变化。

## Required Session Startup

每个新对话或新 agent 在采取项目动作前依次执行：

1. 阅读本文件。
2. 阅读 `docs/project_state.md`。
3. 阅读 `plan/execution_plan.md` 中当前 Phase 和 Gate。
4. 检查现有进程、输出和版本状态，避免重复启动任务或覆盖用户文件。
5. 只执行当前 Gate 所需的最小工作；通过后再推进下一 Phase。

若文档状态与运行时证据冲突，以运行时证据为准，并更新 `docs/project_state.md`。

## Data and Artifact Policy

- 大型下载或生成数据必须优先写入 `/home/hdd-2t/zjy_dataset/videoitg_smart_clip`。
- 仓库 `data/` 只保存 README、小型 manifest、schema 和抽样示例，不保存原始视频、帧、向量索引或模型权重。
- 每次实验必须有唯一 `run_id`，并记录代码版本、配置、数据版本、模型版本、硬件、随机种子、开始/结束状态和指标路径。
- 不覆盖历史实验输出；新结果写入独立 run 目录。
- 用户已有改动不得被回滚、覆盖或清理。

## Experiment Discipline

- 先完成最小 smoke test，再做 pilot，再做完整实验。
- 固定测试集和困难集不得参与训练或阈值选择。
- 数据划分至少按 `video_id` 隔离；若数据包含同源视频或系列内容，还需检查近重复泄漏。
- 基线、优化和消融必须使用同一数据划分、候选预算定义和评测脚本。
- 动态控制器阈值必须来自开发集 profiling；禁止从测试集调参。
- Badcase 修复后必须运行普通集、困难集和历史回归池，报告新增修复与新增退化。
- 无匹配阈值需报告 precision/recall/F1 或 PR 曲线，不能只报告单点准确率。
- 如报告 P50/P95、超时率或 GPU 秒数，需写明测试配置；本项目不要求生产吞吐、线上容量或真实硬件复现。

## Service and Safety

- 默认仅绑定 `127.0.0.1`；开放外网、长期后台服务或上传真实用户数据前需用户明确批准。
- 上传文件必须做大小、时长、格式和路径校验；FFmpeg 调用不得拼接未经处理的 shell 输入。
- 记录请求 ID、模型/索引/阈值版本和降级原因，但不把原始用户视频、查询或敏感路径写入公开日志。
- 精排失败允许返回明确标注的粗召回结果；不得静默把降级结果伪装成完整精排结果。
- 模型切换、阈值更新和索引更新必须可回滚。

## Communication During Execution

- 执行多步操作时，必须同步用语言说明每一步在做什么、为什么做、以及当前进展，不得只做工具调用而不给用户上下文。
- 每个阶段开始前简要说明意图，结束后说明结果和意义。中间的关键决策或异常也应口头解释。

## Documentation Rules

- `docs/project_state.md` 是当前事实状态的唯一入口，完成 Gate、启动/结束长任务或发现阻塞时更新。
- 原计划未变化时不写变更记录；只有计划被修改或纠正时，才在 `records/plan_changes.md` 中追加简洁记录。
- README 只展示已验证能力和可复现命令。实验前不得填写简历数字。
- 工程完成、训练完成、离线评测完成和小规模上线验证必须分开报告。

## Current Status

截至 2026-09-03：项目处于 **Offline prototype / 自动化验证收尾完成**。主模型改为 TimeLens-8B Temporal Grounder；VideoITG-8B 降为 baseline。G0–G9 的离线自动化范围已完成：360 条媒体池 cache/Retriever/TimeLens 统计、合成后处理与负样本契约、失败/超时/取消状态机及已有运行时参考均有独立记录。人工标注、真实用户验收、真实负样本构建、真实部署和生产吞吐均不是完成条件；构造负样本可用于离线 No-Match、Ranking 和降级逻辑测试。当前全量回归为 79 passed（1 warning），release audit 为 117/117 pass；新会话交接详见 `docs/project_state.md` 的 `New conversation handoff`。
