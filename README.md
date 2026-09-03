# Query-aware Smart Clip

面向长视频的自然语言片段检索系统。用户上传长视频并输入自然语言指令，系统通过 Query-aware Coarse-to-Fine Temporal Grounding 返回 Top-K 候选时间区间、关键帧、置信度和可预览片段。

> 当前状态：**离线原型已完成当前范围。** TimeLens-8B 是主 Grounder，VideoITG-8B 仅保留为 baseline/粗召回参考。项目以本地数据、合成输入、接口契约和自动评测为验收依据，不要求真实部署、真实用户验收或生产吞吐。

## 项目定位

项目采用两阶段 Query-aware Coarse-to-Fine Temporal Grounding：

```text
Long Video + Query -> 视频解析与特征缓存 -> Query-aware 粗召回
        -> Top-N 候选窗口 -> TimeLens-8B Temporal Grounding
        -> Boundary Refinement -> Ranking -> Temporal Deduplication
        -> No-Match 判断 -> Top-K 片段 -> 预览/边界调整 -> 反馈/Badcase 回归
```

模型职责与边界：

1. TimeLens-8B 只在 Top-N 候选窗口内预测绝对时间边界。
2. VideoITG-8B 不承担最终片段预测，仅作为 baseline 和粗召回参考。
3. Retriever、Grounder、Boundary、Ranking、Dedup、No-Match 通过统一接口解耦；视频侧特征按版本缓存并支持重复 Query 复用。
4. 项目只负责找到、排序、预览、调整和反馈，不负责自动拼接、字幕、转场、音乐、特效或生成最终成片。

当 No-Match 阈值判定事件不存在时，用户结果不强制展示 Top-K；候选中间结果仍保留在诊断字段中。

## 目录

```text
videoitg_smart_clip/
├── AGENTS.md                    # 新对话/新 agent 的项目约束与恢复入口
├── README.md                    # 项目概览与当前事实状态
├── plan/execution_plan.md       # 分阶段详细执行计划和验收门槛
├── docs/
│   ├── architecture.md          # 系统边界、组件职责和关键接口
│   ├── project_state.md         # 当前阶段、证据与下一步
│   └── service.md               # steady 同步与 burst 异步服务契约
├── configs/default.yaml         # 第一版可调整配置；阈值需经 profiling 确定
├── data/README.md               # 大数据外置约定与标注格式
├── src/videoitg_smart_clip/
│   ├── preprocessing/           # 镜头切分、抽帧和特征提取
│   ├── retrieval/               # 镜头级召回和索引
│   ├── grounding/               # TimeLens-8B Temporal Grounder
│   ├── pipeline/                # 稳定契约、后处理与两阶段编排
│   ├── reranker/                # VideoITG baseline 适配
│   ├── budgeting/               # 动态 Top-K/FPS/降级控制器
│   ├── boundaries/              # 分数平滑、区间生成和去重
│   ├── evaluation/              # 离线指标、性能压测和对比报告
│   ├── badcase/                 # 错误分类、难负样本和回归池
│   └── service/                 # FastAPI、任务状态、反馈与版本管理
├── frontend/                    # 时间轴预览与反馈界面（Phase 7 最小版本）
├── scripts/                     # 可复现实验/服务命令
├── tests/                       # 单元、集成和回归测试
├── records/                     # 实验清单、评测证据和必要的计划变更记录
└── outputs/                     # 小型报告；大输出仍放外置数据盘
```

## 数据与模型位置

仓库内只保存小型清单、配置和结果摘要。大文件统一放在：

```text
/home/hdd-2t/zjy_dataset/videoitg_smart_clip/
├── raw/
├── annotations/
├── processed/
├── frames/
├── features/
├── indexes/
├── models/
└── cache/
```

## 开始工作

新会话先依次阅读：

1. `AGENTS.md`
2. `docs/project_state.md`
3. `plan/execution_plan.md`

模型下载步骤和固定路径见 [`docs/model_download.md`](docs/model_download.md)。
TimeLens-8B 主模型的独立核验清单见 [`docs/timelens_model.md`](docs/timelens_model.md)；G4 离线适配、窗口输入和评测结果已归档，真实 GPU/服务复现仅为可选参考。
官方 VideoITG-40K 元数据下载步骤见 [`docs/dataset_download.md`](docs/dataset_download.md)。

受限服务交付和本地前端验证见 [`docs/release_checklist.md`](docs/release_checklist.md)。

现有数据收尾证据见 [`records/phase_g8/current_data_completion_20260903.json`](records/phase_g8/current_data_completion_20260903.json)；离线验证范围与合成负样本方案见 [`docs/real_acceptance_plan.md`](docs/real_acceptance_plan.md)。人工标注、真实浏览器用户验收、真实环境复现和生产吞吐不在当前范围。

离线评测统一入口为 `python eval/run_eval.py --config configs/eval.yaml`；若已有 JSONL 预测，可追加 `--input <predictions.jsonl>`，输出 `metrics.json`、`badcases.json` 和 `report.md`。

G8 验证清单的 present/synthetic 清单生成和数据边界见
[`docs/validation_manifest.md`](docs/validation_manifest.md)；脚本不会自动臆造负样本标签。

已有真实本地链路 smoke 可作为参考：`scripts/g4_timelens_smoke.py`（TimeLens 窗口/真 batch）、`scripts/g6_realtime_smoke.py`（SigLIP→Retriever→TimeLens）和 `scripts/g6_realtime_service_smoke.py`（`/tasks` 异步接口）。这些记录不是离线项目的必需环境，也不代表生产质量或并发能力。

第一阶段先完成 G0 现有系统审计和接口迁移；随后分别验证 Feature Cache、两种轻量 Retriever、TimeLens adapter、边界后处理、No-Match 阈值和降级路径。所有阈值、模型和指标在离线验证集（包括 synthetic 清单）上确定并记录，不能凭经验写死。
