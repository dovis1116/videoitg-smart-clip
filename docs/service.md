# Coarse-to-fine service contract

服务目标是 Query-aware Retriever → TimeLens Grounder → 独立后处理链路。当前保留旧 VideoITG backend 仅用于 baseline/兼容 smoke；TimeLens checkpoint、候选窗口 API 和真 batch smoke 已通过，质量/失败/超时验证仍 pending。

## Endpoints

兼容 `/v1/*` 路径，同时提供：

- `POST /tasks`：提交异步任务（旧路径 `/v1/tasks`）。
- `GET /tasks/{task_id}`：查询状态、progress、current_stage、degraded，以及任务的 `model_version`、`created_at`、`updated_at`。
- `GET /tasks/{task_id}/results`：读取完整候选记录；成功任务正常返回，带显式 Level-2 粗窗口 fallback 的 `TIMEOUT` 任务也可只读获取。响应同时提供兼容字段 `status` 和统一字段 `canonical_status`。
- `DELETE /tasks/{task_id}`：取消任务。
- `POST /v1/tasks/upload`：上传视频并创建任务。
- `POST /v1/feedback`：保存候选反馈和可选人工边界调整。

默认前端通过 `http://127.0.0.1:8080` 访问 API；服务只允许该地址和
`http://localhost:8080` 的显式 loopback CORS，不开放通配符来源。

任务状态文档名为 `PENDING`、`PREPROCESSING`、`INDEXING`、`RETRIEVING`、`GROUNDING`、`POSTPROCESSING`、`SUCCESS`、`FAILED`、`CANCELLED`、`TIMEOUT`。旧 SDK 的 lowercase 状态仍兼容返回，但新字段 `current_stage` 使用上述 canonical 名称。

服务重启加载 JSONL 状态时，任何此前处于 `queued`/`running` 的任务都会显式恢复为 `FAILED`，并设置 `current_stage=FAILED`、`error_code=SERVICE_RESTART_RECOVERY`；不会把中断任务伪装成成功或继续运行。该恢复原因与模型版本会保留在任务记录中。

## Result contract

每个候选必须保留粗窗口、raw grounding、refined boundary、retrieval/grounding/final 分数、duplication penalty、rank、`pre_dedup_rank`、版本、阶段延迟、degraded/degrade_level/degrade_reason。Boundary 的 start/end 偏移与 padding 独立配置，未验证前保持 0。`raw_start/raw_end` 永不被人工调整或 refined 结果覆盖。`NO_MATCH` 状态不强制返回 `predictions`；完整候选（包括 `deduplicated=true` 的重复行）保存在 `candidates` 诊断字段。

Boundary 参数必须是有限值，扩展量和 padding 不能为负；若组合后得到折叠区间，pipeline 会保留 raw 边界并显式进入 Level-1 降级，不把非法区间交给后续 Dedup。

## Degradation

- Level 0：Retriever → TimeLens → Boundary → Ranking/Dedup。
- Level 1：Boundary 异常，返回 raw grounding。
- Level 2：TimeLens 超时或异常，返回 coarse windows。

降级必须显式标记，队列满、超时、解码失败和模型异常均不得伪装为成功。运行中
任务收到 DELETE 后不会强杀 GPU 调用；后端自然返回时结果被丢弃，任务最终持久化
为 `CANCELLED`，不会被覆盖成 `SUCCESS`。
异步任务使用独立 `async_timeout_ms`（默认 120000 ms），与同步 `/v1/steady/rerank` 的 `deadline_ms` 分离；watchdog 到预算即持久化 `TIMEOUT`/`error_code=TIMEOUT`，不强杀正在运行的 GPU 调用。若后端随后返回候选，结果接口会更新为显式 `degraded=true`、`degrade_level=2` 的粗窗口；若尚未返回，客户端应按 `409` 重试结果接口，而不是把空结果当作成功。

反馈事件关联 task/video/candidate/query/model/final score，并额外保留 `query_hash`；反馈文件属于受控本地运行时存储，不写入原始视频内容或视频路径。人工边界调整始终与模型边界分开保存为 `model_start/model_end` 与 `user_start/user_end`（同时保留带 `_s` 的兼容字段），且开始/结束边界必须成对提交；lowercase 旧标签在落盘前规范化为八类大写标签。

## Run contract smoke

```bash
cd /home/zjy/projects/videoitg_smart_clip
PYTHONPATH=src:external/VideoITG /home/zjy/miniconda3/bin/python -m pytest -q
PYTHONPATH=src:external/VideoITG /home/zjy/miniconda3/bin/python scripts/run_service.py \
  --backend stub --devices cpu --host 127.0.0.1 --port 8000
```

真实新链路需显式提供本地 SigLIP 和 TimeLens checkpoint；可先用
`--top-n 1 --timelens-batch-size 1` 做单请求 smoke，再增加候选数。省略
`--timelens-model-path` 时不会静默使用模型，而是按 Level-2 记录 coarse
候选降级。`--timelens-max-new-tokens` 和 `--timelens-total-pixels` 控制
生成/视觉预算，默认值与 `configs/default.yaml` 同步。

后处理模块通过启动参数独立注入：`--boundary-refinement-enabled`（或
`--no-boundary-refinement-enabled`）及边界 offset/padding、五个
`--ranking-*-weight`、`--dedup-temporal-iou-threshold` 和三个
`--no-match-*-threshold`。未完成验证集校准时 No-Match 阈值保持为空，任务状态
不会被误报为 `CONFIDENT`。

视频侧采样预算也必须显式配置：`--feature-sample-fps` 和
`--feature-max-frames` 默认分别为 `1.0` 与 `16`。两者会同时写入
Feature Cache 的采样身份；后续 Query 只重新校验缓存，不重新执行完整视频编码。

服务默认只绑定 `127.0.0.1`，上传文件受扩展名、大小、可探测时长和允许目录限制；超时会持久化为 `TIMEOUT` 终态。大文件、特征和模型仍写入 `/home/hdd-2t/zjy_dataset/videoitg_smart_clip`。
