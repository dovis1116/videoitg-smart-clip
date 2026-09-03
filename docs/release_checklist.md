# Restricted local release checklist

这是离线原型交付，不要求真实环境复现、生产部署、公网服务或线上吞吐。

## Architecture checks

- [x] Feature Cache 命中时第二个 Query 未重新完整编码视频（真实双 Query smoke；完整验证集仍 pending）。
- [x] Retriever 至少比较两个轻量方案并报告 Recall@5/10/20、延迟、显存、缓存大小（target-present pilot；负样本/困难集仍 pending）。
- [x] 当前 Python 运行时 Decord wheel tag、`pip check`、真实 `VideoReader` smoke 及 FFmpeg 6 源码 runtime 依赖闭包均已验证；wheel 基线为 manylinux_2_39，较老部署系统需重新打包（见 `records/phase_g9/decord_build_20260901.json`）。
- [x] TimeLens 只接收候选窗口；VideoITG 仅作为 baseline（单窗口、串行双窗口、真 batch smoke）。
- [x] Boundary、Ranking、Dedup、No-Match 可独立开关并保留 raw/refined 字段（合成契约 smoke；启动参数可注入，质量消融仍 pending）。
- [x] Synthetic No-Match/Ranking 清单支持四类显式负样本，并由校验器检查类别覆盖；结果必须标注 `synthetic`。
- [x] 离线 No-Match/Ranking 阈值逻辑可在构造数据上运行；不宣称真实世界校准。
- [x] 降级 Level 0/1/2 显式记录（合成故障与真实 async watchdog/fallback smoke）。
- [x] CPU stub 故障矩阵覆盖成功/幂等、失败、TIMEOUT→Level-2、取消、队列满、OOM 后恢复、双 worker 并发和线程清理（不替代真实 TimeLens 并发/显存证据）。
- [x] 现有 360 条媒体的解码/标签审计和 360 份 SigLIP 缓存覆盖/大小/提取耗时已归档；target-present weak-label 结果不外推为真实质量。
- [x] Synthetic 四类负样本支持 No-Match/Ranking 离线机制测试；不把构造数据称为真实负样本。
- [x] GPU/真实模型运行记录作为可选参考已归档；不要求在其他真实环境复现，也不作为离线项目门槛。
- [x] 自动 TimeLens 参数矩阵覆盖合法窗口/batch、视觉与生成预算、高预算未复现 OOM、损坏媒体 failure 和异步 TIMEOUT；已有预测上的 Boundary/Ranking/Dedup replay 已归档。

## Service checks

```bash
cd /home/zjy/projects/videoitg_smart_clip
PYTHONPATH=src:external/VideoITG /home/zjy/miniconda3/bin/python -m pytest -q
PYTHONPATH=src:external/VideoITG /home/zjy/miniconda3/bin/python scripts/validate_local_workflow.py \
  --iterations 8 --output records/phase8/local_workflow_protocol.json
python3 -m http.server 8080 --directory frontend --bind 127.0.0.1
```

- [x] `POST /tasks`、`GET /tasks/{id}`、`GET /tasks/{id}/results`、`DELETE /tasks/{id}` 可用（单请求 HTTP smoke 与契约测试）。
- [x] CPU Stub 有界队列压力探针可复现 6×200、2×429、0 超时；记录见 `records/phase_g6/stub_pressure_8.json`（不替代真实 TimeLens 并发验收）。
- [x] 跨端口 loopback 前端/API CORS 预检通过；Playwright headless UI smoke 完成上传→轮询→候选选择→反馈（仅作自动接口回归，记录见 `records/phase8/g7_browser_smoke_20260901.json`）。
- [x] 前端预览、候选切换、边界修改和反馈代码路径可用；真实浏览器用户验收不在范围。
- [x] 前端静态资源可通过 `python3 -m http.server` 提供，`index.html` 与 `app.js` 可访问；这只是静态 smoke，不替代真实浏览器交互验收。
- [x] 反馈同时保存 `model_start/model_end` 和 `user_start/user_end`，不覆盖 raw 预测。
- [x] 反馈事件包含 task/video/candidate/query/model/final score，并保留 query_hash 便于审计。
- [x] 人工标注、真实用户验证、生产吞吐和真实负样本均不属于完成条件；synthetic 结果不冒充真实质量结论。

离线交付审计：

```bash
PYTHONPATH=src:external/VideoITG /home/zjy/miniconda3/bin/python scripts/audit_release.py --output records/phase9/release_audit_20260903_final.json
```

审计会检查新路线关键文件、scope marker、离线 workflow 以及可选运行时 artifact；`g8_status` 固定反映人工标注/真实用户验收不属于当前范围。
