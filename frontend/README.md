# Frontend

# Local frontend

已提供一个无第三方依赖的本地静态界面：上传视频、提交异步任务、轮询 coarse-to-fine 状态与阶段进度、预览候选、选择候选区间、调整 refined 边界和提交结构化反馈。界面同时兼容旧 `start_s/end_s/score` 与新 `refined_start/refined_end/final_score` 候选字段。任务进入 `TIMEOUT` 后，界面会继续读取 results 端点；后端返回迟到的 Level-2 coarse fallback 时，候选会显示并明确标记为降级结果。

启动方式：

```bash
cd /home/zjy/projects/videoitg_smart_clip
python3 -m http.server 8080 --directory frontend --bind 127.0.0.1
```

同时按 [`docs/service.md`](../docs/service.md) 启动 loopback 服务。打开 `http://127.0.0.1:8080`，默认 API 地址为 `http://127.0.0.1:8000`。浏览器直接打开文件不保证跨域可用，使用上述静态服务器。

API 已显式允许这两个 loopback 前端 origin 的 CORS 预检；不会开放通配符来源。可用 Playwright 做自动化 UI smoke（不等同真实用户验收）：

```bash
PLAYWRIGHT_BROWSERS_PATH=/home/hdd-2t/zjy_dataset/videoitg_smart_clip/browser_cache \
PYTHONPATH=src /home/zjy/miniconda3/bin/python scripts/g7_browser_smoke.py \
  --video tests/fixtures/corrupt.mp4 \
  --output records/phase8/g7_browser_smoke_20260901.json
```

反馈只写入外置运行时 JSONL，记录 task/video/candidate/query/model/final score、标签和可选边界调整；不保存原始视频内容或视频路径。
