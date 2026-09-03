# TimeLens-8B model audit and optional local runtime

本文件记录 TimeLens-8B 的可选本地复现方式；离线原型不要求下载 checkpoint、GPU 运行或在其他真实环境重现。

TimeLens-8B is the target primary Temporal Grounding model after the
2026-08-31 architecture migration. The official model is
`TencentARC/TimeLens-8B`, based on Qwen3-VL-8B-Instruct, and its model card
declares BSD-3-Clause. Official sources:

- Model: <https://huggingface.co/TencentARC/TimeLens-8B>
- Code: <https://github.com/TencentARC/TimeLens>

The checkpoint has been downloaded and verified at
`/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B` (four
safetensors shards plus `model.safetensors.index.json`, about 17 GB).

The official Transformers example accepts a video plus a text prompt and
returns timestamp text. It does not expose a native candidate-window array;
the project adapter must therefore materialize/localize each candidate window
and map local timestamps back to the original video timeline.

## Download

```bash
HF_HOME=/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface \
/home/hdd-2t/zjy_dataset/videoitg_smart_clip/envs/videoitg312/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TencentARC/TimeLens-8B",
    local_dir="/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B",
)
PY
```

The local Miniconda runtime has the model-card-compatible loader dependencies:
Transformers 5.8.1, Accelerate, `qwen-vl-utils==0.0.14`, PyTorch and Decord. Local
`AutoConfig` (`qwen3_vl`) and `AutoProcessor` loading both pass. The official
example also uses CUDA and `flash-attn==2.7.4.post1` for inference.

下载命令可使用隔离的 `envs/videoitg312`，但当前项目真实推理/服务验证统一使用
`/home/zjy/miniconda3/bin/python`；用系统 `/usr/bin/python3` 检查会误报上述依赖缺失。

## Required G4 evidence

- official repository/checkpoint and license (verified above);
- candidate-window input format and maximum window/frame budget;
- raw start/end output parser and absolute-time mapping;
- batch behavior, latency, GPU memory and failure modes;
- one-video smoke test with saved raw predictions.

The first real-video single-window and two-window serial smokes passed on
2026-08-31: TimeLens returned timestamp ranges for 0–5 second and 5–10 second
candidates and the adapter mapped them to absolute time. A true two-window
batch smoke also passed after serializing one chat template per candidate; it
returned two answers in 27.4 s with a GPU peak of 18.47 GiB allocated (19.08
GiB reserved). This is an adapter correctness/resource smoke, not a throughput
or quality benchmark.

On the current 24 GiB RTX 4090D runtime, `flash-attn` is not installed. The
adapter therefore selects PyTorch SDPA and uses a default `total_pixels` budget
of 4,194,304 (configurable through `--timelens-total-pixels`). A real async
HTTP smoke with this budget completed in about 20.7 s with
`grounder_version=TimeLens-8B`, `degraded=false`, and an absolute-time result;
the record is `records/phase_g6/g6_realtime_service_smoke.json`. The earlier
14,680,064-pixel SDPA attempt OOMed and is retained as a failure signal; do not
compare its memory/performance to the lower-budget run.

The adapter currently sets `grounding_score=1.0` to indicate a parseable model
answer; this is not a calibrated model confidence and must be replaced or
calibrated before No-Match thresholds are enabled.

The fixed target-present quality pilot exposed a quality gap: five cached
feature samples achieved Top-1 R@1 IoU>=0.3 of 0.20 with mean boundary error
about 37.2 s. The expanded 360-video weak-label offline run is the current
diagnostic baseline: R@1 IoU>=0.3 is 0.3056 and mIoU is 0.1959. These records
verify the evaluation path and show that parseable timestamps are not
sufficient evidence of real-world grounding quality; the numbers are not a
production claim. See
`records/phase_g4/g4_timelens_pilot_eval/metrics.json` and
`records/phase_g4/g4_timelens_top20_batch2_eval/metrics.json`.

A one-variable Retriever query-normalization experiment removed the fixed
dataset answer-format suffix before text embedding while preserving the raw
Query for TimeLens. On the same five samples, Top-1 IoU hit rate stayed at
0.20, while mean boundary error decreased from about 37.2 s to 20.1 s. The
change is retained as a preprocessing fix, but it is not a G4 quality pass;
the paired output and evaluation are in
`records/phase_g4/g4_timelens_pilot_query_normalized.jsonl` and
`records/phase_g4/g4_timelens_pilot_query_normalized_eval_v1/`.

The first real-model profiler run on one 10-second clip (one candidate, one
RTX 4090D) measured about 24.8 s end to end: indexing 5.95 s, retrieval 21.7
ms, grounding 16.04 s and postprocess 2.44 s. CUDA peak was 19.26 GiB
allocated / 19.69 GiB reserved. The profiler showed large matrix-multiply,
host-to-device copy and FlashAttention costs; this is a baseline profile, not
an optimization result. Full candidate-load, concurrency and timeout
benchmarks are optional for this offline project.

The current SDPA/4M-pixel profile on the same clip measured 23.6 s end to end
(indexing 4.11 s, retrieval 341.8 ms, grounding 16.62 s, postprocess 1.82 s)
with a 20.29 GiB allocated CUDA peak. This is a separate environment/budget
measurement, not a claim of improvement over FlashAttention.

A later 120-second asynchronous smoke was interrupted while the task was in
`INDEXING`: both `nvidia-smi -L` and a bounded `torch.cuda.is_available()` probe
stopped responding, indicating an unresponsive host CUDA driver/device call.
The process was terminated and the persisted task was recovered as explicit
`FAILED`/`SERVICE_RESTART_RECOVERY`; this historical environment failure is
retained for diagnosis but is not an offline-project blocker. See the issue
log for the evidence and recovery procedure.

The follow-up probe also found the local VS Code NVIDIA monitor extension was
launching an unbounded `nvidia-smi` child every two seconds after the driver
stalled. Disable/reload that extension before GPU validation; this is an
environment repair prerequisite, not a TimeLens model change.

`grounding.timelens.TimeLensGrounder` has offline adapter, failure, timeout and
weak-label diagnostic evidence; synthetic inputs may be used for additional
mechanism checks. The service may use documented Level-2 coarse-window
degradation for those cases.
Do not place the checkpoint in the repository; large model files belong under
`/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/`.
