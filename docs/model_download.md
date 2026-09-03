# Model download and optional local runtime

模型下载、GPU 推理和本地运行时仅是可选参考，不属于离线原型的完成条件。

本文件保留 VideoITG-8B 的 baseline 下载步骤；新主路径使用 TimeLens-8B，
其独立核验、显存预算和候选窗口适配见 [`docs/timelens_model.md`](timelens_model.md)。

VideoITG-8B is retained only for historical baseline and compatibility smoke tests after the architecture migration. The primary Grounder is TimeLens-8B; its checkpoint/API and local candidate-window adapter have been smoke-verified in G4, while quality and failure/timeout validation remain separate gates.

本文件只负责下载和核验模型，不代表模型已经下载或已经通过项目 Gate。

## Confirmed upstream artifacts

- Code: `https://github.com/NVlabs/VideoITG`
- Checkpoint: `nvidia/VideoITG-8B`
- Local code path: `/home/zjy/projects/videoitg_smart_clip/external/VideoITG`
- Local model path: `/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/VideoITG-8B`
- Primary local Grounder: `/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B`
- Hugging Face cache: `/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface`
- Verified runtime environment: `/home/zjy/miniconda3` (Transformers 5.8.1, Decord, static-ffmpeg); the older `/home/hdd-2t/zjy_dataset/videoitg_smart_clip/envs/videoitg` venv is retained for historical VideoITG checks.

官方 checkpoint 当前为 8B、BF16、4 个 safetensors shard；模型页面标记为 NVIDIA License，并注明 research preview / non-commercial use。下载前应确认个人项目用途符合该许可；不能把该权重直接包装成商业生产模型。

## Observed download size

按 2026-07-13 远程文件 `Content-Length` 核对，当前命令的落盘大小约为：

| 内容 | 大小 |
|---|---:|
| 4 个模型 shard | 13.999 GiB |
| 配置、tokenizer、LICENSE、README | 15.2 MiB |
| `video_itg_data.json` | 236.9 MiB |
| `imax.mp4` | 22.2 MiB |
| 官方代码仓库 | 约 56.4 MiB |
| **合计** | **约 14.32 GiB** |

建议数据盘至少预留 30 GiB，以覆盖 Hugging Face 缓存、下载临时文件和后续日志；实际大小会随上游 revision 变化，下载前可重新执行 HEAD 检查。

## Step 0 — Check disk and tools

```bash
df -h /home/hdd-2t
command -v git
command -v python3
python3 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

本机已核验：系统 `python3` 为 3.12.3，项目运行时使用 `/home/zjy/miniconda3/bin/python`（Python 3.13，torch/Transformers/Decord/static-ffmpeg/safetensors 均可导入），Git 和 GPU 可用。不要用系统 `python3` 判断模型推理依赖；它只用于静态 HTTP 前端。上游 Decord 0.6.0 wheel 的 `cp36` tag 不兼容当前 Python；项目已用 FFmpeg 6.1.1 + Decord 源码构建，并通过 `auditwheel repair` 打包完整 FFmpeg runtime 依赖，归档并安装 `deps/decord-0.6.0-cp313-cp313-manylinux_2_39_x86_64.whl`，`pip check` 与真实 `VideoReader` smoke 均通过。该 wheel 的 manylinux_2_39 标签对应当前主机；跨更老系统需重新构建。构建细节见 `records/phase_g9/decord_build_20260901.json`。

## Step 1 — Clone the official code

```bash
export PROJECT_ROOT=/home/zjy/projects/videoitg_smart_clip
export VIDEOITG_CODE=$PROJECT_ROOT/external/VideoITG

mkdir -p "$PROJECT_ROOT/external"
git clone --depth 1 https://github.com/NVlabs/VideoITG.git "$VIDEOITG_CODE"
```

如果目标目录已经存在，先检查是否是用户已有内容，不要直接删除或覆盖：

```bash
git -C "$VIDEOITG_CODE" remote -v
git -C "$VIDEOITG_CODE" status --short
```

## Step 2 — Create an isolated download environment (optional)

这一步只安装下载器，不先安装完整 VideoITG 依赖，避免在上游兼容性尚未验证前修改大量环境。当前已验证的推理运行时是 `/home/zjy/miniconda3`；如果只需复用已有本地模型，可跳过本步骤。

```bash
export DATA_ROOT=/home/hdd-2t/zjy_dataset/videoitg_smart_clip
export VENV_ROOT=$DATA_ROOT/envs/videoitg

mkdir -p "$DATA_ROOT"/models "$DATA_ROOT"/cache/huggingface "$DATA_ROOT"/envs
python3 -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip
python -m pip install "huggingface_hub==0.28.1"
```

## Step 3 — Download the checkpoint to the data disk

```bash
export HF_HOME=/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export MODEL_ROOT=/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/VideoITG-8B

mkdir -p "$MODEL_ROOT"

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/VideoITG-8B",
    local_dir="/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/VideoITG-8B",
    local_dir_use_symlinks=False,
)
PY
```

如果 Hugging Face 返回 401/403，先在当前 venv 中执行登录，再重试；不要把 token 写进脚本或日志：

```bash
huggingface-cli login
```

该模型页面当前显示为 public、not gated；登录通常不是必需步骤。下载过程中断时，重复执行 Step 3 会复用已有文件。

## Step 4 — Verify the local snapshot

```bash
test -f "$MODEL_ROOT/config.json"
test -f "$MODEL_ROOT/model.safetensors.index.json"
test -f "$MODEL_ROOT/tokenizer.json"
test "$(find "$MODEL_ROOT" -maxdepth 1 -name 'model-*.safetensors' | wc -l)" -eq 4

du -sh "$MODEL_ROOT"
find "$MODEL_ROOT" -maxdepth 1 -type f -printf '%f\n' | sort
```

这里仅证明 snapshot 文件齐全，不证明 GPU 推理可用。下一步必须按照 `plan/execution_plan.md` 的 Phase 0 完成依赖安装、单视频 smoke test、显存和耗时测量。

## Step 5 — Install runtime dependencies only after the snapshot is verified

```bash
source /home/hdd-2t/zjy_dataset/videoitg_smart_clip/envs/videoitg/bin/activate
pip install -r /home/zjy/projects/videoitg_smart_clip/external/VideoITG/requirements.txt
```

官方 requirements 固定了 `torch==2.6.0`、`torchvision==0.21.0`（CUDA 12.4 index）以及 `transformers==4.47.1` 等版本。当前 Miniconda 推理环境实际使用 Transformers 5.8.1、cp313 Decord、`qwen-vl-utils==0.0.14` 和 `static-ffmpeg`；TimeLens adapter 会检测 `flash-attn`，缺失时使用 SDPA。安装后先做最小 import 检查，再决定是否安装训练专用的 `flash-attn`；本项目第一阶段只需要推理，不应预先安装训练依赖。复现部署可直接安装外置归档 wheel：`pip install --force-reinstall --no-deps /home/hdd-2t/zjy_dataset/videoitg_smart_clip/deps/decord-0.6.0-cp313-cp313-manylinux_2_39_x86_64.whl`。该 wheel 已通过 auditwheel 打包 FFmpeg 运行库并解决当前 Python 的 pip tag 检查；跨更老系统时需重新构建。

当前项目服务/评测命令应明确使用：

```bash
PYTHON=/home/zjy/miniconda3/bin/python
PYTHONPATH=src "$PYTHON" -c 'import torch, transformers, decord, static_ffmpeg, safetensors; print("runtime imports: ok")'
```

视频探测工具也安装在隔离环境内，不写入系统目录：

```bash
pip install "static-ffmpeg==3.0"
python - <<'PY'
import static_ffmpeg
static_ffmpeg.add_paths()
PY
```

`static_ffmpeg` 提供静态 `ffprobe`；用 `command -v ffprobe` 或环境内的绝对路径验证，避免依赖系统 apt 权限。

TimeLens 本地分片核验（主 Grounder）：

```bash
PYTHON=/home/zjy/miniconda3/bin/python
MODEL_ROOT=/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B
"$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/models/TimeLens-8B")
shards = sorted(set(json.loads((root / "model.safetensors.index.json").read_text())["weight_map"].values()))
assert all((root / name).is_file() for name in shards), shards
assert len(shards) == 4
print("TimeLens shards: ok", shards)
PY
```

## Expected layout after download

```text
/home/zjy/projects/videoitg_smart_clip/
└── external/VideoITG/           # 官方代码，体积相对小

/home/hdd-2t/zjy_dataset/videoitg_smart_clip/
├── cache/huggingface/            # HF 缓存
├── envs/videoitg/                # 隔离运行环境
└── models/VideoITG-8B/           # 本项目实际加载的本地 checkpoint
    ├── config.json
    ├── model-00001-of-00004.safetensors
    ├── model-00002-of-00004.safetensors
    ├── model-00003-of-00004.safetensors
    ├── model-00004-of-00004.safetensors
    ├── model.safetensors.index.json
    ├── tokenizer.json
    └── ...
```

## Do not download yet

- 不要现在下载 VideoITG-40K 全部视频；它不是完成本项目 Phase 0 的必要条件。
- 不需要另行下载 Qwen3-VL：TimeLens-8B snapshot 已包含其本地模型配置和 tokenizer；仅按 `docs/timelens_model.md` 核验 `AutoProcessor`/`AutoConfig` 与本地候选窗口推理。
- 不要把 checkpoint 放进项目仓库或 `/home/zjy/projects/videoitg_smart_clip/data`。
