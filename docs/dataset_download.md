# VideoITG-40K Dataset Download

这里假设你要下载的是官方 `NVEagle/VideoITG-40K`。架构迁移后该数据仅用于 VideoITG baseline/格式审计和 smoke test，不作为 TimeLens 主路径证据，也不应直接当成最终长视频业务数据集。数据卡给出的平均视频时长约 120 秒、范围约 30 秒到 3 分钟，和本项目目标的 5–30 分钟长视频并不相同。

## Paths

```text
数据集元数据（推荐先下载）：
/home/hdd-2t/zjy_dataset/videoitg_smart_clip/annotations/VideoITG-40K

Hugging Face 缓存：
/home/hdd-2t/zjy_dataset/videoitg_smart_clip/cache/huggingface

项目原始视频（后续自行筛选，不能由本命令自动得到）：
/home/hdd-2t/zjy_dataset/videoitg_smart_clip/raw
```

## Recommended: download annotation metadata first

先下载主标注 JSON、README 和官方处理脚本，不下载较大的完整 clip description 文件：

```bash
export DATA_ROOT=/home/hdd-2t/zjy_dataset/videoitg_smart_clip
export DATASET_ROOT=$DATA_ROOT/annotations/VideoITG-40K
export HF_HOME=$DATA_ROOT/cache/huggingface

source $DATA_ROOT/envs/videoitg/bin/activate
mkdir -p "$DATASET_ROOT"

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NVEagle/VideoITG-40K",
    repo_type="dataset",
    local_dir="/home/hdd-2t/zjy_dataset/videoitg_smart_clip/annotations/VideoITG-40K",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "README.md",
        "video_itg_data.json",
        "*.py",
    ],
)
PY
```

## Verify the downloaded metadata

```bash
DATASET_ROOT=/home/hdd-2t/zjy_dataset/videoitg_smart_clip/annotations/VideoITG-40K

test -s "$DATASET_ROOT/README.md"
test -s "$DATASET_ROOT/video_itg_data.json"
python - <<'PY'
import json
from pathlib import Path

p = Path("/home/hdd-2t/zjy_dataset/videoitg_smart_clip/annotations/VideoITG-40K/video_itg_data.json")
with p.open() as f:
    data = json.load(f)
assert isinstance(data, list), type(data)
assert data, "empty dataset"
required = {"id", "video", "question", "answer"}
missing = required - set(data[0])
assert not missing, missing
print({"records": len(data), "first_keys": sorted(data[0])})
PY
```

## Optional: download the full metadata snapshot

只有需要研究 clip 描述或复核数据构造时再下载 `all_clip_desc.json`：

```bash
export DATA_ROOT=/home/hdd-2t/zjy_dataset/videoitg_smart_clip
export DATASET_ROOT=$DATA_ROOT/annotations/VideoITG-40K
export HF_HOME=$DATA_ROOT/cache/huggingface

source $DATA_ROOT/envs/videoitg/bin/activate

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NVEagle/VideoITG-40K",
    repo_type="dataset",
    local_dir="/home/hdd-2t/zjy_dataset/videoitg_smart_clip/annotations/VideoITG-40K",
    local_dir_use_symlinks=False,
)
PY
```

当前官方数据仓库 API 显示的主文件名是 `video_itg_data.json`；不要照抄部分旧说明中的 `videoitg_data.json`。

## Important limitation: this is not the video download

`VideoITG-40K` 仓库发布的是标注/描述和处理脚本，数据卡说明视频源自 LLaVA-Video；上述命令不会自动把 40,000 个视频下载到 `raw/`。下载后先检查 `video` 字段中的路径/ID，再根据来源许可和官方来源说明逐项获取可用视频。

## Where the source videos live

部分源视频媒体发布在另一个数据集：[`lmms-lab/LLaVA-Video-178K`](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)。该仓库包含按 split 划分的 `.tar.gz` 视频分片，完整仓库页面显示体积约 699GB；数据卡限制为 academic research and education use。

不要直接下载整个 699GB 仓库。应先从 `video_itg_data.json` 的 `video` 字段统计所需 split，再只下载一个小分片做 smoke test。例如，确认样本属于 `2_3_m_youtube_v0_1` 后，可以先下载一个候选 shard：

```bash
export DATA_ROOT=/home/hdd-2t/zjy_dataset/videoitg_smart_clip
export MEDIA_ROOT=$DATA_ROOT/raw/LLaVA-Video-178K
export HF_HOME=$DATA_ROOT/cache/huggingface

source $DATA_ROOT/envs/videoitg/bin/activate
mkdir -p "$MEDIA_ROOT"

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lmms-lab/LLaVA-Video-178K",
    repo_type="dataset",
    local_dir="/home/hdd-2t/zjy_dataset/videoitg_smart_clip/raw/LLaVA-Video-178K",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "2_3_m_youtube_v0_1/2_3_m_youtube_v0_1_videos_1.tar.gz",
    ],
)
PY
```

这只是一个候选分片，不保证包含你选中的 VideoITG 样本。下载后解压并用 `video` 相对路径做匹配：

```bash
tar -xzf \
  "$MEDIA_ROOT/2_3_m_youtube_v0_1/2_3_m_youtube_v0_1_videos_1.tar.gz" \
  -C "$MEDIA_ROOT/2_3_m_youtube_v0_1"
```

如果分片中没有目标视频，应先根据 manifest 映射选择正确的 `videos_N.tar.gz`，不要盲目扩大下载范围。

本次 Phase 1 实际只下载并解压了两个已核验分片：

```text
2_3_m_nextqa/2_3_m_nextqa_videos_1.tar.gz       184 MB
30_60_s_nextqa/30_60_s_nextqa_videos_2.tar.gz   1.35 GB
```

随后为扩展来源覆盖，又下载了：

```text
30_60_s_activitynetqa/30_60_s_activitynetqa_videos_1.tar.gz  3.37 GB
```

解压时分别放入对应 source group 目录，使 tar 内的 `NextQA/...` 路径能与标注中的
`LLaVA-Video-178K/<source_group>/NextQA/...` 对齐。当前形成 50 条真实媒体 pilot，
不是完整数据集下载。

项目后续数据流程应是：

```text
VideoITG-40K metadata audit
    -> select legally accessible videos
    -> copy/download raw videos to raw/
    -> normalize to project schema
    -> video_id-level train/dev/test split
    -> manually audit hard cases and no-match samples
```

不要把自动生成的 VideoITG 标注直接当作最终困难集或业务线上标签；它们需要经过抽样审计。
