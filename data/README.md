# Data Policy and Layout

本目录不存放大型视频、帧、特征、索引或模型权重。

大文件根目录：

```text
/home/hdd-2t/zjy_dataset/videoitg_smart_clip
```

仓库内允许保存：

- schema；
- 小型、去敏的 manifest；
- 不包含媒体内容的示例记录；
- 数据审计摘要；
- 许可和来源说明。

任何公开数据下载前先记录来源、许可、版本、校验和与下载日期。任何人工或合成无匹配样本都必须保留构造规则与抽样复核证据。

官方 VideoITG-40K 元数据、LLaVA-Video-178K 源视频分片，以及“不要直接下载完整媒体仓库”的限制见 [`../docs/dataset_download.md`](../docs/dataset_download.md)。
