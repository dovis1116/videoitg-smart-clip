---
configs:
- config_name: 0_30_s_academic_v0_1
  data_files:
  - split: caption
    path: 0_30_s_academic_v0_1/*cap*.json
  - split: open_ended
    path: 0_30_s_academic_v0_1/*oe*.json
  - split: multi_choice
    path: 0_30_s_academic_v0_1/*mc*.json
- config_name: 0_30_s_youtube_v0_1
  data_files:
  - split: caption
    path: 0_30_s_youtube_v0_1/*cap*.json
  - split: open_ended
    path: 0_30_s_youtube_v0_1/*oe*.json
  - split: multi_choice
    path: 0_30_s_youtube_v0_1/*mc*.json
- config_name: 0_30_s_activitynet
  data_files:
  - split: open_ended
    path: 0_30_s_activitynet/*oe*.json
- config_name: 0_30_s_perceptiontest
  data_files:
  - split: multi_choice
    path: 0_30_s_perceptiontest/*mc*.json
- config_name: 0_30_s_nextqa
  data_files:
  - split: open_ended
    path: 0_30_s_nextqa/*oe*.json
  - split: multi_choice
    path: 0_30_s_nextqa/*mc*.json
- config_name: 30_60_s_academic_v0_1
  data_files:
  - split: caption
    path: 30_60_s_academic_v0_1/*cap*.json
  - split: open_ended
    path: 30_60_s_academic_v0_1/*oe*.json
  - split: multi_choice
    path: 30_60_s_academic_v0_1/*mc*.json
- config_name: 30_60_s_youtube_v0_1
  data_files:
  - split: caption
    path: 30_60_s_youtube_v0_1/*cap*.json
  - split: open_ended
    path: 30_60_s_youtube_v0_1/*oe*.json
  - split: multi_choice
    path: 30_60_s_youtube_v0_1/*mc*.json
- config_name: 30_60_s_activitynet
  data_files:
  - split: open_ended
    path: 30_60_s_activitynet/*oe*.json
- config_name: 30_60_s_perceptiontest
  data_files:
  - split: multi_choice
    path: 30_60_s_perceptiontest/*mc*.json
- config_name: 30_60_s_nextqa
  data_files:
  - split: open_ended
    path: 30_60_s_nextqa/*oe*.json
  - split: multi_choice
    path: 30_60_s_nextqa/*mc*.json
- config_name: 1_2_m_youtube_v0_1
  data_files:
  - split: caption
    path: 1_2_m_youtube_v0_1/*cap*.json
  - split: open_ended
    path: 1_2_m_youtube_v0_1/*oe*.json
  - split: multi_choice
    path: 1_2_m_youtube_v0_1/*mc*.json
- config_name: 1_2_m_academic_v0_1
  data_files:
  - split: caption
    path: 1_2_m_academic_v0_1/*cap*.json
  - split: open_ended
    path: 1_2_m_academic_v0_1/*oe*.json
  - split: multi_choice
    path: 1_2_m_academic_v0_1/*mc*.json
- config_name: 1_2_m_activitynet
  data_files:
  - split: open_ended
    path: 1_2_m_activitynet/*oe*.json
- config_name: 1_2_m_nextqa
  data_files:
  - split: open_ended
    path: 1_2_m_nextqa/*oe*.json
  - split: multi_choice
    path: 1_2_m_nextqa/*mc*.json
- config_name: 2_3_m_youtube_v0_1
  data_files:
  - split: caption
    path: 2_3_m_youtube_v0_1/*cap*.json
  - split: open_ended
    path: 2_3_m_youtube_v0_1/*oe*.json
  - split: multi_choice
    path: 2_3_m_youtube_v0_1/*mc*.json
- config_name: 2_3_m_academic_v0_1
  data_files:
  - split: caption
    path: 2_3_m_academic_v0_1/*cap*.json
  - split: open_ended
    path: 2_3_m_academic_v0_1/*oe*.json
  - split: multi_choice
    path: 2_3_m_academic_v0_1/*mc*.json
- config_name: 2_3_m_activitynet
  data_files:
  - split: open_ended
    path: 2_3_m_activitynet/*oe*.json
- config_name: 2_3_m_nextqa
  data_files:
  - split: open_ended
    path: 2_3_m_nextqa/*oe*.json
  - split: multi_choice
    path: 2_3_m_nextqa/*mc*.json
- config_name: llava_hound
  data_files:
  - split: open_ended
    path: llava_hound/sharegptvideo_qa_255k_processed.json
language:
- en
task_categories:
- visual-question-answering
- video-text-to-text
tags:
- video
---



# Dataset Card for LLaVA-Video-178K

## Dataset Description
- **Curated by:** Yuanhan Zhang, Jinming Wu, Wei Li
- **Language(s) (NLP):** English, Chinese
- **License:** Apache License 2.0

## Uses
This dataset is used for the training of the LLaVA-Video model. We only allow the use of this dataset for academic research and education purpose. For OpenAI GPT-4 generated data, we recommend the users to check the [OpenAI Usage Policy](https://openai.com/policies/usage-policies/).

### Data Sources
For the training of LLaVA-Video, we utilized video-language data from five primary sources:
- **LLaVA-Video-178K**: This dataset includes **178,510** caption entries, 960,792 open-ended QA (question and answer) items, and 196,198 multiple-choice QA items. These data were newly annotated for this project.
  - We include this dataset in this repository: LLaVA-Video-178K/XXX_academic_v0_1 and LLaVA-Video-178K/XXX_youtube_v0_1.
- **NeXT-QA**: Comprises 17,090 open-ended QA items and 17,024 multiple-choice QA items.
  - We include this dataset in this repository: LLaVA-Video-178K/XXX_nextqa.
- **ActivityNetQA**: Includes 23,530 open-ended QA items,
  - We include this dataset in this repository: LLaVA-Video-178K/XXX_activitynetqa.
- **PerceptionTest**: Includes 1,803 open-ended QA items.
  - We include this dataset in this repository: LLaVA-Video-178K/XXX_perceptiontest.
- **LLaVA-Hound**: Contains 240,000 open-ended QA items and 15,000 caption entries.
  - The video data and annotations are available at the following URLs:
  - Video data: [train_300k](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/train_300k)
  - Annotation data: LLaVA-Video-178K/llava_hound
  - loading function is specified here: [function](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/7125e3654d88063cb467ed242db76f1e2b184d4c/llava/train/train.py#L1162)
 
The **LLaVA-Video-178K** dataset is the only contribution from this repository; we provide additional datasets for reproducing LLaVA-Video.

- **Project Page:** [Project Page](https://llava-vl.github.io/blog/2024-09-30-llava-video/).
- **Paper**: For more details, please check our [paper](https://arxiv.org/abs/2410.02713) 

### Annotation Pipeline
The following directories are provided for generating captions and QA data:
- **Captions**: `LLaVA-Video-178K/gpt4o_caption_prompt`
- **QA**: `LLaVA-Video-178K/gpt4o_qa_prompt`

### The subset used in the LLaVA-OneVision
We have included captions and open-ended questions in the [0_30_s_academic_v0_1 split](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main/0_30_s_academic_v0_1), along with 240,000 open-ended QA items and 15,000 caption entries, as part of the video data in LLaVA-Hound for LLaVA-OneVision.
- [**0_30_s_academic_v0_1 caption**](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/blob/main/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.json)
- [**0_30_s_academic_v0_1 open-ended QA**](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/blob/main/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.json)
- **LLaVA-Hound**: Same as above.



## Citation

```bibtex

@misc{zhang2024videoinstructiontuningsynthetic,
    title={Video Instruction Tuning With Synthetic Data}, 
    author={Yuanhan Zhang and Jinming Wu and Wei Li and Bo Li and Zejun Ma and Ziwei Liu and Chunyuan Li},
    year={2024},
    eprint={2410.02713},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2410.02713}, 
}
```

## Dataset Card Contact

[Yuanhan Zhang](https://zhangyuanhan-ai.github.io/)

[Jinming Wu](https://scholar.google.com/citations?user=eh-XJIoAAAAJ&hl=zh-CN)

[Wei Li](https://scholar.google.com/citations?user=q8ZrKVIAAAAJ&hl=zh-CN)