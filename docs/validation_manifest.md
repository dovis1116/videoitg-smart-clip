# G8 验证清单生成与自动评测边界

当前项目是离线原型，不执行人工标注或真实用户验收。仓库提供
`scripts/build_validation_manifest.py` 生成显式的 present/absent 清单：默认行是
`label_status=pending`；使用 `--synthetic` 可将明确构造的负样本标为
`label_status=synthetic`。脚本不会把数据集字段自动推断为真实负样本。
Synthetic 清单可用于 No-Match/Ranking 阈值逻辑、状态机和相对对比测试，报告中
必须保留 synthetic 标记，不能解释为真实世界质量。

## 生成正样本 scaffold

```bash
PYTHONPATH=src /home/zjy/miniconda3/bin/python scripts/build_validation_manifest.py \
  records/phase4/boundary_train_dev_manifest.jsonl \
  records/phase_g8/validation_present_pending.jsonl \
  --match-class present --split dev
```

`ground_truth_segments` 会原样保留；只有旧数据提供 `clip_num` 时才按 5 秒连续区间转换。
这只是数据格式转换，不是人工语义复核。

## 生成合成负样本

每个负样本族需显式声明源视频/Query 和构造规则。示例：

```bash
PYTHONPATH=src /home/zjy/miniconda3/bin/python scripts/build_validation_manifest.py \
  <pool.jsonl> <out.jsonl> --match-class absent \
  --negative-type wrong_action --split dev --synthetic
```

可用类型为 `event_absent`、`wrong_action`、`wrong_object`、`theme_unrelated`。
`--synthetic` 只表示“按命令显式构造的离线负样本”，不表示人工确认或真实负样本。
生成后由自动校验器检查字段、类别覆盖和 `actual_match` 一致性。

## 离线校验与评测

```bash
PYTHONPATH=src /home/zjy/miniconda3/bin/python scripts/validate_validation_manifest.py \
  records/phase_g8/validation_manifest.jsonl \
  --require-complete --require-negative-categories --require-files
PYTHONPATH=src /home/zjy/miniconda3/bin/python eval/run_eval.py \
  --config configs/eval.yaml --input <predictions.jsonl> --output-dir <eval-output>
```

对于 synthetic 清单，可以运行上述校验器的 `--require-complete --require-negative-categories`
并将预测交给 `eval/run_eval.py`，得到合成数据上的 No-Match 指标。报告必须写明
`data_type=synthetic`。这验证的是离线逻辑和相对差异，不是生产部署、真实用户体验或
真实世界 No-Match Accuracy/FPR/FNR。
