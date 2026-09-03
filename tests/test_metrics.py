from videoitg_smart_clip.evaluation.metrics import calibrate_no_match_thresholds, duplicate_rate, evaluate_sample, feedback_product_metrics, gt_segments_from_clip_num, no_match_metrics, segment_iou, top_k_useful_rate


def test_eval_config_is_loaded_by_unified_entrypoint(tmp_path):
    import importlib.util

    module_path = __import__("pathlib").Path(__file__).parents[1] / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("videoitg_eval_entrypoint", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = tmp_path / "eval.yaml"
    config.write_text("evaluation:\n  output_top_k: 2\n  retriever_top_n: [3, 7]\n  duplicate_iou_threshold: 0.6\n  data_version: d1\n  split_version: s1\n", encoding="utf-8")
    loaded = module._load_eval_config(config)
    assert loaded["output_top_k"] == 2
    assert loaded["retriever_top_n"] == [3, 7]
    assert loaded["duplicate_iou_threshold"] == 0.6
    assert loaded["data_version"] == "d1"


def test_empty_eval_keeps_complete_metric_schema(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys

    module_path = __import__("pathlib").Path(__file__).parents[1] / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("videoitg_eval_empty", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = tmp_path / "eval.yaml"
    config.write_text("evaluation:\n  output_top_k: 3\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", str(config), "--output-dir", str(output_dir)])
    module.main()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    required = {
        "recall_at_1_iou_0.3", "recall_at_1_iou_0.5", "recall_at_1_iou_0.7",
        "miou", "mean_boundary_error_s", "duplicate_rate", "top_k_useful_rate",
        "no_match_accuracy", "false_positive_rate", "false_negative_rate",
        "failure_rate", "degrade_rate", "user_adoption_rate",
        "mean_manual_boundary_adjustment_seconds",
    }
    assert required.issubset(metrics)
    assert all(metrics[key] is None for key in required)


def test_validation_manifest_scaffold_requires_explicit_class(tmp_path):
    import importlib.util

    module_path = __import__("pathlib").Path(__file__).parents[1] / "scripts" / "build_validation_manifest.py"
    spec = importlib.util.spec_from_file_location("validation_scaffold", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    source = [{
        "sample_id": "s1", "video_id": "v1", "video_path": "/tmp/v.mp4",
        "query": "find the event", "clip_num": [0, 1, 3], "split": "dev",
    }]
    rows = module.build_rows(source, match_class="present")
    assert rows[0]["actual_match"] is True
    assert rows[0]["negative_type"] == "none"
    assert rows[0]["ground_truth"] == [[0.0, 10.0], [15.0, 20.0]]
    assert rows[0]["label_status"] == "pending"
    try:
        module.build_rows(source, match_class="present", label_status="complete")
    except TypeError:
        # The public scaffold API intentionally has no complete-status
        # override; callers must promote labels through human review.
        pass
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("scaffold must not accept a complete-status override")
    synthetic = module.build_rows(source, match_class="absent", negative_type="wrong_action", synthetic=True)
    assert synthetic[0]["actual_match"] is False
    assert synthetic[0]["label_status"] == "synthetic"
    assert synthetic[0]["scaffold_assumption"] == "synthetic_absent"
    try:
        module.build_rows(source, match_class="absent")
    except ValueError as exc:
        assert "negative types" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("absent scaffold must require an explicit negative type")


def test_no_match_metrics_uses_match_presence_labels():
    # [present, absent, present, absent] with two correct NO_MATCH decisions.
    result = no_match_metrics(["CONFIDENT", "NO_MATCH", "NO_MATCH", "POSSIBLE"], [True, False, True, False])
    assert result == {"no_match_accuracy": 0.5, "false_positive_rate": 0.5, "false_negative_rate": 0.5}


def test_no_match_threshold_calibration_requires_both_classes():
    result = calibrate_no_match_thresholds([
        {"retrieval_score": 0.9, "grounding_score": 0.9, "actual_match": True},
        {"retrieval_score": 0.1, "grounding_score": 0.2, "actual_match": False},
    ])
    assert result["validation_count"] == 2
    assert result["metrics"]["no_match_accuracy"] == 1.0


def test_no_match_calibration_can_use_empty_predictions_as_negative_evidence():
    result = calibrate_no_match_thresholds([
        {"retrieval_score": 0.9, "grounding_score": 0.9, "actual_match": True},
        {"retrieval_score": 0.2, "grounding_score": 0.0, "actual_match": False},
    ])
    assert result["validation_count"] == 2


def test_eval_does_not_report_no_match_rates_for_single_class_pilot(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys

    module_path = __import__("pathlib").Path(__file__).parents[1] / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("videoitg_eval_single_class", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = tmp_path / "eval.yaml"
    config.write_text("evaluation:\n  output_top_k: 1\n", encoding="utf-8")
    input_path = tmp_path / "pilot.jsonl"
    input_path.write_text(json.dumps({
        "ground_truth": [[0.0, 5.0]],
        "predictions": [{"start_s": 0.0, "end_s": 5.0}],
        "coarse_windows": [{"start": 0.0, "end": 5.0, "score": 0.9}],
        "status": "POSSIBLE",
        "actual_match": True,
    }) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", str(config), "--input", str(input_path), "--output-dir", str(output_dir)])
    module.main()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["no_match_accuracy"] is None
    assert metrics["false_positive_rate"] is None
    assert metrics["false_negative_rate"] is None


def test_gt_clip_numbers_are_five_second_intervals():
    assert gt_segments_from_clip_num([0, 1, 3]) == [[0.0, 10.0], [15.0, 20.0]]


def test_segment_iou_and_topk_thresholds():
    assert segment_iou([0, 5], [0, 5]) == 1.0
    metrics = evaluate_sample(
        [{"start_s": 10, "end_s": 15}, {"start_s": 0, "end_s": 5}],
        [[0, 5]],
        output_top_k=2,
    )
    assert metrics["recall_at_1_iou_0.3"] == 0
    assert metrics["recall_at_1_iou_0.7"] == 0
    assert metrics["topk_hit_iou_0.3"] == 1
    assert "miou" in metrics and "mean_boundary_error_s" in metrics


def test_metrics_accept_lossless_pipeline_bounds():
    metrics = evaluate_sample(
        [{"refined_start": 0.0, "refined_end": 5.0}],
        [[0.0, 5.0]],
        output_top_k=1,
    )
    assert metrics["recall_at_1_iou_0.7"] == 1
    assert metrics["miou"] == 1.0


def test_duplicate_rate_uses_pipeline_deduplication_marker():
    rows = [
        {"refined_start": 0.0, "refined_end": 5.0, "deduplicated": False},
        {"refined_start": 0.5, "refined_end": 5.5, "deduplicated": True},
    ]
    assert duplicate_rate(rows) == 0.5


def test_product_metrics_compute_useful_rate_adoption_and_adjustment():
    rows = [
        {"useful": True, "label": "ACCEPT", "model_start": 1, "model_end": 3, "user_start": 2, "user_end": 4},
        {"useful": False, "label": "IRRELEVANT"},
    ]
    assert top_k_useful_rate(rows) == 0.5
    assert feedback_product_metrics(rows) == {"user_adoption_rate": 0.5, "mean_manual_boundary_adjustment_seconds": 1.0}
