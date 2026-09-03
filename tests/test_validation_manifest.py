import pytest

from videoitg_smart_clip.evaluation.validation import ValidationManifestError, validate_manifest_rows


def _row(sample_id: str, actual_match: bool, negative_type: str = "none", status: str = "complete") -> dict:
    return {"sample_id": sample_id, "video_id": sample_id, "query": "event", "actual_match": actual_match, "negative_type": negative_type, "ground_truth": [[0.0, 1.0]] if actual_match else [], "split": "dev", "label_status": status}


def test_validation_manifest_requires_negative_coverage_for_gate():
    rows = [_row("p", True)] + [_row(name, False, name) for name in ("event_absent", "wrong_action", "wrong_object")]
    with pytest.raises(ValidationManifestError, match="missing required negative"):
        validate_manifest_rows(rows, require_negative_categories=True)


def test_validation_manifest_reports_complete_coverage():
    rows = [_row("p", True)] + [_row(name, False, name) for name in ("event_absent", "wrong_action", "wrong_object", "theme_unrelated")]
    summary = validate_manifest_rows(rows, require_complete=True, require_negative_categories=True)
    assert summary["complete"] is True
    assert summary["negative_coverage_complete"] is True


def test_validation_manifest_accepts_synthetic_rows_as_offline_complete():
    rows = [_row("p", True)] + [_row(name, False, name, status="synthetic") for name in ("event_absent", "wrong_action", "wrong_object", "theme_unrelated")]
    summary = validate_manifest_rows(rows, require_complete=True, require_negative_categories=True)
    assert summary["complete"] is True
    assert summary["synthetic_rows"] == 4


def test_validation_manifest_rejects_invalid_ground_truth_bounds():
    row = _row("bad", True)
    row["ground_truth"] = [[2.0, 1.0]]
    with pytest.raises(ValidationManifestError, match="invalid bounds"):
        validate_manifest_rows([row])


def test_validation_manifest_can_require_local_video_files(tmp_path):
    row = _row("file", True)
    row["video_path"] = str(tmp_path / "missing.mp4")
    with pytest.raises(ValidationManifestError, match="does not exist"):
        validate_manifest_rows([row], require_files=True)
