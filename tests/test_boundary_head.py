from videoitg_smart_clip.boundary_head import calibrated_predictions, fit_boundary_head, frame_feature_matrix


def _row(scores):
    return {
        "fps": 1.0,
        "duration_s": 20.0,
        "frame_scores": [{"frame_index": index, "score": score} for index, score in enumerate(scores)],
        "ground_truth_segments": [[5.0, 10.0]],
    }


def test_boundary_head_features_are_temporally_sorted_and_bounded():
    matrix, frames = frame_feature_matrix(_row([0.1, 0.9, 0.2]))
    assert [item["frame_index"] for item in frames] == [0, 1, 2]
    assert matrix.shape == (3, 11)


def test_boundary_head_fit_and_apply_keep_nonoverlap():
    rows = [_row([0.1, 0.9, 0.2]), _row([0.2, 0.8, 0.1])]
    model = fit_boundary_head(rows)
    predictions = calibrated_predictions(rows[0], model)
    assert model["format"] == "videoitg_boundary_head_v1"
    assert 1 <= len(predictions) <= 3
    for left, right in zip(predictions, predictions[1:]):
        assert left["end_s"] <= right["start_s"] or right["end_s"] <= left["start_s"]
