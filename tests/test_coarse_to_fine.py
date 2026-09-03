import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor

from videoitg_smart_clip.grounding import StubTimeLensGrounder, TimeLensGrounder
from videoitg_smart_clip.pipeline import CandidateWindow, CoarseToFinePipeline, GroundingPrediction
from videoitg_smart_clip.pipeline.postprocess import BoundaryRefiner, CandidateRanker, NoMatchDecider, TemporalDeduplicator
from videoitg_smart_clip.preprocessing import FeatureCache, HashFeatureEncoder, cache_identity
from videoitg_smart_clip.retrieval import CachedCosineRetriever, normalize_retrieval_query


def test_feature_cache_reuses_video_side_encoding(tmp_path):
    cache = FeatureCache(tmp_path)
    key = cache_identity("video-1", "encoder-v1", {"segment": "shot"}, {"fps": 1})
    encoder = HashFeatureEncoder()
    calls = []

    def extract():
        calls.append(1)
        return encoder.encode([np.zeros((2, 2, 3), dtype=np.uint8)]), [0.0]

    first = cache.get_or_create(key, video_duration=1, extractor=extract)
    second = cache.get_or_create(key, video_duration=1, extractor=extract)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1


def test_feature_cache_persists_extraction_latency_for_new_entries(tmp_path):
    import time

    cache = FeatureCache(tmp_path)
    key = cache_identity("video-latency", "encoder-v1", {"segment": "shot"}, {"fps": 1})

    def extract():
        time.sleep(0.001)
        return np.ones((1, 4), dtype=np.float32), [0.0]

    created = cache.get_or_create(key, video_duration=1, extractor=extract)
    loaded = cache.get_or_create(key, video_duration=1, extractor=extract)
    assert created.extraction_latency_ms is not None
    assert created.extraction_latency_ms >= 0.0
    assert loaded.extraction_latency_ms == created.extraction_latency_ms


def test_feature_cache_cross_instance_lock_prevents_duplicate_extraction(tmp_path):
    cache_a = FeatureCache(tmp_path)
    cache_b = FeatureCache(tmp_path)
    key = cache_identity("video-concurrent", "encoder-v1", {"segment": "shot"}, {"fps": 1})
    encoder = HashFeatureEncoder()
    calls = []

    def extract():
        calls.append(1)
        # Keep the first writer inside the lock long enough for the second
        # cache instance to contend on the same key.
        import time

        time.sleep(0.03)
        return encoder.encode([np.zeros((2, 2, 3), dtype=np.uint8)]), [0.0]

    def run(cache):
        return cache.get_or_create(key, video_duration=1, extractor=extract)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(run, (cache_a, cache_b))
    assert len(calls) == 1
    assert {first.cache_hit, second.cache_hit} == {False, True}


def test_retrieval_query_normalizes_dataset_answer_boilerplate():
    assert normalize_retrieval_query("find the event\nAnswer the question using few words or phrase.") == "find the event"


def test_pipeline_preserves_raw_and_refined_bounds(tmp_path):
    cache = FeatureCache(tmp_path)
    retriever = CachedCosineRetriever(cache)
    key = cache_identity("v", "m", {}, {})
    retriever.index("v.mp4", "v", key=key, video_duration=20, extractor=lambda: (np.ones((2, 32), dtype=np.float32), [5, 15]))
    result = CoarseToFinePipeline(retriever, StubTimeLensGrounder(), top_n=2, top_k=2).search("v.mp4", "v", "event")
    assert result["status"] in {"CONFIDENT", "POSSIBLE"}
    assert all("raw_start" in row and "refined_start" in row for row in result["predictions"])


def test_temporal_dedup_keeps_highest_score():
    dedup = TemporalDeduplicator(temporal_iou_threshold=0.7)
    rows = [
        {"candidate_id": "a", "refined_start": 0, "refined_end": 10, "final_score": 0.5},
        {"candidate_id": "b", "refined_start": 1, "refined_end": 10, "final_score": 0.9},
    ]
    assert [row["candidate_id"] for row in dedup.apply(rows)] == ["b"]


def test_timelens_timestamp_parser_clamps_local_window():
    assert TimeLensGrounder.parse_timestamps("The event happens in 2.5 - 9.0 seconds", 5.0) == (2.5, 5.0)


def test_timelens_materialize_normalizes_odd_video_dimensions(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        assert check is True

    monkeypatch.setattr("videoitg_smart_clip.grounding.timelens.subprocess.run", fake_run)
    grounder = TimeLensGrounder()
    monkeypatch.setattr(grounder, "_ffmpeg", lambda: "ffmpeg")
    output = grounder._materialize_window("odd.mp4", CandidateWindow(1.0, 6.0, 0.5, "odd"), tmp_path)
    assert output.name == "odd.mp4"
    assert "scale=trunc(iw/2)*2:trunc(ih/2)*2" in captured["command"]


def test_timelens_without_checkpoint_fails_explicitly():
    grounder = TimeLensGrounder()
    with pytest.raises(RuntimeError, match="model_path is required"):
        grounder.predict("missing.mp4", "event", [CandidateWindow(0.0, 5.0, 0.5, "c")])


def test_ranking_uses_boundary_confidence():
    ranker = CandidateRanker()
    high = GroundingPrediction("high", 0, 5, 1.0, boundary_confidence=1.0, completeness_score=0.5)
    low = GroundingPrediction("low", 0, 5, 1.0, boundary_confidence=0.0, completeness_score=0.5)
    assert ranker.score(0.5, high) > ranker.score(0.5, low)


def test_ranking_and_no_match_reject_invalid_configuration():
    with pytest.raises(ValueError, match="ranking weights"):
        CandidateRanker(weights={"retrieval": float("nan"), "grounding": 0.4, "completeness": 0.2})
    with pytest.raises(ValueError, match="grounding_threshold"):
        NoMatchDecider(grounding_threshold=1.5)


def test_pipeline_preserves_duplication_penalty_input(tmp_path):
    cache = FeatureCache(tmp_path)
    retriever = CachedCosineRetriever(cache, window_seconds=10.0)
    key = cache_identity("v", "m", {}, {})
    retriever.index("v.mp4", "v", key=key, video_duration=30, extractor=lambda: (np.ones((3, 32), dtype=np.float32), [5, 10, 15]))
    result = CoarseToFinePipeline(retriever, StubTimeLensGrounder(), top_n=3, top_k=3).search("v.mp4", "v", "event")
    assert all("duplication_penalty" in row for row in result["candidates"])


def test_pipeline_keeps_deduplicated_candidates_for_diagnostics(tmp_path):
    cache = FeatureCache(tmp_path)
    retriever = CachedCosineRetriever(cache, window_seconds=10.0)
    key = cache_identity("v", "m", {}, {})
    retriever.index("v.mp4", "v", key=key, video_duration=20, extractor=lambda: (np.ones((2, 32), dtype=np.float32), [5, 6]))
    result = CoarseToFinePipeline(retriever, StubTimeLensGrounder(), top_n=2, top_k=1).search("v.mp4", "v", "event")
    assert len(result["candidates"]) == 2
    assert any(row["deduplicated"] for row in result["candidates"])
    assert len(result["predictions"]) == 1
    assert all(row["rank"] is not None and row["pre_dedup_rank"] is not None for row in result["candidates"])


def test_boundary_refinement_clamps_to_video_duration():
    from videoitg_smart_clip.pipeline.postprocess import BoundaryRefiner

    prediction = GroundingPrediction("x", 8.0, 10.0, 1.0)
    refined = BoundaryRefiner(enabled=True, expansion_seconds=2.0).refine(prediction, duration=10.0)
    assert refined["refined_start"] == 6.0
    assert refined["refined_end"] == 10.0


def test_boundary_refinement_supports_directional_offsets_and_padding():
    prediction = GroundingPrediction("x", 8.0, 10.0, 1.0)
    refined = BoundaryRefiner(enabled=True, start_offset_seconds=-1.0, end_offset_seconds=1.0, start_padding_seconds=0.5, end_padding_seconds=0.5).refine(prediction, duration=20.0)
    assert refined["refined_start"] == 6.5
    assert refined["refined_end"] == 11.5


def test_boundary_refinement_rejects_non_finite_or_negative_padding():
    from videoitg_smart_clip.pipeline.postprocess import BoundaryRefiner

    with pytest.raises(ValueError):
        BoundaryRefiner(expansion_seconds=-0.1)
    with pytest.raises(ValueError):
        BoundaryRefiner(start_offset_seconds=float("nan"))


def test_boundary_refinement_reports_invalid_collapsed_interval():
    prediction = GroundingPrediction("x", 2.0, 3.0, 1.0)
    from videoitg_smart_clip.pipeline.postprocess import BoundaryRefiner

    with pytest.raises(ValueError, match="start < end"):
        BoundaryRefiner(start_offset_seconds=2.0, end_offset_seconds=-2.0).refine(prediction)


def test_no_match_uses_retrieval_score_when_available():
    decider = NoMatchDecider(retrieval_threshold=0.5, grounding_threshold=0.5, margin_threshold=0.1)
    assert decider.decide([{"final_score": 0.9, "retrieval_score": 0.2, "grounding_score": 0.9}]) == "NO_MATCH"


def test_pipeline_does_not_force_candidates_for_no_match(tmp_path):
    cache = FeatureCache(tmp_path)
    retriever = CachedCosineRetriever(cache)
    key = cache_identity("v", "m", {}, {})
    retriever.index("v.mp4", "v", key=key, video_duration=20, extractor=lambda: (np.ones((1, 32), dtype=np.float32), [5]))
    result = CoarseToFinePipeline(
        retriever,
        StubTimeLensGrounder(),
        no_match=NoMatchDecider(retrieval_threshold=0.9, grounding_threshold=0.4, margin_threshold=0.1),
        top_n=1,
        top_k=1,
    ).search("v.mp4", "v", "event")
    assert result["status"] == "NO_MATCH"
    assert result["predictions"] == []
    assert result["candidates"] and result["candidates"][0]["raw_start"] == 0.0


def test_empty_retrieval_short_circuits_grounder_as_no_match():
    class EmptyRetriever:
        version = "empty-v1"

        def retrieve(self, video_id, query, top_n):
            return []

    class MustNotRunGrounder:
        model_version = "must-not-run"

        def predict(self, video_path, query, candidate_windows):
            raise AssertionError("grounder must not run for empty retrieval")

    result = CoarseToFinePipeline(EmptyRetriever(), MustNotRunGrounder()).search("missing.mp4", "v", "absent")
    assert result == {
        "status": "NO_MATCH",
        "predictions": [],
        "candidates": [],
        "degraded": False,
        "degrade_level": 0,
        "degrade_reason": None,
    }


def test_pipeline_assigns_unique_candidate_ids_when_retriever_omits_them():
    windows = [CandidateWindow(0.0, 5.0, 0.4), CandidateWindow(5.0, 10.0, 0.3)]
    normalized = CoarseToFinePipeline._normalize_windows("video", windows)
    assert [window.candidate_id for window in normalized] == ["video:r0", "video:r1"]


def test_contracts_reject_non_finite_temporal_values():
    with pytest.raises(ValueError, match="finite"):
        CandidateWindow(0.0, 5.0, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        GroundingPrediction("x", 0.0, 1.0, float("inf"))


def test_grounder_failure_is_explicit_level2_degradation(tmp_path):
    class FailingGrounder:
        model_version = "failing"

        def predict(self, video_path, query, candidate_windows):
            raise TimeoutError("synthetic timeout")

    cache = FeatureCache(tmp_path)
    retriever = CachedCosineRetriever(cache)
    key = cache_identity("v", "m", {}, {})
    retriever.index("v.mp4", "v", key=key, video_duration=20, extractor=lambda: (np.ones((2, 32), dtype=np.float32), [5, 15]))
    result = CoarseToFinePipeline(retriever, FailingGrounder(), top_n=2, top_k=2).search("v.mp4", "v", "event")
    assert result["degraded"] is True
    assert result["degrade_level"] == 2
    assert all(row["degraded"] is True and row["raw_start"] == row["coarse_start"] for row in result["predictions"])


def test_boundary_failure_is_explicit_level1_degradation(tmp_path):
    class FailingBoundary:
        version = "failing-boundary"

        def refine(self, prediction, *, duration=None):
            raise RuntimeError("synthetic boundary failure")

    cache = FeatureCache(tmp_path)
    retriever = CachedCosineRetriever(cache)
    key = cache_identity("v", "m", {}, {})
    retriever.index("v.mp4", "v", key=key, video_duration=20, extractor=lambda: (np.ones((1, 32), dtype=np.float32), [5]))
    result = CoarseToFinePipeline(retriever, StubTimeLensGrounder(), boundary_refiner=FailingBoundary(), top_n=1, top_k=1).search("v.mp4", "v", "event")
    assert result["degraded"] is True
    assert result["degrade_level"] == 1
    assert result["predictions"][0]["degraded"] is True
    assert result["predictions"][0]["degrade_level"] == 1
    assert result["predictions"][0]["raw_start"] == result["predictions"][0]["refined_start"]
