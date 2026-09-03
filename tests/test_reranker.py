from videoitg_smart_clip.reranker import CandidateSegment
from videoitg_smart_clip.reranker.videoitg_adapter import VideoITGReranker, interval_frame_indices


def test_interval_indices_are_clamped_and_bounded():
    values = interval_frame_indices(100, 10.0, 0.0, 20.0, target_fps=2.0, max_frames=8)
    assert len(values) == 8
    assert values == sorted(set(values))
    assert values[0] >= 0 and values[-1] < 100


def test_candidate_rejects_invalid_interval():
    try:
        CandidateSegment("video.mp4", 3.0, 3.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid candidate interval was accepted")


def test_full_video_candidate_requires_zero_start():
    candidate = CandidateSegment("video.mp4", 0.0, None)
    assert candidate.end_s is None
    try:
        CandidateSegment("video.mp4", 1.0, None)
    except ValueError:
        pass
    else:
        raise AssertionError("non-zero full-video candidate was accepted")


def test_cuda_cache_policy_is_explicit_and_backward_compatible():
    assert VideoITGReranker("dummy").empty_cache_each_candidate is True
    assert VideoITGReranker("dummy", empty_cache_each_candidate=False).empty_cache_each_candidate is False


def test_rank_batch_rejects_empty_queries_before_model_load():
    reranker = VideoITGReranker("dummy")
    candidate = CandidateSegment("video.mp4", 0.0, None)
    try:
        reranker.rank_batch(["", ""], [candidate, candidate])
    except ValueError:
        pass
    else:
        raise AssertionError("rank_batch accepted an empty query")
