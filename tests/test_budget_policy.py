import pytest

from videoitg_smart_clip.budgeting.policy import BudgetPolicy


def test_unconfigured_policy_is_fixed_by_default():
    decision = BudgetPolicy().choose(queue_length=0, uncertainty=0.9)
    assert (decision.mode, decision.max_frames) == ("fixed", 16)


def test_explicit_policy_routes_full_and_bypass():
    policy = BudgetPolicy(
        full_uncertainty_threshold=0.7,
        reduced_queue_threshold=1,
        bypass_queue_threshold=4,
    )
    assert policy.choose(queue_length=0, uncertainty=0.8).mode == "full"
    assert policy.choose(queue_length=4, uncertainty=0.8).mode == "retrieval_only"
    assert policy.choose(queue_length=2, uncertainty=0.2).mode == "fixed"


def test_policy_rejects_invalid_signals():
    with pytest.raises(ValueError):
        BudgetPolicy().choose(queue_length=-1)
    with pytest.raises(ValueError):
        BudgetPolicy().choose(queue_length=0, uncertainty=1.1)
