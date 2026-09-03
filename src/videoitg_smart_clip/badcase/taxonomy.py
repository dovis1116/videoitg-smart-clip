"""Canonical Badcase taxonomy shared by offline reports and feedback triage."""

from __future__ import annotations

from enum import StrEnum


class BadcaseType(StrEnum):
    retrieval_miss = "retrieval_miss"
    wrong_event = "wrong_event"
    boundary_shift = "boundary_shift"
    short_event = "short_event"
    long_event = "long_event"
    duplicate = "duplicate"
    false_positive = "false_positive"
    no_match_error = "no_match_error"
    ranking_error = "ranking_error"
    latency_timeout = "latency_timeout"
    video_decode_error = "video_decode_error"


BADCASE_TYPES = frozenset(item.value for item in BadcaseType)


def normalize_badcase_type(value: str | None) -> str:
    """Return a canonical label without silently inventing a semantic class."""

    if value in BADCASE_TYPES:
        return str(value)
    return "unclassified"
