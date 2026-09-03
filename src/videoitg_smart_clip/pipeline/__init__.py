"""Composable query-aware coarse-to-fine temporal grounding pipeline."""

from .contracts import CandidateWindow, GroundingPrediction, TemporalGrounder, TemporalRetriever
from .postprocess import (
    BoundaryRefiner,
    CandidateRanker,
    NoMatchDecider,
    TemporalDeduplicator,
)
from .service import CoarseToFinePipeline

__all__ = [
    "CandidateWindow",
    "GroundingPrediction",
    "TemporalGrounder",
    "TemporalRetriever",
    "BoundaryRefiner",
    "CandidateRanker",
    "NoMatchDecider",
    "TemporalDeduplicator",
    "CoarseToFinePipeline",
]
