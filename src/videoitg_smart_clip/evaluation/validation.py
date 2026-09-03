"""Validation-manifest contract for grounding and no-match evaluation."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Iterable, Mapping

NEGATIVE_TYPES = frozenset({"none", "event_absent", "wrong_action", "wrong_object", "theme_unrelated"})
REQUIRED_NEGATIVE_TYPES = frozenset({"event_absent", "wrong_action", "wrong_object", "theme_unrelated"})
SPLITS = frozenset({"train", "dev", "test"})


class ValidationManifestError(ValueError):
    """Raised when a validation row violates the evaluation contract."""


def validate_manifest_rows(
    rows: Iterable[Mapping],
    *,
    require_complete: bool = False,
    require_negative_categories: bool = False,
    require_files: bool = False,
) -> dict:
    """Validate rows and return an auditable summary.

    ``actual_match`` is a match-presence label: True means the event exists.
    ``negative_type`` identifies the four required no-match families.
    """
    materialized = list(rows)
    if not materialized:
        raise ValidationManifestError("manifest must contain at least one row")
    seen: set[str] = set()
    split_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    pending = 0
    required = ("sample_id", "video_id", "query", "actual_match", "negative_type", "ground_truth", "split", "label_status")
    for index, row in enumerate(materialized, 1):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValidationManifestError(f"row {index} missing required fields: {','.join(missing)}")
        sample_id = str(row["sample_id"]).strip()
        video_id = str(row["video_id"]).strip()
        query = str(row["query"]).strip()
        if not sample_id or not video_id or not query:
            raise ValidationManifestError(f"row {index} has blank sample_id, video_id or query")
        video_path = str(row.get("video_path", "")).strip()
        if require_files:
            if not video_path:
                raise ValidationManifestError(f"row {index} missing video_path")
            if not Path(video_path).expanduser().is_file():
                raise ValidationManifestError(f"row {index} video_path does not exist: {video_path}")
        if sample_id in seen:
            raise ValidationManifestError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if not isinstance(row["actual_match"], bool):
            raise ValidationManifestError(f"row {index} actual_match must be boolean")
        negative_type = str(row["negative_type"])
        if negative_type not in NEGATIVE_TYPES:
            raise ValidationManifestError(f"row {index} unsupported negative_type: {negative_type}")
        if row["actual_match"] and negative_type != "none":
            raise ValidationManifestError(f"row {index} matched sample must use negative_type=none")
        if not row["actual_match"] and negative_type == "none":
            raise ValidationManifestError(f"row {index} no-match sample needs a negative_type")
        ground_truth = row["ground_truth"]
        if not isinstance(ground_truth, list):
            raise ValidationManifestError(f"row {index} ground_truth must be a list")
        if row["actual_match"] and not ground_truth:
            raise ValidationManifestError(f"row {index} matched sample needs ground_truth")
        for segment in ground_truth:
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                raise ValidationManifestError(f"row {index} ground_truth segments must be [start, end]")
            start, end = (float(value) for value in segment)
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValidationManifestError(f"row {index} ground_truth has invalid bounds")
        if not row["actual_match"] and ground_truth:
            raise ValidationManifestError(f"row {index} no-match sample must have empty ground_truth")
        split = str(row["split"])
        if split not in SPLITS:
            raise ValidationManifestError(f"row {index} unsupported split: {split}")
        label_status = str(row["label_status"])
        if label_status not in {"complete", "pending", "synthetic"}:
            raise ValidationManifestError(f"row {index} label_status must be complete, pending or synthetic")
        pending += label_status == "pending"
        split_counts[split] += 1
        negative_counts[negative_type] += 1
        match_counts["present" if row["actual_match"] else "absent"] += 1
    if require_complete and pending:
        raise ValidationManifestError(f"manifest contains {pending} pending rows")
    if require_negative_categories:
        missing_categories = sorted(REQUIRED_NEGATIVE_TYPES - set(negative_counts))
        if missing_categories:
            raise ValidationManifestError("missing required negative categories: " + ",".join(missing_categories))
        if not match_counts["present"] or not match_counts["absent"]:
            raise ValidationManifestError("manifest must contain both present and absent samples")
    return {
        "row_count": len(materialized),
        "split_counts": dict(split_counts),
        "match_counts": dict(match_counts),
        "negative_type_counts": dict(negative_counts),
        "pending_rows": pending,
        "complete": pending == 0,
        "synthetic_rows": sum(str(row["label_status"]) == "synthetic" for row in materialized),
        "negative_coverage_complete": REQUIRED_NEGATIVE_TYPES.issubset(negative_counts),
    }
