#!/usr/bin/env python3
"""Audit an existing versioned feature cache without re-encoding videos.

The report is intentionally limited to facts present in cache metadata/NPZ
artifacts.  In particular, extraction latency is reported as ``null`` unless
it was persisted by the producer; it is never reconstructed from timestamps.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from statistics import mean, median

import numpy as np


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": float(mean(values)) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = []
    if args.manifest is not None:
        manifest_rows = [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    manifest_by_id = {str(row.get("sample_id")): row for row in manifest_rows}

    metadata_paths = sorted(args.cache_root.glob("*.json"))
    rows: list[dict] = []
    errors: list[dict] = []
    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_path = Path(metadata["feature_path"])
            if not feature_path.is_file():
                raise FileNotFoundError(str(feature_path))
            with np.load(feature_path) as arrays:
                features = np.asarray(arrays["features"])
                timestamps = np.asarray(arrays["timestamps"])
            if features.ndim != 2 or timestamps.ndim != 1 or len(features) != len(timestamps):
                raise ValueError("features/timestamps shape mismatch")
            sampling = json.loads(metadata["sampling_config"])
            row = {
                "video_id": str(metadata["video_id"]),
                "feature_model": metadata.get("feature_model"),
                "feature_model_version": metadata.get("feature_model_version"),
                "sampling_config": sampling,
                "video_duration_s": float(metadata["video_duration"]),
                "sampled_frame_count": int(len(features)),
                "feature_dimension": int(features.shape[1]),
                "feature_dtype": str(features.dtype),
                "timestamp_first_s": float(timestamps[0]) if len(timestamps) else None,
                "timestamp_last_s": float(timestamps[-1]) if len(timestamps) else None,
                "cache_bytes_npz": int(feature_path.stat().st_size),
                "metadata_bytes_json": int(metadata_path.stat().st_size),
                "cache_hit_on_write": bool(metadata.get("cache_hit", False)),
                "feature_extraction_latency_ms": (
                    float(metadata["extraction_latency_ms"])
                    if metadata.get("extraction_latency_ms") is not None
                    else None
                ),
                "created_at": metadata.get("created_at"),
            }
            manifest_row = manifest_by_id.get(row["video_id"])
            if manifest_row is not None:
                row["sample_id"] = manifest_row.get("sample_id")
                row["video_path"] = manifest_row.get("video_path")
            rows.append(row)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile):
            errors.append({"metadata_path": str(metadata_path), "error": "invalid_metadata_or_npz"})

    feature_bytes = [float(row["cache_bytes_npz"]) for row in rows]
    metadata_bytes = [float(row["metadata_bytes_json"]) for row in rows]
    durations = [float(row["video_duration_s"]) for row in rows]
    sampled_frames = [float(row["sampled_frame_count"]) for row in rows]
    extraction_latencies = [
        float(row["feature_extraction_latency_ms"])
        for row in rows
        if row["feature_extraction_latency_ms"] is not None
    ]
    dimensions = sorted({int(row["feature_dimension"]) for row in rows})
    sampling_configs = sorted({json.dumps(row["sampling_config"], sort_keys=True) for row in rows})
    video_ids = {row["video_id"] for row in rows}
    missing_manifest_ids = sorted(set(manifest_by_id) - video_ids)
    extra_cache_ids = sorted(video_ids - set(manifest_by_id)) if manifest_by_id else []

    report = {
        "run_id": f"g2_feature_cache_audit_{time.strftime('%Y%m%d_%H%M%S')}",
        "cache_root": str(args.cache_root),
        "manifest": str(args.manifest) if args.manifest else None,
        "scope": "existing_cache_artifacts_without_reencoding",
        "sample_count": len(rows),
        "metadata_file_count": len(metadata_paths),
        "valid_cache_count": len(rows),
        "invalid_cache_count": len(errors),
        "manifest_row_count": len(manifest_rows),
        "missing_cache_for_manifest_count": len(missing_manifest_ids),
        "extra_cache_not_in_manifest_count": len(extra_cache_ids),
        "feature_model_versions": sorted({str(row["feature_model_version"]) for row in rows}),
        "sampling_configs": [json.loads(value) for value in sampling_configs],
        "feature_dimensions": dimensions,
        "feature_dtypes": sorted({str(row["feature_dtype"]) for row in rows}),
        "video_duration_s": _summary(durations),
        "sampled_frame_count": _summary(sampled_frames),
        "cache_bytes_npz": _summary(feature_bytes),
        "metadata_bytes_json": _summary(metadata_bytes),
        "total_cache_bytes_npz": int(sum(feature_bytes)),
        "total_metadata_bytes_json": int(sum(metadata_bytes)),
        "cache_hit_on_write_count": sum(bool(row["cache_hit_on_write"]) for row in rows),
        "feature_extraction_latency_ms": _summary(extraction_latencies),
        "measurement_limits": [
            (
                "Feature extraction latency is available only for cache entries produced by a producer version that persists it; "
                "legacy entries remain null and no latency is inferred from created_at or file timestamps."
                if extraction_latencies
                else "The cache producer did not persist feature extraction latency; no latency is inferred from created_at or file timestamps."
            ),
            f"This audit describes {len(rows)} existing manifest rows and is not a full negative/difficult validation set.",
        ],
        "missing_cache_video_ids": missing_manifest_ids,
        "extra_cache_video_ids": extra_cache_ids,
        "errors": errors,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "run_id", "sample_count", "valid_cache_count", "missing_cache_for_manifest_count",
        "total_cache_bytes_npz", "video_duration_s", "sampled_frame_count", "cache_bytes_npz",
        "feature_extraction_latency_ms", "measurement_limits",
    )}, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0 if not errors and not missing_manifest_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
