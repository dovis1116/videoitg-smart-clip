#!/usr/bin/env python
"""Replay a fixed-rate steady request stream against the local service.

This intentionally uses the synchronous endpoint and records each HTTP status,
wall time, timeout, and returned model runtime. It is a pressure artifact, not
an availability or production benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def post_one(base_url: str, video_path: str, query: str, request_id: str, timeout_s: float, scheduled_at: float) -> dict:
    wait_s = scheduled_at - time.monotonic()
    if wait_s > 0:
        time.sleep(wait_s)
    payload = json.dumps({"query": query, "video_path": video_path, "request_id": request_id}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/steady/rerank",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    status = None
    body = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            body = response.read().decode()
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode()
    except Exception as exc:  # pragma: no cover - exercised by a live server
        body = f"{type(exc).__name__}: {exc}"
    finished = time.monotonic()
    result = {
        "request_id": request_id,
        "scheduled_offset_s": round(scheduled_at, 6),
        "status_code": status,
        "wall_seconds": round(finished - started, 6),
    }
    try:
        decoded = json.loads(body)
        result["response"] = decoded
        detail = decoded.get("detail") if isinstance(decoded, dict) else None
        if isinstance(detail, dict):
            result["task_status"] = detail.get("status")
        elif isinstance(decoded, dict):
            result["task_status"] = decoded.get("status")
    except json.JSONDecodeError:
        result["body"] = body[:500]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8767")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--query", default="find the segment with a person")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests <= 0 or args.interval_s <= 0 or args.timeout_s <= 0:
        raise SystemExit("requests, interval, and timeout must be positive")
    video = args.video.expanduser().resolve(strict=True)
    start = time.monotonic() + 1.0
    futures = []
    with ThreadPoolExecutor(max_workers=min(args.requests, 16)) as pool:
        for index in range(args.requests):
            futures.append(
                pool.submit(
                    post_one,
                    args.base_url,
                    str(video),
                    args.query,
                    f"steady-pressure-{index:03d}",
                    args.timeout_s,
                    start + index * args.interval_s,
                )
            )
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda item: item["request_id"])
    successes = [item for item in results if item["status_code"] == 200 and item.get("task_status") == "succeeded"]
    timeouts = [item for item in results if item["status_code"] == 504]
    durations = sorted(item["wall_seconds"] for item in results if item["status_code"] is not None)
    p95 = durations[max(0, int(0.95 * len(durations)) - 1)] if durations else None
    summary = {
        "contract": "G5-A steady-1s",
        "requests": args.requests,
        "interval_s": args.interval_s,
        "successful": len(successes),
        "timeouts": len(timeouts),
        "status_counts": {str(code): sum(item["status_code"] == code for item in results) for code in sorted({item["status_code"] for item in results}, key=str)},
        "wall_p95_s": p95,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("contract", "requests", "interval_s", "successful", "timeouts", "status_counts", "wall_p95_s")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
