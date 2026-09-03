#!/usr/bin/env python
"""Isolated CUDA allocator OOM/recovery smoke.

This intentionally runs outside the service and model process. It requests a
bounded allocation larger than currently free memory, catches the expected
allocator error, clears the cache, and verifies that a small allocation still
works. It is a runtime-safety probe, not a VideoITG capacity benchmark.
"""

from __future__ import annotations

import json


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"status": "skipped", "reason": "cuda_unavailable"}))
        return 2
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    request_bytes = int(free_bytes + 512 * 1024 * 1024)
    request_elements = request_bytes // torch.tensor([], dtype=torch.float32).element_size()
    caught = False
    error_name = None
    try:
        # The child process is expected to fail this allocation before any
        # model or service state is touched.
        _ = torch.empty((request_elements,), dtype=torch.float32, device=device)
    except RuntimeError as exc:
        caught = "out of memory" in str(exc).lower()
        error_name = type(exc).__name__
        if not caught:
            raise
    finally:
        torch.cuda.empty_cache()
    probe = torch.empty((1024,), dtype=torch.float32, device=device)
    probe.fill_(1.0)
    probe_sum = float(probe.sum().item())
    print(json.dumps({
        "status": "passed" if caught and probe_sum == 1024.0 else "failed",
        "device": str(device),
        "free_before_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "requested_bytes": int(request_bytes),
        "caught_oom": caught,
        "error_type": error_name,
        "post_recovery_probe_sum": probe_sum,
    }, ensure_ascii=False))
    return 0 if caught and probe_sum == 1024.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
