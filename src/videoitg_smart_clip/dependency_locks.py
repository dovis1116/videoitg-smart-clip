"""Locks for third-party lazy import paths used by concurrent workers."""

from __future__ import annotations

import threading


# Transformers exposes several objects through a lazy module. Concurrent
# first-time attribute resolution from multiple worker threads can otherwise
# yield a transient ImportError even though the installation is valid.
TRANSFORMERS_IMPORT_LOCK = threading.Lock()
