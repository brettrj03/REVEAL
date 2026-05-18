"""Utility helpers for optional TRACE_AUDIT logging."""

from __future__ import annotations

import json
import os
from typing import Any

TRACE_FLAG = os.getenv("TRACE_AUDIT", "").strip().lower() in {"1", "true", "yes"}


def _safe_json(data: dict[str, Any]) -> str:
    """Serialize data to JSON, falling back to string representations."""

    try:
        return json.dumps(data, default=str)
    except TypeError:
        safe = {k: str(v) for k, v in data.items()}
        return json.dumps(safe)


def trace_event(event: str, **fields: Any) -> None:
    """Print trace messages when TRACE_AUDIT=1."""

    if not TRACE_FLAG:
        return

    payload = _safe_json(fields)
    print(f"[TRACE_AUDIT] {event}: {payload}")

