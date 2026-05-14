from __future__ import annotations

import os
from typing import Any


_HIT_LABELS: set[str] = set()


def enable_debug_breakpoints(
    *,
    repeat: bool = False,
    labels: str | None = None,
) -> None:
    os.environ["AIN_DEBUG_BREAKPOINTS"] = "1"
    os.environ["AIN_DEBUG_BREAKPOINTS_REPEAT"] = "1" if repeat else "0"
    if labels:
        os.environ["AIN_DEBUG_BREAKPOINT_LABELS"] = labels


def _enabled() -> bool:
    return os.environ.get("AIN_DEBUG_BREAKPOINTS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _label_allowed(label: str) -> bool:
    raw = os.environ.get("AIN_DEBUG_BREAKPOINT_LABELS", "").strip()
    if not raw:
        return True
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    return label in allowed


def _compact(value: Any) -> str:
    text = repr(value)
    if len(text) > 120:
        return text[:117] + "..."
    return text


def debug_breakpoint(label: str, **context: Any) -> None:
    if not _enabled() or not _label_allowed(label):
        return

    repeat = os.environ.get("AIN_DEBUG_BREAKPOINTS_REPEAT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not repeat and label in _HIT_LABELS:
        return

    _HIT_LABELS.add(label)
    print(f"\n[DEBUG BREAKPOINT] {label}")
    for key, value in context.items():
        print(f"  {key}: {_compact(value)}")
    breakpoint()
