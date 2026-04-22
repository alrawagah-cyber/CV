"""Prediction drift monitoring.

Tracks rolling distributions of damage types, severity grades, detected
parts, and overall assessments over the last *N* predictions so operators
can spot model degradation before it impacts claim quality.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Any

_DEFAULT_WINDOW = 1000


class DriftMonitor:
    """Thread-safe rolling-window distribution tracker.

    Parameters
    ----------
    window : int
        Maximum number of prediction records to keep.
    """

    def __init__(self, window: int = _DEFAULT_WINDOW) -> None:
        self._window = window
        self._lock = threading.Lock()
        # Each entry is a list of per-part dicts extracted from a report.
        self._records: deque[dict[str, Any]] = deque(maxlen=window)

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record(self, report: dict) -> None:
        """Ingest a ClaimReport dict after a successful /assess call."""
        entry: dict[str, Any] = {
            "overall_assessment": report.get("overall_assessment", "unknown"),
            "parts": [],
        }
        for part in report.get("parts", []):
            entry["parts"].append(
                {
                    "part": part.get("part", "unknown"),
                    "primary_damage_type": part.get("primary_damage_type"),
                    "severity_grade": (part.get("severity") or {}).get("grade"),
                }
            )
        with self._lock:
            self._records.append(entry)

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def get_stats(self) -> dict[str, Any]:
        """Return current rolling distributions and basic statistics."""
        with self._lock:
            records = list(self._records)

        damage_counter: Counter[str] = Counter()
        severity_counter: Counter[str] = Counter()
        parts_counter: Counter[str] = Counter()
        overall_counter: Counter[str] = Counter()
        total_parts = 0

        for rec in records:
            overall_counter[rec["overall_assessment"]] += 1
            for p in rec["parts"]:
                total_parts += 1
                parts_counter[p["part"]] += 1
                if p["primary_damage_type"]:
                    damage_counter[p["primary_damage_type"]] += 1
                if p["severity_grade"]:
                    severity_counter[p["severity_grade"]] += 1

        n = len(records)
        return {
            "window_size": self._window,
            "total_reports": n,
            "total_parts": total_parts,
            "avg_parts_per_report": round(total_parts / n, 2) if n else 0.0,
            "overall_assessment_distribution": dict(overall_counter),
            "damage_type_distribution": dict(damage_counter),
            "severity_distribution": dict(severity_counter),
            "parts_distribution": dict(parts_counter),
        }
