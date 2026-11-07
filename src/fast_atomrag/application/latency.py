"""Rolling per-mode latency tracker for router self-calibration (ADR-009)."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping

from ..domain.router import Retrieval


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile on an already sorted, non-empty list."""
    idx = max(0, min(len(sorted_values) - 1, round(q / 100.0 * (len(sorted_values) - 1))))
    return sorted_values[idx]


class LatencyTracker:
    def __init__(
        self,
        defaults_s: Mapping[Retrieval, float],
        window: int = 256,
        min_samples: int = 10,
    ) -> None:
        if window <= 0 or not 1 <= min_samples <= window:
            raise ValueError("require window > 0 and 1 <= min_samples <= window")
        self._defaults = dict(defaults_s)
        self._min_samples = min_samples
        self._samples: dict[Retrieval, deque[float]] = {m: deque(maxlen=window) for m in Retrieval}
        self._sla_fallbacks = 0
        self._lock = threading.Lock()

    def record(self, mode: Retrieval, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("latency must be >= 0")
        with self._lock:
            self._samples[mode].append(seconds)

    def record_sla_fallback(self) -> None:
        with self._lock:
            self._sla_fallbacks += 1

    def effective_s(self) -> dict[Retrieval, float]:
        """Rolling p95 per mode once calibrated, conservative default otherwise."""
        with self._lock:
            out: dict[Retrieval, float] = {}
            for mode in Retrieval:
                values = self._samples[mode]
                if len(values) >= self._min_samples:
                    out[mode] = _percentile(sorted(values), 95)
                else:
                    out[mode] = self._defaults.get(mode, 0.0)
            return out

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            modes: dict[str, dict[str, float | int | bool]] = {}
            for mode in Retrieval:
                values = sorted(self._samples[mode])
                modes[mode.value] = {
                    "sample_count": len(values),
                    "calibrated": len(values) >= self._min_samples,
                    "p50_ms": round(_percentile(values, 50) * 1000, 3) if values else 0.0,
                    "p95_ms": round(_percentile(values, 95) * 1000, 3) if values else 0.0,
                }
            return {"modes": modes, "sla_fallbacks": self._sla_fallbacks}
