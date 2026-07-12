"""
ThrottleGuard — model-agnostic, runtime thermal-throttle protection for
tri-compute (CPU+GPU+ANE) scheduling.

Motivation (measured on Apple M1 8GB, see benchmarks/python/tricompute_sustained_test.py):
running CPU+GPU+ANE concurrently draws enough combined power to trigger real
thermal throttling on fanless Apple Silicon that GPU-only execution never
triggers under the same sustained load. A burst measurement showed 1.79x
speedup; a 90-second sustained measurement showed the achievable speedup
decays to ~1.25-1.29x as the system heats up (tri-compute: +10.9% latency
over 90s; GPU-only: -0.7%, essentially flat).

ThrottleGuard is a closed-loop controller with no dependency on privileged
thermal APIs (no sudo powermetrics) -- it infers throttling purely from
observed wall-clock latency drift versus a cold-calibrated baseline, which is
portable across any Apple Silicon generation and doesn't require special
permissions. This makes it safe to ship in a production path.

Usage (model-agnostic -- works with ANY (M, K, N) matmul-shaped workload):

    guard = ThrottleGuard()  # auto-calibrates cold baseline from first N calls
    default_ratios = {"gpu": 0.44, "cpu": 0.27, "ane": 0.29}  # from adaptive search

    # Pre-warm every ANE tier's CoreML model ONCE before the timed loop --
    # required, since ANE's cache is keyed by exact row count.
    full_ane_rows = total_rows - int(total_rows*default_ratios["gpu"]) - int(total_rows*default_ratios["cpu"])
    for rows in guard.ane_row_tiers(full_ane_rows):
        if rows > 0:
            ane_matmul(a[:rows], b, compute_units="CPU_AND_NE")

    for _ in range(many_calls):
        rows = guard.recommend_rows(total_rows, default_ratios)  # {"gpu":.., "cpu":.., "ane":..}
        result, wall_ms = execute_split(a, b, rows)   # caller-provided execution
        guard.record(wall_ms)
"""

import time
import statistics
from collections import deque
from typing import Dict, Optional


class ThrottleGuard:
    """
    Tracks recent iteration wall-clock times and dynamically backs off ANE
    (and, if needed, CPU) participation when sustained-load throttling is
    detected, with a hard circuit-breaker to GPU-only if degradation ever
    threatens to erase the tri-compute advantage entirely.

    Fully reusable / model-agnostic: operates purely on the caller-reported
    wall-clock time per call, with no knowledge of what shape or model
    produced it. Any code path (Llama, GPT-2, a user's own model, a
    non-transformer numeric kernel) can share one instance.
    """

    # ANE participation is BINARY (full share, or fully disabled) rather than
    # a graduated set of tiers. Two hard platform constraints force this:
    #   1. Each distinct ANE row-count is a distinct CoreML model (compile
    #      cost ~1.5-1.7s, see benchmarks/python/ane_spike_test.py) -- varying
    #      continuously would force a fresh compile almost every call.
    #   2. More fundamentally: loading a SECOND distinct CoreML model in a
    #      process that has also touched MLX (GPU) segfaults outright (exit
    #      139/-11), regardless of threading or compile-vs-cache-load. Isolated
    #      via benchmarks/python/ane_multi_tier_repro.py -- the crash occurs on
    #      the second distinct model, deterministically, even fully
    #      single-threaded. A process may safely use AT MOST ONE ANE shape
    #      for its entire lifetime once MLX is in play. This rules out any
    #      graduated backoff within a single process; "less ANE" can only mean
    #      "no ANE" here, not "smaller ANE".
    ANE_TIERS = (1.0, 0.0)

    def __init__(
        self,
        window: int = 20,
        calibration_calls: int = 10,
        throttle_threshold: float = 1.08,   # 8% slower than cold baseline -> start backing off
        circuit_breaker_threshold: float = 1.35,  # 35% slower -> fall back to GPU-only entirely
        recovery_threshold: float = 1.02,   # within 2% of baseline -> safe to restore full split
        min_ane_share: float = 0.0,
        cooldown_calls: int = 5,            # calls to wait after a circuit-break before re-attempting tri-compute
        tier_change_cooldown: int = 8,      # min calls between tier changes -- prevents flapping/recompile thrash
    ):
        self.window = window
        self.calibration_calls = calibration_calls
        self.throttle_threshold = throttle_threshold
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.recovery_threshold = recovery_threshold
        self.min_ane_share = min_ane_share
        self.cooldown_calls = cooldown_calls
        self.tier_change_cooldown = tier_change_cooldown

        self._recent: deque = deque(maxlen=window)
        self._calibration_samples: list = []
        self._cold_baseline_ms: Optional[float] = None
        self._ane_tier_idx: int = 0  # index into ANE_TIERS; 0 = full share, len-1 = disabled
        self._circuit_broken: bool = False
        self._cooldown_remaining: int = 0
        self._call_count: int = 0
        self._calls_since_tier_change: int = 0

        # Diagnostics, exposed for logging/paper reproducibility
        self.history: list = []

    @property
    def _ane_backoff_factor(self) -> float:
        return self.ANE_TIERS[self._ane_tier_idx]

    def ane_row_tiers(self, full_ane_rows: int) -> list:
        """
        Pre-compute the exact ANE row counts this guard could ever request for
        a given full-share row count, so the caller can pre-warm (compile)
        every tier's CoreML model ONCE before entering a timed/production loop.
        Returns counts in the same order as ANE_TIERS, deduplicated.
        """
        seen = []
        for factor in self.ANE_TIERS:
            rows = max(self.min_ane_share_rows(full_ane_rows), int(full_ane_rows * factor))
            if rows not in seen:
                seen.append(rows)
        return seen

    def min_ane_share_rows(self, full_ane_rows: int) -> int:
        return int(full_ane_rows * self.min_ane_share)

    @property
    def is_calibrating(self) -> bool:
        return self._cold_baseline_ms is None

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken

    def recommend_rows(self, total_rows: int, default_ratios: Dict[str, float]) -> Dict[str, int]:
        """
        Integer-exact split recommendation. Use this instead of recommend_ratios()
        for any caller that talks to ANE, since ANE's model cache is keyed by
        exact row count -- floating-point ratio math can jitter by ±1 row across
        calls even at a "stable" tier, silently forcing a recompile. This method
        guarantees the SAME ane_rows integer every time a given tier is active,
        by deriving it directly from ane_row_tiers() rather than re-deriving it
        from ratios each call.

        Pre-warm every value ane_row_tiers(full_ane_rows) can return (via
        ane_matmul) before entering a timed loop -- see
        benchmarks/python/tricompute_throttle_guard_test.py for the pattern.
        """
        full_gpu_rows = int(total_rows * default_ratios.get("gpu", 0.0))
        full_ane_rows = int(total_rows * default_ratios.get("ane", 0.0))
        # CPU absorbs the remainder (not a naive ratio*total) so the three
        # counts always sum to EXACTLY total_rows, with no rows silently
        # dropped -- e.g. when a caller has zeroed ratios["ane"] because ANE
        # isn't prewarmed for this shape, that freed capacity must land
        # somewhere, not vanish.
        full_cpu_rows = total_rows - full_gpu_rows - full_ane_rows

        if self._circuit_broken:
            if self._cooldown_remaining > 0:
                return {"gpu": total_rows, "cpu": 0, "ane": 0}
            self._circuit_broken = False
            self._ane_tier_idx = len(self.ANE_TIERS) - 1
            self._calls_since_tier_change = 0

        if self.is_calibrating:
            return {"gpu": full_gpu_rows, "cpu": full_cpu_rows, "ane": full_ane_rows}

        ane_rows = max(self.min_ane_share_rows(full_ane_rows),
                        int(full_ane_rows * self._ane_backoff_factor))
        freed_rows = full_ane_rows - ane_rows
        denom = (full_gpu_rows + full_cpu_rows) or 1
        gpu_rows = full_gpu_rows + freed_rows * full_gpu_rows // denom
        cpu_rows = total_rows - gpu_rows - ane_rows  # remainder absorbs rounding, never touches ane_rows

        return {"gpu": gpu_rows, "cpu": cpu_rows, "ane": ane_rows}

    def record(self, wall_ms: float):
        """Report the observed wall-clock time for the most recent call."""
        self._call_count += 1
        self._calls_since_tier_change += 1

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if self.is_calibrating:
            self._calibration_samples.append(wall_ms)
            if len(self._calibration_samples) >= self.calibration_calls:
                self._cold_baseline_ms = statistics.median(self._calibration_samples)
            self.history.append({"call": self._call_count, "wall_ms": wall_ms, "state": "calibrating"})
            return

        self._recent.append(wall_ms)
        recent_median = statistics.median(self._recent)
        ratio = recent_median / self._cold_baseline_ms
        last_tier_idx = len(self.ANE_TIERS) - 1

        state = "normal"
        if ratio >= self.circuit_breaker_threshold:
            self._circuit_broken = True
            self._cooldown_remaining = self.cooldown_calls
            self._ane_tier_idx = last_tier_idx
            self._calls_since_tier_change = 0
            state = "circuit_broken"
        elif self._calls_since_tier_change >= self.tier_change_cooldown:
            if ratio >= self.throttle_threshold and self._ane_tier_idx < last_tier_idx:
                self._ane_tier_idx += 1  # step DOWN one tier (less ANE)
                self._calls_since_tier_change = 0
                state = "throttled_backoff"
            elif ratio <= self.recovery_threshold and self._ane_tier_idx > 0:
                self._ane_tier_idx -= 1  # step UP one tier (more ANE)
                self._calls_since_tier_change = 0
                state = "recovering"

        self.history.append({
            "call": self._call_count, "wall_ms": wall_ms, "recent_median_ms": recent_median,
            "throttle_ratio": ratio, "ane_tier": self.ANE_TIERS[self._ane_tier_idx], "state": state,
        })

    def summary(self) -> dict:
        states = [h["state"] for h in self.history]
        return {
            "cold_baseline_ms": self._cold_baseline_ms,
            "total_calls": self._call_count,
            "calls_throttled": states.count("throttled_backoff"),
            "calls_circuit_broken": states.count("circuit_broken"),
            "calls_recovering": states.count("recovering"),
            "final_ane_tier": self.ANE_TIERS[self._ane_tier_idx],
        }
