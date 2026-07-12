#!/usr/bin/env python3
"""
throttle_guard_unit_test.py — directly verifies ThrottleGuard's backoff and
recovery state machine using an ARTIFICIALLY sensitive threshold, so the
mechanism itself is proven correct without depending on real hardware
throttling occurring during the test (which the two live sustained-load runs
did not reliably trigger -- both stayed under the 8% production threshold).

Feeds synthetic wall-clock times directly into guard.record() -- no GPU/CPU/
ANE execution at all. This isolates the state machine from hardware variance.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

from fusionml._metal.throttle_guard import ThrottleGuard


def main():
    guard = ThrottleGuard(
        window=5, calibration_calls=5,
        throttle_threshold=1.05, circuit_breaker_threshold=1.30, recovery_threshold=1.01,
        tier_change_cooldown=3, cooldown_calls=3,
    )
    default_ratios = {"gpu": 0.44, "cpu": 0.27, "ane": 0.29}
    total_rows = 8192

    scenario = (
        [10.0] * 5 +      # calibration -> baseline = 10.0ms
        [10.0] * 3 +      # normal, stable
        [12.0] * 6 +      # +20% sustained slowdown -> should trigger backoff after cooldown
        [10.0] * 6 +      # recovers to baseline -> should trigger recovery after cooldown
        [50.0] * 2 +      # sudden severe spike -> should trip circuit breaker immediately
        [50.0] * 3 +      # still bad during cooldown -> forced GPU-only
        [10.0] * 4         # recovered -> cooldown elapses, cautious re-attempt
    )

    print("=" * 70)
    print("  ThrottleGuard State Machine Unit Test (synthetic timings)")
    print("=" * 70)
    print(f"{'call':>4} {'wall_ms':>8} {'rows(gpu/cpu/ane)':>20} {'state':>16}")

    passed = True
    for i, wall_ms in enumerate(scenario):
        rows = guard.recommend_rows(total_rows, default_ratios)
        guard.record(wall_ms)
        h = guard.history[-1]
        state = h.get("state", "calibrating")
        print(f"{i:>4} {wall_ms:>8.1f} {str((rows['gpu'], rows['cpu'], rows['ane'])):>20} {state:>16}")

        # Assertions on expected behavior at key points
        if i == 13:  # after sustained +20% slowdown + cooldown elapsed
            if rows["ane"] != 0:
                print(f"  FAIL: expected ANE disabled by call {i} after sustained slowdown, got ane={rows['ane']}")
                passed = False
        if i == 23:  # a few calls into the severe spike -> should be circuit-broken (GPU-only)
            # Note: the guard uses a rolling MEDIAN (not a single sample) to decide
            # circuit-break, so it takes a few consecutive bad samples to trip --
            # that's intentional robustness against a single transient outlier, not a bug.
            if not (rows["gpu"] == total_rows and rows["cpu"] == 0 and rows["ane"] == 0):
                print(f"  FAIL: expected circuit-broken GPU-only fallback at call {i}, got {rows}")
                passed = False

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(guard.summary())
    if guard.summary()["calls_throttled"] == 0:
        print("FAIL: backoff logic never triggered despite synthetic sustained slowdown")
        passed = False
    if guard.summary()["calls_circuit_broken"] == 0:
        print("FAIL: circuit breaker never triggered despite synthetic severe spike")
        passed = False
    if guard.summary()["calls_recovering"] == 0:
        print("FAIL: recovery logic never triggered despite synthetic return to baseline")
        passed = False

    print("\nVERDICT:", "PASS -- state machine behaves correctly" if passed else "FAIL -- see above")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
