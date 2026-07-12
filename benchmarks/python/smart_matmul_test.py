#!/usr/bin/env python3
"""
smart_matmul_test.py — validates fusionml.smart_matmul().

Part 1 (the real, shippable MVP): default CPU+GPU path, safe by construction
-- correctness, naive-fallback safety net, no crash across multiple distinct
shapes. This is the part that must pass for the MVP to be considered done.

Part 2 (explicitly experimental): ANE opt-in via prewarm_ane_shape(). Reported
for transparency but NOT required to pass -- see smart_matmul.py's module
docstring for the real, unresolved CoreML+MLX instability this exercises.
Run as its own subprocess, isolated from Part 1, so a Part 2 crash can never
prevent Part 1 from being validated.
"""
import os
import sys
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))


SHAPES = [
    (8192, 1600, 6400),
    (4096, 512, 2048),
    (6000, 300, 900),
    (256, 128, 64),  # tiny -- below MIN_ROWS_TO_SPLIT, single-dispatch path
]


def check_shape(scheduler, M, K, N, seed):
    np.random.seed(seed)
    a = (np.random.randn(M, K) * 0.02).astype(np.float32)
    b = (np.random.randn(K, N) * 0.02).astype(np.float32)

    t0 = time.perf_counter()
    result = scheduler(a, b)
    smart_ms = (time.perf_counter() - t0) * 1000.0

    expected = a.astype(np.float64) @ b.astype(np.float64)
    rel_err = float(np.max(np.abs(result.astype(np.float64) - expected)) / (np.max(np.abs(expected)) + 1e-8))

    t0 = time.perf_counter()
    _ = a @ b
    naive_ms = (time.perf_counter() - t0) * 1000.0

    ok = rel_err < 0.05  # generous -- ANE's FP16 weight baking gives ~1.5% baseline error
    print(f"  [{M}x{K}] @ [{K}x{N}]: smart={smart_ms:.2f}ms naive={naive_ms:.2f}ms "
          f"speedup={naive_ms/smart_ms:.2f}x rel_err={rel_err:.4e} {'OK' if ok else 'FAIL'}")

    # Exercise the rest of the probation window (median-of-3 decision, see
    # smart_matmul.py) -- the fallback commitment only finalizes on the
    # PROBATION_CALLS'th call, so only a call AFTER that window closes is
    # guaranteed to reflect the final (possibly fallen-back) behavior.
    from fusionml.smart_matmul import TriComputeMatmul
    for _ in range(TriComputeMatmul.PROBATION_CALLS - 1):
        scheduler(a, b)

    t0 = time.perf_counter()
    _ = scheduler(a, b)
    post_probation_ms = (time.perf_counter() - t0) * 1000.0
    no_regression = post_probation_ms < naive_ms * 2.0  # generous margin, catches catastrophic regressions only
    print(f"    post-probation call: {post_probation_ms:.2f}ms {'OK' if no_regression else 'FAIL -- still regressed'}")
    ok = ok and no_regression

    return ok


def part1():
    import fusionml
    print("=" * 70)
    print("  Part 1: default CPU+GPU scheduler (ANE off) + naive-fallback safety")
    print("=" * 70)
    all_ok = True
    scheduler = fusionml.TriComputeMatmul(enable_ane=False)
    for i, (M, K, N) in enumerate(SHAPES):
        ok = check_shape(scheduler, M, K, N, seed=100 + i)
        all_ok = all_ok and ok
    print("\nPART 1 VERDICT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


def part2():
    import fusionml
    print("=" * 70)
    print("  Part 2 (EXPERIMENTAL): ANE opt-in for ONE shape, fallback for others")
    print("  Not required to pass -- see smart_matmul.py docstring for why.")
    print("=" * 70)
    M0, K0, N0 = SHAPES[0]
    np.random.seed(100)
    a0 = (np.random.randn(M0, K0) * 0.02).astype(np.float32)
    b0 = (np.random.randn(K0, N0) * 0.02).astype(np.float32)

    print(f"  Prewarming ANE for shape {(M0, K0, N0)} in an isolated subprocess...")
    prewarmed = fusionml.prewarm_ane_shape(M0, K0, N0, b0)
    print(f"  Prewarm result: {'OK' if prewarmed else 'FAILED (ANE unavailable or compile failed)'}")
    if not prewarmed:
        print("  Skipping -- ANE unavailable on this machine.")
        return

    scheduler = fusionml.TriComputeMatmul(enable_ane=True)
    print(f"  is_ane_prewarmed(shape0) = {fusionml.is_ane_prewarmed(M0, K0, N0, b0)} (expect True)")
    ok = check_shape(scheduler, M0, K0, N0, seed=100)
    print(f"  ANE locked shape: {scheduler._ane_locked_shape} (expect {(M0, K0, N0)})")

    for i, (M, K, N) in enumerate(SHAPES[1:], start=1):
        ok = check_shape(scheduler, M, K, N, seed=100 + i) and ok
    print(f"  ANE locked shape after all calls: {scheduler._ane_locked_shape} (expect unchanged)")
    print("\nPART 2 RESULT (informational only):", "PASS" if ok else "FAIL")


def main():
    print("Running Part 1 as an isolated subprocess...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python"))
    r1 = subprocess.run([sys.executable, "-u", __file__, "--part1"], env=env, timeout=60)
    part1_ok = r1.returncode == 0
    print(f"\nPart 1 subprocess exit code: {r1.returncode} ({'PASS' if part1_ok else 'FAIL/CRASH'})")

    print("\nRunning Part 2 as an isolated subprocess (experimental, may crash)...")
    r2 = subprocess.run([sys.executable, "-u", __file__, "--part2"], env=env, timeout=60)
    print(f"\nPart 2 subprocess exit code: {r2.returncode} "
          f"({'OK' if r2.returncode == 0 else 'CRASHED/FAILED -- expected possible, not blocking'})")

    print("\n" + "=" * 70)
    print("OVERALL MVP VERDICT (Part 1 only -- Part 2 is experimental):",
          "PASS" if part1_ok else "FAIL")
    print("=" * 70)
    sys.exit(0 if part1_ok else 1)


if __name__ == "__main__":
    if "--part1" in sys.argv:
        part1()
    elif "--part2" in sys.argv:
        part2()
    else:
        main()
