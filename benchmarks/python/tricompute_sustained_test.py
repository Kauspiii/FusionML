#!/usr/bin/env python3
"""
tricompute_sustained_test.py — does the tri-compute (CPU+GPU+ANE) advantage
survive sustained load, or does concurrent 3-unit heat/power draw trigger
thermal throttling faster than GPU-only on this fanless M1?

Runs each condition back-to-back with NO artificial cooldown for a fixed
WALL-CLOCK duration (not a fixed iteration count) -- M1 throttling needs real
sustained heat buildup over tens of seconds, not a handful of quick calls.
Records the full per-iteration time series so a step-change or progressive
slowdown is visible, not just a summary median.

Also reports tokens/sec (rows/sec here, as a throughput proxy for this FFN op)
alongside latency, matching model_comparison.py's existing convention.
"""

import os
import sys
import time
import threading
import subprocess
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

K, N = 1600, 6400
DURATION_SEC = 90.0

# Best tri-compute split found by the adaptive search at 8192 rows
TOTAL_ROWS = 8192
GPU_ROWS, CPU_ROWS, ANE_ROWS = 3611, 2208, 2373


def run_condition(mode):
    import mlx.core as mx
    from fusionml._metal.ane_backend import ane_matmul

    np.random.seed(42)
    a_full = (np.random.randn(TOTAL_ROWS, K) * 0.02).astype(np.float32)
    b = (np.random.randn(K, N) * 0.02).astype(np.float32)
    b_mx = mx.array(b)

    if mode == "gpu_only":
        a_gpu_all_mx = mx.array(a_full)
        for _ in range(5):
            mx.eval(a_gpu_all_mx @ b_mx)

        def one_iter():
            t0 = time.perf_counter()
            mx.eval(a_gpu_all_mx @ b_mx)
            return (time.perf_counter() - t0) * 1000.0

    else:  # tricompute
        a_gpu = a_full[:GPU_ROWS]
        a_cpu = a_full[GPU_ROWS:GPU_ROWS + CPU_ROWS]
        a_ane = a_full[GPU_ROWS + CPU_ROWS:]

        mx.eval(mx.array(a_gpu) @ b_mx)
        ane_matmul(a_ane, b, compute_units="CPU_AND_NE")

        def one_iter():
            result = {}

            def gpu_task():
                mx.eval(mx.array(a_gpu) @ b_mx)

            def cpu_task():
                _ = a_cpu @ b

            def ane_task():
                _ = ane_matmul(a_ane, b, compute_units="CPU_AND_NE")

            threads = [threading.Thread(target=gpu_task),
                       threading.Thread(target=cpu_task),
                       threading.Thread(target=ane_task)]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return (time.perf_counter() - t0) * 1000.0

    times = []
    start = time.perf_counter()
    while (time.perf_counter() - start) < DURATION_SEC:
        times.append(one_iter())

    times = np.array(times)
    n = len(times)
    # Split into thirds to see early/mid/late trend
    third = max(1, n // 3)
    early = times[:third]
    mid = times[third:2 * third]
    late = times[2 * third:]

    tokens_per_sec = TOTAL_ROWS * 1000.0 / times  # per-iteration throughput proxy

    return {
        "mode": mode,
        "duration_sec": DURATION_SEC,
        "n_iterations": n,
        "times_ms": times.tolist(),
        "median_ms": float(np.median(times)),
        "p10_ms": float(np.percentile(times, 10)),
        "p90_ms": float(np.percentile(times, 90)),
        "early_third_median_ms": float(np.median(early)),
        "mid_third_median_ms": float(np.median(mid)),
        "late_third_median_ms": float(np.median(late)),
        "degradation_early_to_late_pct": float((np.median(late) - np.median(early)) / np.median(early) * 100),
        "tokens_per_sec_median": float(np.median(tokens_per_sec)),
        "tokens_per_sec_early_third": float(np.median(TOTAL_ROWS * 1000.0 / early)),
        "tokens_per_sec_late_third": float(np.median(TOTAL_ROWS * 1000.0 / late)),
    }


def run_sub(mode):
    cmd = [sys.executable, __file__, "--sub", "--mode", mode]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python"))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=int(DURATION_SEC) + 60)
    if res.returncode != 0:
        return {"error": f"subprocess crashed: {res.stderr[-1000:]}"}
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {"error": "no JSON output", "stdout": res.stdout[-500:]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--mode", choices=["gpu_only", "tricompute"])
    args = parser.parse_args()

    if args.sub:
        print(json.dumps(run_condition(args.mode)))
        return

    print("=" * 70)
    print("  Sustained-Load Thermal Test (90s continuous, no cooldown)")
    print(f"  Shape: [{TOTAL_ROWS}x{K}] @ [{K}x{N}]  tri-compute split: "
          f"gpu={GPU_ROWS} cpu={CPU_ROWS} ane={ANE_ROWS}")
    print("=" * 70)

    results = {}
    for mode in ["gpu_only", "tricompute"]:
        print(f"\n▶ {mode} -- running for {DURATION_SEC:.0f}s continuously...")
        r = run_sub(mode)
        results[mode] = r
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  Iterations completed: {r['n_iterations']}")
        print(f"  Median: {r['median_ms']:.2f}ms  (p10={r['p10_ms']:.2f}  p90={r['p90_ms']:.2f})")
        print(f"  Early third: {r['early_third_median_ms']:.2f}ms  |  "
              f"Mid third: {r['mid_third_median_ms']:.2f}ms  |  "
              f"Late third: {r['late_third_median_ms']:.2f}ms")
        print(f"  Degradation early->late: {r['degradation_early_to_late_pct']:+.1f}%")
        print(f"  Tokens/sec: median={r['tokens_per_sec_median']:.0f}  "
              f"early={r['tokens_per_sec_early_third']:.0f}  late={r['tokens_per_sec_late_third']:.0f}")

    if "error" not in results.get("gpu_only", {}) and "error" not in results.get("tricompute", {}):
        print("\n" + "=" * 70)
        print("  SUSTAINED-LOAD COMPARISON")
        print("=" * 70)
        go, tc = results["gpu_only"], results["tricompute"]
        print(f"  GPU-only median:    {go['median_ms']:.2f}ms  ({go['tokens_per_sec_median']:.0f} tok/s)")
        print(f"  Tri-compute median: {tc['median_ms']:.2f}ms  ({tc['tokens_per_sec_median']:.0f} tok/s)")
        print(f"  Sustained speedup (median vs median): {go['median_ms']/tc['median_ms']:.3f}x")
        print(f"  GPU-only degradation over 90s:    {go['degradation_early_to_late_pct']:+.1f}%")
        print(f"  Tri-compute degradation over 90s: {tc['degradation_early_to_late_pct']:+.1f}%")
        if tc['degradation_early_to_late_pct'] > go['degradation_early_to_late_pct'] + 5:
            print("  VERDICT: tri-compute throttles MORE than GPU-only under sustained load --")
            print("           the burst-measured speedup likely overstates real sustained throughput.")
        elif abs(tc['degradation_early_to_late_pct'] - go['degradation_early_to_late_pct']) <= 5:
            print("  VERDICT: both conditions throttle similarly (or not at all) -- the earlier")
            print("           iteration-4 collapse in the adaptive search was likely a transient")
            print("           blip, not evidence that tri-compute specifically thermal-limits worse.")
        else:
            print("  VERDICT: tri-compute degrades LESS than GPU-only under sustained load.")

    out_path = os.path.join(os.path.dirname(__file__), "../results",
                             "Apple_M1_8GB_8CPU_7GPU_16ANE", "tricompute_sustained_test.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
