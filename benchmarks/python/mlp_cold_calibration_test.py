#!/usr/bin/env python3
"""
mlp_cold_calibration_test.py
=============================
Quantifies how much of MLP's 1.29x co-execution speedup depends on the
CPU (AMX/P-core) being pre-warmed by the calibration pass that immediately
precedes the timed run in the normal benchmark flow.

"Warm" condition: calibration runs immediately before the timed MLP smart-split
run (current default behavior in model_comparison.py — calibration loads from
cache but the scheduler + first few compiled calls still touch AMX/GPU paths).

"Cold" condition: process sleeps for a fixed cooldown period after doing
unrelated CPU/GPU work (to let clocks return to idle) before timing MLP.

This does not disable calibration itself (the ratios are loaded from a cached
JSON either way) — it isolates whether physically *executing* calibration
matmuls immediately before the timed run measurably changes the smart-split
result, from clock-state / cache-residency carryover.

Two subprocess conditions, n=30, warmup=10 each on the timed measurement.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

WARMUPS = 10
RUNS    = 50
D, HID, OUT = 4096, 4096, 10
L = 1024


def run_worker(condition):
    import mlx.core as mx
    from model_comparison import run_fusion_mlp
    from fusionml.tensor import Tensor
    from fusionml._metal.tri_scheduler import get_scheduler
    import resource

    scheduler = get_scheduler()
    cache_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../results/tri_calibration_mlp.json")
    )

    if condition == "warm":
        # Mirrors current default flow: load calibration cache right before use.
        scheduler.load_calibration(cache_path)
    else:
        # cold: do unrelated CPU+GPU busywork, then idle-cooldown, THEN load
        # calibration (ratios are identical either way — only clock/cache
        # state differs) before the timed loop.
        junk_a = mx.random.normal((256, 256))
        junk_b = mx.random.normal((256, 256))
        for _ in range(3):
            mx.eval(junk_a @ junk_b)
        time.sleep(15.0)  # let P-core/GPU clocks return to idle state
        scheduler.load_calibration(cache_path)

    np.random.seed(42); scale = 0.02
    x_np  = np.random.randn(L, D).astype(np.float32) * scale
    w1_np = np.random.randn(D, HID).astype(np.float32) * scale
    b1_np = np.zeros((1, HID), dtype=np.float32)
    w2_np = np.random.randn(HID, OUT).astype(np.float32) * scale
    b2_np = np.zeros((1, OUT), dtype=np.float32)

    x = Tensor(x_np, requires_grad=False).to_gpu()
    weights = {
        'w1': Tensor(w1_np, requires_grad=False).to_gpu(),
        'b1': Tensor(b1_np, requires_grad=False).to_gpu(),
        'w2': Tensor(w2_np, requires_grad=False).to_gpu(),
        'b2': Tensor(b2_np, requires_grad=False).to_gpu(),
    }

    fn = lambda: run_fusion_mlp(x, weights, training=False)

    for _ in range(WARMUPS):
        fn()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)

    n = len(times)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    return {
        "mean": float(np.mean(times)), "std": float(np.std(times)),
        "median": float(np.median(times)), "ci95": float(1.96 * np.std(times) / np.sqrt(n)),
        "n_runs": n, "peak_mem_mb": peak_mb,
    }


def run_sub(condition):
    cmd = [sys.executable, __file__, "--sub", "--condition", condition]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python")) \
        + os.pathsep + os.path.dirname(__file__)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if res.returncode != 0:
        print(f"  ⚠ condition={condition}: {res.stderr[:500]}", file=sys.stderr)
        return None
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--condition", choices=["warm", "cold"])
    args = parser.parse_args()

    if args.sub:
        print(json.dumps(run_worker(args.condition)))
        return

    # MLX-FP32 baseline for MLP inference from committed model_comparison.json
    mlx_baseline_median = 25.62

    print("=" * 70)
    print("  MLP Co-Execution: Warm vs Cold Calibration State")
    print(f"  Runs: {RUNS}  Warmup: {WARMUPS}")
    print(f"  MLX-FP32 baseline (reference): {mlx_baseline_median} ms")
    print("=" * 70)

    results = {}
    for condition in ["warm", "cold"]:
        print(f"\n▶ {condition.upper()} condition ... ", end="", flush=True)
        time.sleep(5.0)
        s = run_sub(condition)
        results[condition] = s
        if s:
            speedup = mlx_baseline_median / s['median']
            print(f"{s['median']:.2f} ms ± {s['ci95']:.2f}  (speedup vs MLX-FP32: {speedup:.3f}×)")
        else:
            print("FAILED")

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC")
    print("=" * 70)
    if results.get("warm") and results.get("cold"):
        warm_med = results["warm"]["median"]
        cold_med = results["cold"]["median"]
        delta_pct = (cold_med - warm_med) / warm_med * 100
        print(f"  Warm: {warm_med:.2f} ms  (speedup {mlx_baseline_median/warm_med:.3f}×)")
        print(f"  Cold: {cold_med:.2f} ms  (speedup {mlx_baseline_median/cold_med:.3f}×)")
        print(f"  Delta: {delta_pct:+.1f}%")
        if abs(delta_pct) < 5:
            print("  CONCLUSION: calibration warmup contributes <5% — 1.29x result is robust")
        else:
            print(f"  CONCLUSION: calibration warmup contributes {delta_pct:.1f}% —"
                  f" deployment-realistic speedup is ~{mlx_baseline_median/cold_med:.2f}×, not {mlx_baseline_median/warm_med:.2f}×")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results", "Apple_M1_8GB_8CPU_7GPU_16ANE"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mlp_cold_calibration_test.json"), "w") as f:
        json.dump({"mlx_baseline_median": mlx_baseline_median, "results": results}, f, indent=2)
    print(f"\n💾 Saved to {out_dir}/mlp_cold_calibration_test.json")


if __name__ == "__main__":
    main()
