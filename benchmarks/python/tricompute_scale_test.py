#!/usr/bin/env python3
"""
tricompute_scale_test.py — empirically test whether true concurrent 3-way
(CPU+GPU+ANE) execution beats GPU-only at a batched/larger-context scale,
where ANE's ~24ms fixed dispatch cost can plausibly be amortized.

Uses real threading.Thread for genuine concurrent dispatch -- MLX (GPU),
Accelerate/numpy (CPU), and CoreML predict() (ANE) are all native C-extension
calls that release the GIL during compute, so real OS-level parallelism
across threads is physically possible in Python (not simulated/sequential).

Compares against the naive GPU-only baseline and reports the ACHIEVED speedup
vs the THEORETICAL ceiling computed from the fixed+marginal cost model.
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
WARMUPS = 5
RUNS = 15


def run_worker(total_rows, gpu_rows, cpu_rows, ane_rows):
    import mlx.core as mx
    from fusionml._metal.ane_backend import ane_matmul, HAS_COREML

    np.random.seed(42)
    a_full = (np.random.randn(total_rows, K) * 0.02).astype(np.float32)
    b = (np.random.randn(K, N) * 0.02).astype(np.float32)
    b_mx = mx.array(b)

    a_gpu = a_full[:gpu_rows] if gpu_rows > 0 else None
    a_cpu = a_full[gpu_rows:gpu_rows + cpu_rows] if cpu_rows > 0 else None
    a_ane = a_full[gpu_rows + cpu_rows:] if ane_rows > 0 else None

    # Warm caches: GPU compile, ANE model compile+cache, CPU (nothing needed)
    if a_gpu is not None:
        mx.eval(mx.array(a_gpu) @ b_mx)
    if a_ane is not None and HAS_COREML:
        ane_matmul(a_ane, b, compute_units="CPU_AND_NE")  # compiles + caches

    def gpu_task(result):
        t0 = time.perf_counter()
        r = mx.array(a_gpu) @ b_mx
        mx.eval(r)
        result["gpu"] = (time.perf_counter() - t0) * 1000.0

    def cpu_task(result):
        t0 = time.perf_counter()
        _ = a_cpu @ b
        result["cpu"] = (time.perf_counter() - t0) * 1000.0

    def ane_task(result):
        t0 = time.perf_counter()
        _ = ane_matmul(a_ane, b, compute_units="CPU_AND_NE")
        result["ane"] = (time.perf_counter() - t0) * 1000.0

    def run_concurrent():
        result = {}
        threads = []
        if a_gpu is not None:
            threads.append(threading.Thread(target=gpu_task, args=(result,)))
        if a_cpu is not None:
            threads.append(threading.Thread(target=cpu_task, args=(result,)))
        if a_ane is not None:
            threads.append(threading.Thread(target=ane_task, args=(result,)))
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return wall_ms, result

    for _ in range(WARMUPS):
        run_concurrent()

    wall_times = []
    per_unit = {"gpu": [], "cpu": [], "ane": []}
    for _ in range(RUNS):
        wall_ms, result = run_concurrent()
        wall_times.append(wall_ms)
        for k in per_unit:
            if k in result:
                per_unit[k].append(result[k])

    # GPU-only baseline for the same total_rows
    a_all_mx = mx.array(a_full)
    for _ in range(WARMUPS):
        mx.eval(a_all_mx @ b_mx)
    gpu_only_times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        mx.eval(a_all_mx @ b_mx)
        gpu_only_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "total_rows": total_rows, "gpu_rows": gpu_rows, "cpu_rows": cpu_rows, "ane_rows": ane_rows,
        "concurrent_wall_median_ms": float(np.median(wall_times)),
        "concurrent_wall_std_ms": float(np.std(wall_times)),
        "gpu_only_median_ms": float(np.median(gpu_only_times)),
        "per_unit_median_ms": {k: (float(np.median(v)) if v else None) for k, v in per_unit.items()},
        "achieved_speedup": float(np.median(gpu_only_times)) / float(np.median(wall_times)),
    }


def run_sub(total_rows, gpu_rows, cpu_rows, ane_rows):
    cmd = [sys.executable, __file__, "--sub",
           "--total", str(total_rows), "--gpu", str(gpu_rows),
           "--cpu", str(cpu_rows), "--ane", str(ane_rows)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python"))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    if res.returncode != 0:
        return {"error": f"subprocess crashed: {res.stderr[-800:]}"}
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {"error": "no JSON output", "stdout": res.stdout[-500:]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--total", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--ane", type=int)
    args = parser.parse_args()

    if args.sub:
        print(json.dumps(run_worker(args.total, args.gpu, args.cpu, args.ane)))
        return

    print("=" * 70)
    print("  True Concurrent 3-Way (CPU+GPU+ANE) Scale Test")
    print(f"  Shape family: [Rows x {K}] @ [{K}x{N}]  (GPT-2 XL FFN)")
    print("=" * 70)

    # Theoretical-optimal splits computed from the fixed+marginal cost model
    # (gpu: 13.24 us/row, cpu: 23.98 us/row, ane: 24.2ms fixed + 6.91 us/row)
    configs = [
        (4096,  3096, 1000, 0),      # ANE excluded at this scale per the model -- 2-way only, sanity check
        (4096,  2400,  996,  700),   # ANE included per optimal-T model (~17% share)
        (8192,  4800, 1990, 1402),   # ANE ~36% notional share (scaled proportionally)
    ]

    results = []
    for total, gpu_r, cpu_r, ane_r in configs:
        print(f"\n▶ total={total}  gpu={gpu_r} cpu={cpu_r} ane={ane_r}")
        time.sleep(3.0)
        r = run_sub(total, gpu_r, cpu_r, ane_r)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  Concurrent wall: median={r['concurrent_wall_median_ms']:.2f}ms std={r['concurrent_wall_std_ms']:.2f}ms")
        print(f"  Per-unit: {r['per_unit_median_ms']}")
        print(f"  GPU-only baseline: {r['gpu_only_median_ms']:.2f}ms")
        print(f"  ACHIEVED speedup: {r['achieved_speedup']:.3f}x")

    out_path = os.path.join(os.path.dirname(__file__), "../results",
                             "Apple_M1_8GB_8CPU_7GPU_16ANE", "tricompute_scale_test.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
