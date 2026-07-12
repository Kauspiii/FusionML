#!/usr/bin/env python3
"""
tricompute_adaptive_search.py — find the true contention-aware optimal 3-way
(CPU+GPU+ANE) split via iterative feedback, since the isolated-rate model
(tricompute_scale_test.py) proved wrong once all three units run concurrently
(GPU's real throughput degrades under CPU/ANE memory-bandwidth contention).

Algorithm: measure per-unit wall time for a candidate split, then rebalance
rows proportionally toward equalizing finish time (classic load-balancing
fixed-point iteration), repeat until converged or max iterations.

Runs entirely within one subprocess so the ANE CoreML model (for whatever
ane_rows shape is current) stays warm across iterations where possible, and
so thread/warmup cost isn't paid per-iteration via process relaunch.
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
ANE_FIXED_FLOOR_ROWS = 300  # below this, ANE's fixed overhead makes any share pointless
MAX_ITERS = 5
MEASURE_RUNS = 8


def measure_split(a_full, b, b_mx, gpu_rows, cpu_rows, ane_rows):
    import mlx.core as mx
    from fusionml._metal.ane_backend import ane_matmul

    a_gpu = a_full[:gpu_rows] if gpu_rows > 0 else None
    a_cpu = a_full[gpu_rows:gpu_rows + cpu_rows] if cpu_rows > 0 else None
    a_ane = a_full[gpu_rows + cpu_rows:gpu_rows + cpu_rows + ane_rows] if ane_rows > 0 else None

    if a_gpu is not None:
        mx.eval(mx.array(a_gpu) @ b_mx)  # warm compile
    if a_ane is not None:
        ane_matmul(a_ane, b, compute_units="CPU_AND_NE")  # warm compile+cache

    def gpu_task(result):
        t0 = time.perf_counter()
        mx.eval(mx.array(a_gpu) @ b_mx)
        result["gpu"] = (time.perf_counter() - t0) * 1000.0

    def cpu_task(result):
        t0 = time.perf_counter()
        _ = a_cpu @ b
        result["cpu"] = (time.perf_counter() - t0) * 1000.0

    def ane_task(result):
        t0 = time.perf_counter()
        out = ane_matmul(a_ane, b, compute_units="CPU_AND_NE")
        result["ane"] = (time.perf_counter() - t0) * 1000.0
        result["ane_out"] = out

    wall_times, per_unit = [], {"gpu": [], "cpu": [], "ane": []}
    last_ane_out = None
    for _ in range(MEASURE_RUNS):
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
        wall_times.append((time.perf_counter() - t0) * 1000.0)
        for k in per_unit:
            if k in result:
                per_unit[k].append(result[k])
        if "ane_out" in result:
            last_ane_out = result["ane_out"]

    # Correctness check for ANE's chunk (only needs checking once per shape)
    ane_correct = None
    if a_ane is not None and last_ane_out is not None:
        expected = a_ane.astype(np.float64) @ b.astype(np.float64)
        rel_err = float(np.max(np.abs(last_ane_out.astype(np.float64) - expected)) / (np.max(np.abs(expected)) + 1e-8))
        ane_correct = rel_err

    return {
        "wall_median_ms": float(np.median(wall_times)),
        "per_unit_median_ms": {k: (float(np.median(v)) if v else None) for k, v in per_unit.items()},
        "ane_rel_err": ane_correct,
    }


def adaptive_search(total_rows):
    import mlx.core as mx

    np.random.seed(42)
    a_full = (np.random.randn(total_rows, K) * 0.02).astype(np.float32)
    b = (np.random.randn(K, N) * 0.02).astype(np.float32)
    b_mx = mx.array(b)

    # GPU-only baseline
    for _ in range(3):
        mx.eval(mx.array(a_full) @ b_mx)
    gpu_only_times = []
    for _ in range(MEASURE_RUNS):
        t0 = time.perf_counter()
        mx.eval(mx.array(a_full) @ b_mx)
        gpu_only_times.append((time.perf_counter() - t0) * 1000.0)
    gpu_only_ms = float(np.median(gpu_only_times))

    # Start from a rough proportional split (isolated-rate model, known to be
    # imperfect under contention -- this is just the search's starting point)
    gpu_rows = int(total_rows * 0.55)
    cpu_rows = int(total_rows * 0.25)
    ane_rows = total_rows - gpu_rows - cpu_rows
    if ane_rows < ANE_FIXED_FLOOR_ROWS:
        ane_rows = 0
        cpu_rows = total_rows - gpu_rows

    history = []
    for it in range(MAX_ITERS):
        m = measure_split(a_full, b, b_mx, gpu_rows, cpu_rows, ane_rows)
        speedup = gpu_only_ms / m["wall_median_ms"]
        history.append({
            "iter": it, "gpu_rows": gpu_rows, "cpu_rows": cpu_rows, "ane_rows": ane_rows,
            "wall_ms": m["wall_median_ms"], "per_unit": m["per_unit_median_ms"],
            "speedup_vs_gpu_only": speedup, "ane_rel_err": m["ane_rel_err"],
        })

        times = m["per_unit_median_ms"]
        # Rebalance: shift rows from slower-finishing unit's "excess" time toward
        # faster units, proportional to each unit's own measured rows/time rate.
        active = {k: v for k, v in times.items() if v is not None}
        if len(active) < 2:
            break
        target_t = np.mean(list(active.values()))  # aim for all units finishing together
        rates = {}  # rows per ms, from THIS iteration's real (contended) measurement
        cur_rows = {"gpu": gpu_rows, "cpu": cpu_rows, "ane": ane_rows}
        for k, t in active.items():
            if cur_rows[k] > 0 and t > 0:
                rates[k] = cur_rows[k] / t
        if not rates:
            break
        new_rows = {k: max(0, int(rates[k] * target_t)) for k in rates}
        # normalize to preserve total_rows exactly
        total_new = sum(new_rows.values())
        if total_new == 0:
            break
        scale = total_rows / total_new
        new_rows = {k: int(v * scale) for k, v in new_rows.items()}
        # fix rounding remainder onto gpu
        remainder = total_rows - sum(new_rows.values())
        new_rows["gpu"] = new_rows.get("gpu", 0) + remainder

        prev = (gpu_rows, cpu_rows, ane_rows)
        gpu_rows = new_rows.get("gpu", 0)
        cpu_rows = new_rows.get("cpu", 0)
        ane_rows = new_rows.get("ane", 0) if ane_rows > 0 or "ane" in new_rows else 0
        if ane_rows < ANE_FIXED_FLOOR_ROWS:
            # fold ANE's tiny leftover into CPU rather than pay ANE's fixed cost for nothing
            cpu_rows += ane_rows
            ane_rows = 0

        if (gpu_rows, cpu_rows, ane_rows) == prev:
            break  # converged

    best = max(history, key=lambda h: h["speedup_vs_gpu_only"])
    return {
        "total_rows": total_rows,
        "gpu_only_ms": gpu_only_ms,
        "history": history,
        "best": best,
    }


def run_sub(total_rows):
    cmd = [sys.executable, __file__, "--sub", "--total", str(total_rows)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python"))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
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
    parser.add_argument("--total", type=int)
    args = parser.parse_args()

    if args.sub:
        print(json.dumps(adaptive_search(args.total)))
        return

    print("=" * 70)
    print("  Adaptive Contention-Aware 3-Way Split Search")
    print("=" * 70)

    all_results = []
    for total in [2048, 4096, 8192]:
        print(f"\n▶ total_rows={total}")
        time.sleep(3.0)
        r = run_sub(total)
        all_results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  GPU-only baseline: {r['gpu_only_ms']:.2f}ms")
        for h in r["history"]:
            print(f"  iter {h['iter']}: gpu={h['gpu_rows']:5d} cpu={h['cpu_rows']:5d} ane={h['ane_rows']:5d} "
                  f"-> wall={h['wall_ms']:7.2f}ms  speedup={h['speedup_vs_gpu_only']:.3f}x  "
                  f"per_unit={h['per_unit']}  ane_rel_err={h['ane_rel_err']}")
        b = r["best"]
        print(f"  BEST: gpu={b['gpu_rows']} cpu={b['cpu_rows']} ane={b['ane_rows']} "
              f"-> {b['speedup_vs_gpu_only']:.3f}x speedup")

    out_path = os.path.join(os.path.dirname(__file__), "../results",
                             "Apple_M1_8GB_8CPU_7GPU_16ANE", "tricompute_adaptive_search.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
