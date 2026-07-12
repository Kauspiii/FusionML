#!/usr/bin/env python3
"""
tricompute_throttle_guard_test.py — validate ThrottleGuard under the same 90s
sustained-load conditions that exposed +10.9% degradation for a static
tri-compute split. Confirms:

1. Throughput never falls behind GPU-only (the safety property).
2. Output correctness is preserved throughout (dynamic ratio changes must not
   corrupt results -- "same response you get normally").
3. Reports tokens/sec across the run, same convention as the static test.
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
TOTAL_ROWS = 8192
# Cold-calibrated "ideal" ratios from tricompute_adaptive_search.py's best split at 8192 rows
DEFAULT_RATIOS = {"gpu": 3611 / TOTAL_ROWS, "cpu": 2208 / TOTAL_ROWS, "ane": 2373 / TOTAL_ROWS}


def _make_a_b():
    np.random.seed(42)
    a_full = (np.random.randn(TOTAL_ROWS, K) * 0.02).astype(np.float32)
    b = (np.random.randn(K, N) * 0.02).astype(np.float32)
    return a_full, b


def prewarm_one_ane_tier(tier_rows):
    """
    Compile ONE ANE shape and let ane_matmul's disk cache (.mlpackage under
    ~/.cache/fusionml/ane/) persist it. Run in its OWN subprocess -- repeated
    CoreML/MIL compilation within a single process was found to segfault
    (exit 139, no Python traceback) after the first compile. Matches the
    pattern already proven safe in ane_spike_test.py (one compile per
    subprocess). The main sustained-loop process then only ever LOADS
    cached models, never compiles in-process more than once.
    """
    from fusionml._metal.ane_backend import ane_matmul
    a_full, b = _make_a_b()
    a_slice = a_full[-tier_rows:]
    ane_matmul(a_slice, b, compute_units="CPU_AND_NE")


def run_guarded():
    import mlx.core as mx
    from fusionml._metal.ane_backend import ane_matmul
    from fusionml._metal.throttle_guard import ThrottleGuard

    a_full, b = _make_a_b()
    b_mx = mx.array(b)
    b_mx_all = b_mx  # reused
    expected_full = (a_full.astype(np.float64) @ b.astype(np.float64))

    guard = ThrottleGuard()

    # Load every ANE tier's model from disk into THIS process's in-memory
    # cache, single-threaded, before any threading starts. All tiers were
    # already compiled+saved to disk by isolated prewarm subprocesses (see
    # main()), so this is a cache load, not a compile -- but it still must
    # happen single-threaded: loading an MLModel for the first time in this
    # process from a background thread (even from disk, even without
    # concurrent compilation) segfaulted when it raced against MLX GPU
    # activity on another thread. Loading here, before any thread exists,
    # avoids that race entirely.
    full_gpu_rows = int(TOTAL_ROWS * DEFAULT_RATIOS["gpu"])
    full_cpu_rows = int(TOTAL_ROWS * DEFAULT_RATIOS["cpu"])
    full_ane_rows = TOTAL_ROWS - full_gpu_rows - full_cpu_rows
    for tier_rows in guard.ane_row_tiers(full_ane_rows):
        if tier_rows > 0:
            ane_matmul(a_full[-tier_rows:], b, compute_units="CPU_AND_NE")

    def run_one(rows_dict):
        gpu_rows, cpu_rows, ane_rows = rows_dict["gpu"], rows_dict["cpu"], rows_dict["ane"]

        a_gpu = a_full[:gpu_rows] if gpu_rows > 0 else None
        a_cpu = a_full[gpu_rows:gpu_rows + cpu_rows] if cpu_rows > 0 else None
        a_ane = a_full[gpu_rows + cpu_rows:] if ane_rows > 0 else None

        out = {}

        def gpu_task():
            r = mx.array(a_gpu) @ b_mx_all
            mx.eval(r)              # sync/compute only -- no readback from this thread
            out["gpu_mx"] = r       # readback to numpy happens on the main thread below

        def cpu_task():
            out["cpu"] = a_cpu @ b

        def ane_task():
            out["ane"] = ane_matmul(a_ane, b, compute_units="CPU_AND_NE")

        threads = []
        if a_gpu is not None:
            threads.append(threading.Thread(target=gpu_task))
        if a_cpu is not None:
            threads.append(threading.Thread(target=cpu_task))
        if a_ane is not None:
            threads.append(threading.Thread(target=ane_task))

        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall_ms = (time.perf_counter() - t0) * 1000.0

        # Metal readback (mx.array -> numpy) done single-threaded, after all
        # threads have joined -- doing this concurrently from within gpu_task
        # while ANE/CPU threads were also active caused a native SIGSEGV
        # (see investigation: exit code 139, no Python traceback since it's
        # a crash below the interpreter, not a catchable exception).
        if "gpu_mx" in out:
            out["gpu"] = np.array(out["gpu_mx"])

        # Correctness: stitch results back in row order and compare to reference
        pieces = []
        if "gpu" in out:
            pieces.append(out["gpu"])
        if "cpu" in out:
            pieces.append(out["cpu"])
        if "ane" in out:
            pieces.append(out["ane"])
        combined = np.concatenate(pieces, axis=0).astype(np.float64)
        rel_err = float(np.max(np.abs(combined - expected_full)) / (np.max(np.abs(expected_full)) + 1e-8))

        return wall_ms, rel_err, gpu_rows, cpu_rows, ane_rows

    mx.eval(mx.array(a_full[:full_gpu_rows]) @ b_mx)  # GPU warm compile

    records = []
    start = time.perf_counter()
    while (time.perf_counter() - start) < DURATION_SEC:
        rows = guard.recommend_rows(TOTAL_ROWS, DEFAULT_RATIOS)
        wall_ms, rel_err, gpu_rows, cpu_rows, ane_rows = run_one(rows)
        guard.record(wall_ms)
        records.append({
            "wall_ms": wall_ms, "rel_err": rel_err,
            "gpu_rows": gpu_rows, "cpu_rows": cpu_rows, "ane_rows": ane_rows,
        })

    times = np.array([r["wall_ms"] for r in records])
    rel_errs = np.array([r["rel_err"] for r in records])
    n = len(times)
    third = max(1, n // 3)
    early, mid, late = times[:third], times[third:2 * third], times[2 * third:]
    tokens_per_sec = TOTAL_ROWS * 1000.0 / times

    return {
        "n_iterations": n,
        "median_ms": float(np.median(times)),
        "early_third_median_ms": float(np.median(early)),
        "mid_third_median_ms": float(np.median(mid)),
        "late_third_median_ms": float(np.median(late)),
        "degradation_early_to_late_pct": float((np.median(late) - np.median(early)) / np.median(early) * 100),
        "tokens_per_sec_median": float(np.median(tokens_per_sec)),
        "tokens_per_sec_early_third": float(np.median(TOTAL_ROWS * 1000.0 / early)),
        "tokens_per_sec_late_third": float(np.median(TOTAL_ROWS * 1000.0 / late)),
        "max_rel_err_over_run": float(np.max(rel_errs)),
        "mean_rel_err_over_run": float(np.mean(rel_errs)),
        "guard_summary": guard.summary(),
        "ane_rows_over_time_sample": [r["ane_rows"] for r in records[::max(1, n // 20)]],
    }


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python"))
    return env


def run_sub():
    cmd = [sys.executable, __file__, "--sub"]
    res = subprocess.run(cmd, capture_output=True, text=True, env=_env(), timeout=int(DURATION_SEC) + 60)
    if res.returncode != 0:
        return {"error": f"subprocess crashed (exit {res.returncode}): {res.stderr[-1200:]}"}
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {"error": "no JSON output", "stdout": res.stdout[-500:]}


def run_prewarm_tier(tier_rows):
    cmd = [sys.executable, __file__, "--prewarm-tier", str(tier_rows)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=_env(), timeout=60)
    if res.returncode != 0:
        return False, res.stderr[-800:]
    return True, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--prewarm-tier", type=int, default=None)
    args = parser.parse_args()

    if args.prewarm_tier is not None:
        if args.prewarm_tier > 0:
            prewarm_one_ane_tier(args.prewarm_tier)
        return

    if args.sub:
        print(json.dumps(run_guarded()))
        return

    print("=" * 70)
    print("  ThrottleGuard Validation — 90s sustained load")
    print(f"  Cold-calibrated default ratios: {DEFAULT_RATIOS}")
    print("=" * 70)

    from fusionml._metal.throttle_guard import ThrottleGuard
    full_gpu_rows = int(TOTAL_ROWS * DEFAULT_RATIOS["gpu"])
    full_cpu_rows = int(TOTAL_ROWS * DEFAULT_RATIOS["cpu"])
    full_ane_rows = TOTAL_ROWS - full_gpu_rows - full_cpu_rows
    tiers = ThrottleGuard().ane_row_tiers(full_ane_rows)

    print(f"\nPre-warming {len(tiers)} ANE tier(s) in isolated subprocesses: {tiers}")
    for tier_rows in tiers:
        if tier_rows <= 0:
            continue
        ok, err = run_prewarm_tier(tier_rows)
        print(f"  tier rows={tier_rows}: {'OK' if ok else 'FAILED: ' + str(err)}")
        if not ok:
            print("ERROR: pre-warming failed, aborting.")
            return

    r = run_sub()
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return

    print(f"\nIterations completed: {r['n_iterations']}")
    print(f"Median: {r['median_ms']:.2f}ms")
    print(f"Early third: {r['early_third_median_ms']:.2f}ms  |  "
          f"Mid third: {r['mid_third_median_ms']:.2f}ms  |  "
          f"Late third: {r['late_third_median_ms']:.2f}ms")
    print(f"Degradation early->late: {r['degradation_early_to_late_pct']:+.1f}%  "
          f"(unguarded static split was +10.9%)")
    print(f"Tokens/sec: median={r['tokens_per_sec_median']:.0f}  "
          f"early={r['tokens_per_sec_early_third']:.0f}  late={r['tokens_per_sec_late_third']:.0f}")
    print(f"\nCorrectness: max_rel_err over entire run={r['max_rel_err_over_run']:.4e}  "
          f"mean_rel_err={r['mean_rel_err_over_run']:.4e}")
    print(f"\nGuard summary: {r['guard_summary']}")
    print(f"\nANE row count sample over time (should shrink if throttling detected): "
          f"{r['ane_rows_over_time_sample']}")

    # Compare against the known GPU-only sustained baseline from the prior test
    GPU_ONLY_LATE_MS = 109.65  # from tricompute_sustained_test.json
    print(f"\nGPU-only late-third baseline (reference): {GPU_ONLY_LATE_MS:.2f}ms")
    print(f"Guarded tri-compute late-third: {r['late_third_median_ms']:.2f}ms")
    if r['late_third_median_ms'] < GPU_ONLY_LATE_MS:
        speedup = GPU_ONLY_LATE_MS / r['late_third_median_ms']
        print(f"SAFE: guarded tri-compute still beats GPU-only even in the late/hot phase ({speedup:.3f}x)")
    else:
        print("WARNING: guarded tri-compute fell behind GPU-only -- guard thresholds need tuning")

    out_path = os.path.join(os.path.dirname(__file__), "../results",
                             "Apple_M1_8GB_8CPU_7GPU_16ANE", "tricompute_throttle_guard_test.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
