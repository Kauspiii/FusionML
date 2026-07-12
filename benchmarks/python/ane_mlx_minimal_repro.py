#!/usr/bin/env python3
"""
ane_mlx_minimal_repro.py — minimal isolation of the SIGSEGV (exit -11) seen
in tricompute_throttle_guard_test.py after three targeted fixes failed.

Tests, one flag at a time, whether:
  --mlx-first: touching MLX (GPU) before loading a disk-cached ANE model crashes
  --thread: calling ane_matmul from a background thread (no concurrent MLX) crashes
  --both: MLX activity AND ane_matmul from a thread, concurrently, crashes
  (baseline, no flags): just load+run a disk-cached ANE model, nothing else

Requires a tier already disk-cached by an earlier run (2373 rows, K=1600,N=6400,
seed=42) -- reuses the exact same weight/shape so ane_matmul's disk cache hits.
"""

import sys
import threading
import numpy as np

K, N = 1600, 6400
TIER_ROWS = 2373
TOTAL_ROWS = 8192


def make_a_b():
    np.random.seed(42)
    a_full = (np.random.randn(TOTAL_ROWS, K) * 0.02).astype(np.float32)
    b = (np.random.randn(K, N) * 0.02).astype(np.float32)
    return a_full, b


def main():
    mlx_first = "--mlx-first" in sys.argv
    use_thread = "--thread" in sys.argv
    both = "--both" in sys.argv

    a_full, b = make_a_b()
    a_ane = a_full[-TIER_ROWS:]

    if mlx_first or both:
        print("Step: touching MLX (GPU) first...", flush=True)
        import mlx.core as mx
        a_mx = mx.array(a_full[:1000])
        b_mx = mx.array(b)
        mx.eval(a_mx @ b_mx)
        print("Step: MLX GPU op done.", flush=True)

    from fusionml._metal.ane_backend import ane_matmul

    if both:
        print("Step: launching concurrent MLX + ANE threads...", flush=True)
        import mlx.core as mx
        b_mx = mx.array(b)

        def gpu_task():
            r = mx.array(a_full[:3000]) @ b_mx
            mx.eval(r)

        def ane_task():
            print("  ane_task: calling ane_matmul...", flush=True)
            out = ane_matmul(a_ane, b, compute_units="CPU_AND_NE")
            print("  ane_task: done.", flush=True)

        t1 = threading.Thread(target=gpu_task)
        t2 = threading.Thread(target=ane_task)
        t1.start(); t2.start()
        t1.join(); t2.join()
        print("Step: both threads joined successfully.", flush=True)

    elif use_thread:
        print("Step: calling ane_matmul from a background thread (no concurrent MLX)...", flush=True)
        result = {}

        def ane_task():
            print("  ane_task: calling ane_matmul...", flush=True)
            result["out"] = ane_matmul(a_ane, b, compute_units="CPU_AND_NE")
            print("  ane_task: done.", flush=True)

        t = threading.Thread(target=ane_task)
        t.start()
        t.join()
        print("Step: thread joined successfully.", flush=True)

    else:
        print("Step: calling ane_matmul directly (main thread, no concurrency)...", flush=True)
        out = ane_matmul(a_ane, b, compute_units="CPU_AND_NE")
        print("Step: ane_matmul returned successfully.", flush=True)

    print("SUCCESS: reached end of script without crashing.")


if __name__ == "__main__":
    main()
