#!/usr/bin/env python3
"""Isolate whether loading 3 DIFFERENT disk-cached ANE models sequentially,
single-threaded, in one process (no threading at all) crashes -- matching
run_guarded()'s warm-up loop exactly, minus any concurrency."""
import numpy as np

K, N = 1600, 6400
TOTAL_ROWS = 8192
TIERS = [2373, 1566, 783]


def make_a_b():
    np.random.seed(42)
    a_full = (np.random.randn(TOTAL_ROWS, K) * 0.02).astype(np.float32)
    b = (np.random.randn(K, N) * 0.02).astype(np.float32)
    return a_full, b


def main():
    import mlx.core as mx
    from fusionml._metal.ane_backend import ane_matmul

    a_full, b = make_a_b()
    b_mx = mx.array(b)
    mx.eval(mx.array(a_full[:1000]) @ b_mx)
    print("MLX GPU touch done.", flush=True)

    for tier_rows in TIERS:
        print(f"Loading tier {tier_rows}...", flush=True)
        out = ane_matmul(a_full[-tier_rows:], b, compute_units="CPU_AND_NE")
        print(f"Tier {tier_rows} done, output shape {out.shape}", flush=True)

    print("SUCCESS: all tiers loaded sequentially without crashing.")


if __name__ == "__main__":
    main()
