#!/usr/bin/env python3
"""
cholesky_benchmark.py — the "math related problem" benchmark: blocked
Cholesky decomposition of a large symmetric positive-definite matrix,
A = L @ L.T, demonstrating fusionml.smart_matmul() on a numerical linear
algebra workload distinct from the transformer-block benchmarks (Llama-3-8B,
GPT-2 XL) elsewhere in this project.

Blocked Cholesky structure, per block column i:
  1. Factorize the small diagonal block (sequential, LAPACK-backed, not worth
     splitting -- np.linalg.cholesky).
  2. Triangular solve for the panel below the diagonal block (sequential
     dependency chain, also not matmul-parallelizable in the way that
     matters here).
  3. Trailing submatrix (Schur complement) update: A[i+1:,i+1:] -= L_panel @
     L_panel.T -- the actual matmul-heavy hot loop, and the only step
     dispatched through smart_matmul.

Compares three implementations of step 3:
  - naive:   plain numpy `@` (baseline)
  - gpu:     MLX (GPU) only
  - smart:   fusionml.smart_matmul() -- CPU+GPU tri-compute

Verifies correctness via reconstruction residual ||A - L@L.T|| / ||A||.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

import fusionml

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


def make_spd_matrix(n, seed=42):
    """Random symmetric positive-definite matrix via A = R@R.T + n*I."""
    np.random.seed(seed)
    R = np.random.randn(n, n).astype(np.float32) * 0.1
    A = R @ R.T
    A += n * np.eye(n, dtype=np.float32)  # guarantee SPD, well-conditioned
    return A


def blocked_cholesky(A, block_size, matmul_fn):
    """
    Blocked left-looking Cholesky. `matmul_fn(a, b)` computes a @ b and is
    the only pluggable hot-loop operation -- everything else (small diagonal
    factorization, triangular solve) is identical across all three variants
    being compared, so the comparison isolates the trailing-update cost.
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float32)
    A = A.copy()

    for i in range(0, n, block_size):
        bs = min(block_size, n - i)

        # Step 1: factorize the diagonal block (sequential, small, LAPACK)
        L[i:i+bs, i:i+bs] = np.linalg.cholesky(A[i:i+bs, i:i+bs])

        if i + bs >= n:
            break

        # Step 2: triangular solve for the panel below the diagonal block
        L[i+bs:, i:i+bs] = np.linalg.solve(
            L[i:i+bs, i:i+bs].T, A[i+bs:, i:i+bs].T
        ).T

        # Step 3: trailing submatrix (Schur complement) update -- the hot loop
        panel = L[i+bs:, i:i+bs]
        update = matmul_fn(panel, panel.T)
        A[i+bs:, i+bs:] -= update

    return L


def naive_matmul(a, b):
    return a @ b


def gpu_matmul(a, b):
    if not _HAS_MLX:
        return a @ b
    r = mx.array(a) @ mx.array(b)
    mx.eval(r)
    return np.array(r)


def run_variant(name, A, block_size, matmul_fn, warmups=1, runs=2):
    times = []
    L = None
    for i in range(warmups + runs):
        t0 = time.perf_counter()
        L = blocked_cholesky(A, block_size, matmul_fn)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if i >= warmups:
            times.append(elapsed)

    A_reconstructed = L @ L.T
    residual = float(np.linalg.norm(A - A_reconstructed) / np.linalg.norm(A))

    median_ms = float(np.median(times))
    n = A.shape[0]
    # "factorizations/sec" throughput proxy, matching this project's existing
    # tokens/sec convention for other benchmarks -- here, "elements/sec" of
    # the matrix being factorized.
    elements_per_sec = (n * n) / (median_ms / 1000.0)

    print(f"  {name:8s}: median={median_ms:8.2f}ms  residual={residual:.2e}  "
          f"elements/sec={elements_per_sec:,.0f}")
    return median_ms, residual


def main():
    print("=" * 70)
    print("  Cholesky Decomposition Benchmark — Math Problem")
    print("  (A = L @ L.T, blocked algorithm, trailing-update dispatch varies)")
    print("=" * 70)

    # Sizes deliberately span the crossover found elsewhere in this project's
    # investigation (project_tricompute_ane_findings.md): CPU+GPU concurrent
    # dispatch has fixed thread/MLX overhead that needs several-thousand-row
    # matmuls to amortize. N=2048/4096 sit below that crossover (trailing
    # updates are only ~1-6k rows); N=8192/16384 sit at or above it (trailing
    # updates reach 6k-12k rows), matching the scale where real speedups were
    # measured for the transformer-block benchmarks.
    for n, block_size in [(2048, 512), (4096, 1024), (8192, 2048), (16384, 4096)]:
        print(f"\n▶ N={n}, block_size={block_size}")
        A = make_spd_matrix(n)

        scheduler = fusionml.TriComputeMatmul(enable_ane=False)

        results = {}
        for name, fn in [
            ("naive", naive_matmul),
            ("gpu", gpu_matmul),
            ("smart", scheduler),
        ]:
            median_ms, residual = run_variant(name, A, block_size, fn)
            results[name] = (median_ms, residual)

        naive_ms = results["naive"][0]
        print(f"\n  Speedup vs naive:")
        for name in ("gpu", "smart"):
            ms = results[name][0]
            print(f"    {name}: {naive_ms/ms:.2f}x")

        all_correct = all(r < 1e-3 for _, r in results.values())
        print(f"  All variants numerically correct (residual < 1e-3): {all_correct}")


if __name__ == "__main__":
    main()
