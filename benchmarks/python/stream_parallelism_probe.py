#!/usr/bin/env python3
"""
stream_parallelism_probe.py — does MLX actually run cpu/gpu streams
concurrently when the CPU op's input is produced by a GPU op in the SAME
eval graph? Isolates why chained per-layer splits lose while isolated
splits win 1.3x.

Cases (all 2048x1600 @ 1600x6400, fp16, ratio=0.35, n=20 medians):
  A gpu_only_ready      : input pre-evaluated, plain GPU matmul
  B split_ready         : input pre-evaluated, split matmul (calibration case)
  C gpu_only_dep        : input = relu(x) on GPU in same graph, GPU matmul
  D split_dep           : input = relu(x) on GPU in same graph, split matmul
  E split_eager_boundary: relu evaluated first (separate eval), then split
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))
import mlx.core as mx

M, K, N = 2048, 1600, 6400
RATIO = 0.35
RUNS = 20

np.random.seed(0)
x = mx.array((np.random.randn(M, K) * 0.02).astype(np.float32)).astype(mx.float16)
w = mx.array((np.random.randn(K, N) * 0.02).astype(np.float32)).astype(mx.float16)
mx.eval(x, w)
cpu_rows = int(M * RATIO)


def split_mm(h):
    c_cpu = mx.matmul(h[:cpu_rows], w, stream=mx.cpu)
    c_gpu = mx.matmul(h[cpu_rows:], w, stream=mx.gpu)
    return mx.concatenate([c_cpu, c_gpu], axis=0)


def bench(fn):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


cases = {}
cases["A gpu_only_ready"] = bench(lambda: mx.eval(x @ w))
cases["B split_ready"] = bench(lambda: mx.eval(split_mm(x)))


def c_case():
    h = mx.maximum(x, 0)
    mx.eval(h @ w)
cases["C gpu_only_dep"] = bench(c_case)


def d_case():
    h = mx.maximum(x, 0)
    mx.eval(split_mm(h))
cases["D split_dep"] = bench(d_case)


def e_case():
    h = mx.maximum(x, 0)
    mx.eval(h)                    # eager boundary — input materialized
    mx.eval(split_mm(h))
cases["E split_eager_boundary"] = bench(e_case)

for k, v in cases.items():
    print(f"  {k:24s}: {v:7.2f} ms")
print(f"\n  B vs A: {cases['A gpu_only_ready']/cases['B split_ready']:.3f}x")
print(f"  D vs C: {cases['C gpu_only_dep']/cases['D split_dep']:.3f}x")
print(f"  E vs C: {cases['C gpu_only_dep']/cases['E split_eager_boundary']:.3f}x")
