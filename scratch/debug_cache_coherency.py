import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

import mlx.core as mx
from fusionml._metal.tri_scheduler import get_scheduler

# Matrix dimensions (GPT-2 XL QKV shape)
M, K, N = 1024, 1600, 4800
gpu_pct = 75

a_mx = mx.random.normal((M, K))
b_mx = mx.random.normal((K, N))
mx.eval(a_mx, b_mx)

b_np = np.array(b_mx)
scheduler = get_scheduler()

gpu_rows = int(M * gpu_pct / 100)
cpu_rows = M - gpu_rows
a_cpu = a_mx[:cpu_rows]
a_gpu = a_mx[cpu_rows:]

# Function with copy
def run_with_copy():
    mx.eval(a_mx)
    a_cpu_np = np.array(a_cpu)
    c_gpu = a_gpu @ b_mx
    future_cpu = scheduler._pool.submit(np.matmul, a_cpu_np, b_np)
    mx.eval(c_gpu)
    c_cpu_np = future_cpu.result()
    c_cpu = mx.array(c_cpu_np)
    res = mx.concatenate([c_cpu, c_gpu], axis=0)
    mx.eval(res)
    return res

# Function with zero copy
def run_zero_copy():
    mx.eval(a_mx)
    a_cpu_np = np.array(a_cpu, copy=False)
    c_gpu = a_gpu @ b_mx
    future_cpu = scheduler._pool.submit(np.matmul, a_cpu_np, b_np)
    mx.eval(c_gpu)
    c_cpu_np = future_cpu.result()
    c_cpu = mx.array(c_cpu_np)
    res = mx.concatenate([c_cpu, c_gpu], axis=0)
    mx.eval(res)
    return res

# Pure GPU
def run_pure_gpu():
    mx.eval(a_mx)
    res = a_mx @ b_mx
    mx.eval(res)
    return res

# Warmup
for _ in range(10):
    run_with_copy()
    run_zero_copy()
    run_pure_gpu()

# Measure run_with_copy
times = []
for _ in range(50):
    t0 = time.perf_counter()
    run_with_copy()
    times.append((time.perf_counter() - t0) * 1000)
print(f"With Copy: {np.median(times):.3f} ms")

# Measure run_zero_copy
times = []
for _ in range(50):
    t0 = time.perf_counter()
    run_zero_copy()
    times.append((time.perf_counter() - t0) * 1000)
print(f"Zero Copy: {np.median(times):.3f} ms")

# Measure run_pure_gpu
times = []
for _ in range(50):
    t0 = time.perf_counter()
    run_pure_gpu()
    times.append((time.perf_counter() - t0) * 1000)
print(f"Pure GPU: {np.median(times):.3f} ms")
