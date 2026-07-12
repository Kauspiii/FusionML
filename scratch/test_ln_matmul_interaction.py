import mlx.core as mx
import mlx.core.fast as mxf
import numpy as np
import time

L, D = 1024, 4096
np.random.seed(42)
x_np = np.random.randn(L, D).astype(np.float32)
g_np = np.ones(D, dtype=np.float32)
b_np = np.zeros(D, dtype=np.float32)
w_np = np.random.randn(D, D).astype(np.float32) * 0.02

x = mx.array(x_np)
g = mx.array(g_np)
b = mx.array(b_np)
w = mx.array(w_np)

def step_fast(x, g, b, w):
    h = mxf.layer_norm(x, g, b, 1e-5)
    return h @ w

def step_manual(x, g, b, w):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    h = g * (x - mean) / mx.sqrt(var + 1e-5) + b
    return h @ w

fast_compiled = mx.compile(step_fast)
manual_compiled = mx.compile(step_manual)

# Warmup
for _ in range(20):
    mx.eval(fast_compiled(x, g, b, w))
    mx.eval(manual_compiled(x, g, b, w))

runs = 100

t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(fast_compiled(x, g, b, w))
print(f"Fast LN + Matmul (Compiled): {(time.perf_counter() - t0)*1000/runs:.3f} ms")

t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(manual_compiled(x, g, b, w))
print(f"Manual LN + Matmul (Compiled): {(time.perf_counter() - t0)*1000/runs:.3f} ms")

t0 = time.perf_counter()
for _ in range(runs):
    h = mxf.layer_norm(x, g, b, 1e-5)
    res = h @ w
    mx.eval(res)
print(f"Fast LN + Matmul (Eager): {(time.perf_counter() - t0)*1000/runs:.3f} ms")

t0 = time.perf_counter()
for _ in range(runs):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    h = g * (x - mean) / mx.sqrt(var + 1e-5) + b
    res = h @ w
    mx.eval(res)
print(f"Manual LN + Matmul (Eager): {(time.perf_counter() - t0)*1000/runs:.3f} ms")
