import mlx.core as mx
import mlx.core.fast as mxf
import numpy as np
import time

L, D = 1024, 4096
np.random.seed(42)
x_np = np.random.randn(L, D).astype(np.float32)
g_np = np.ones(D, dtype=np.float32)
b_np = np.zeros(D, dtype=np.float32)

x = mx.array(x_np)
g = mx.array(g_np)
b = mx.array(b_np)

def ln_fast(x, g, b):
    return mxf.layer_norm(x, g, b, 1e-5)

def ln_div(x, g, b):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / mx.sqrt(var + 1e-5) + b

def ln_rsqrt(x, g, b):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return g * (x - mean) * mx.rsqrt(var + 1e-5) + b

fast_compiled = mx.compile(ln_fast)
div_compiled = mx.compile(ln_div)
rsqrt_compiled = mx.compile(ln_rsqrt)

# Warmup
for _ in range(20):
    mx.eval(fast_compiled(x, g, b))
    mx.eval(div_compiled(x, g, b))
    mx.eval(rsqrt_compiled(x, g, b))

runs = 100

t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(fast_compiled(x, g, b))
print(f"mlx.core.fast.layer_norm: {(time.perf_counter() - t0)*1000/runs:.3f} ms")

t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(div_compiled(x, g, b))
print(f"Manual (division): {(time.perf_counter() - t0)*1000/runs:.3f} ms")

t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(rsqrt_compiled(x, g, b))
print(f"Manual (rsqrt): {(time.perf_counter() - t0)*1000/runs:.3f} ms")
