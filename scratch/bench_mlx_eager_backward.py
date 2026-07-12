import mlx.core as mx
import mlx.core.fast as mxf
import mlx.nn as mlx_nn
import time
import numpy as np

def benchmark_backward(fn, inputs, dtype, name):
    # Warmup
    for _ in range(10):
        val, grad = mx.value_and_grad(fn)(*inputs)
        mx.eval(val, grad)
        
    t0 = time.perf_counter()
    for _ in range(50):
        val, grad = mx.value_and_grad(fn)(*inputs)
        mx.eval(val, grad)
    ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    print(f"  {name:<35} ({str(dtype):<15}): {ms:.3f} ms")

print("--------------------------------------------------")
print("1. Softmax Backward (1024 x 1024)")
print("--------------------------------------------------")
for dtype in [mx.float32, mx.float16]:
    np.random.seed(42)
    scores = mx.array(np.random.randn(1024, 1024).astype(np.float32)).astype(dtype)
    def fn_softmax(s):
        # Cast to float32 inside softmax is standard
        return mx.mean(mx.softmax(s.astype(mx.float32), axis=-1).astype(s.dtype))
    benchmark_backward(fn_softmax, [scores], dtype, "Softmax")

print("\n--------------------------------------------------")
print("2. LayerNorm Backward")
print("--------------------------------------------------")
for dtype in [mx.float32, mx.float16]:
    np.random.seed(42)
    x = mx.array(np.random.randn(1024, 4096).astype(np.float32)).astype(dtype)
    g = mx.ones((4096,), dtype=dtype)
    b = mx.zeros((4096,), dtype=dtype)
    def fn_ln(x_in, g_in, b_in):
        return mx.mean(mxf.layer_norm(x_in, g_in, b_in, 1e-5))
    benchmark_backward(fn_ln, [x, g, b], dtype, "LayerNorm (1024x4096)")

print("\n--------------------------------------------------")
print("3. SiLU (Llama GLU) vs GELU (GPT-2)")
print("--------------------------------------------------")
for dtype in [mx.float32, mx.float16]:
    np.random.seed(42)
    g = mx.array(np.random.randn(1024, 14336).astype(np.float32)).astype(dtype)
    up = mx.array(np.random.randn(1024, 14336).astype(np.float32)).astype(dtype)
    def fn_glu(g_in, up_in):
        return mx.mean(mlx_nn.silu(g_in) * up_in)
    benchmark_backward(fn_glu, [g, up], dtype, "Llama GLU (1024x14336)")

for dtype in [mx.float32, mx.float16]:
    np.random.seed(42)
    fc = mx.array(np.random.randn(1024, 6400).astype(np.float32)).astype(dtype)
    def fn_gelu(fc_in):
        return mx.mean(mlx_nn.gelu_approx(fc_in))
    benchmark_backward(fn_gelu, [fc], dtype, "GPT-2 GELU (1024x6400)")
