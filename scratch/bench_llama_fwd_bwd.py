import mlx.core as mx
import mlx.core.fast as mxf
import mlx.nn as mlx_nn
import time
import numpy as np

def benchmark_llama(dtype, compiled, name):
    D, F = 4096, 14336
    L = 1024
    np.random.seed(42); s = 0.02
    
    def _arr(shape):
        return mx.array(np.random.randn(*shape).astype(np.float32) * s).astype(dtype)
        
    w_q   = _arr((D, D));  w_k = _arr((D, D)); w_v = _arr((D, D)); w_o = _arr((D, D))
    w_gate = _arr((D, F)); w_up = _arr((D, F)); w_down = _arr((F, D))
    ln1_g = mx.ones((D,), dtype=dtype); ln1_b = mx.zeros((D,), dtype=dtype)
    ln2_g = mx.ones((D,), dtype=dtype); ln2_b = mx.zeros((D,), dtype=dtype)
    x = mx.array(np.zeros((L, D), dtype=np.float32))
    
    def _fwd(x_in, w_q_in, w_k_in, w_v_in, w_o_in, w_gate_in, w_up_in, w_down_in, ln1_g_in, ln1_b_in, ln2_g_in, ln2_b_in):
        xd = x_in.astype(dtype)
        h  = mxf.layer_norm(xd, ln1_g_in, ln1_b_in, 1e-5)
        q  = h @ w_q_in; k = h @ w_k_in; v = h @ w_v_in
        scale = 1.0 / (D ** 0.5)
        scores = (q @ mx.transpose(k)) * scale
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        h1 = xd + (attn @ v) @ w_o_in
        h2 = mxf.layer_norm(h1, ln2_g_in, ln2_b_in, 1e-5)
        g  = h2 @ w_gate_in
        out = (mlx_nn.silu(g) * (h2 @ w_up_in)) @ w_down_in
        return mx.mean(h1 + out).astype(mx.float32)
        
    grad_fn = mx.value_and_grad(_fwd, argnums=list(range(1, 12)))
    if compiled:
        grad_fn = mx.compile(grad_fn)
        
    # Warmup
    for _ in range(10):
        loss, grads = grad_fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b)
        mx.eval(loss, grads)
        
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        loss, grads = grad_fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b)
        mx.eval(loss, grads)
        times.append((time.perf_counter() - t0) * 1000.0)
        
    mean = np.mean(times)
    std = np.std(times)
    ci95 = 1.96 * std / np.sqrt(len(times))
    print(f"  {name:<25} | Median: {np.median(times):>8.2f} ms | Mean: {mean:>8.2f} ± {ci95:>5.2f} ms | Std: {std:>5.2f} ms")

print("--------------------------------------------------------------------------------")
print("LLaMA-3-8B Cold Training Step Benchmark (n=50)")
print("--------------------------------------------------------------------------------")
benchmark_llama(mx.float16, False, "FP16 Eager (NoCompile)")
print("  Cooldown 10s...")
time.sleep(10.0)
benchmark_llama(mx.float16, True, "FP16 Compiled")
