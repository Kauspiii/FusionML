import mlx.core as mx
import mlx.core.fast as mxf
import mlx.nn as mlx_nn
import numpy as np
import time

# Config
D = 4096
FFN = 14336
L = 1024
scale = 0.02

# Data
np.random.seed(42)
x_np = np.random.randn(L, D).astype(np.float32) * scale
w_q_np = np.random.randn(D, D).astype(np.float32) * scale
w_k_np = np.random.randn(D, D).astype(np.float32) * scale
w_v_np = np.random.randn(D, D).astype(np.float32) * scale
w_o_np = np.random.randn(D, D).astype(np.float32) * scale
w_gate_np = np.random.randn(D, FFN).astype(np.float32) * scale
w_up_np = np.random.randn(D, FFN).astype(np.float32) * scale
w_down_np = np.random.randn(FFN, D).astype(np.float32) * scale
ln1_g_np = np.ones(D, dtype=np.float32)
ln1_b_np = np.zeros(D, dtype=np.float32)
ln2_g_np = np.ones(D, dtype=np.float32)
ln2_b_np = np.zeros(D, dtype=np.float32)

x = mx.array(x_np)
weights = {
    'w_q': mx.array(w_q_np), 'w_k': mx.array(w_k_np), 'w_v': mx.array(w_v_np),
    'w_o': mx.array(w_o_np), 'w_gate': mx.array(w_gate_np), 'w_up': mx.array(w_up_np),
    'w_down': mx.array(w_down_np),
    'ln1_g': mx.array(ln1_g_np), 'ln1_b': mx.array(ln1_b_np),
    'ln2_g': mx.array(ln2_g_np), 'ln2_b': mx.array(ln2_b_np),
}

w_qkv = mx.concatenate([weights['w_q'], weights['w_k'], weights['w_v']], axis=-1)
mx.eval(w_qkv)

def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / mx.sqrt(var + eps) + beta

# 1. MLX Eager
def run_mlx_eager(x, weights):
    h = _layer_norm(x, weights['ln1_g'], weights['ln1_b'])
    q = h @ weights['w_q']
    k = h @ weights['w_k']
    v = h @ weights['w_v']
    kT = mx.transpose(k)
    scores = q @ kT
    attended = scores @ v
    projected = attended @ weights['w_o']
    h1 = x + projected

    h2 = _layer_norm(h1, weights['ln2_g'], weights['ln2_b'])
    gate = h2 @ weights['w_gate']
    silu_gate = gate * mx.sigmoid(gate)
    up = h2 @ weights['w_up']
    intermediate = silu_gate * up
    output = intermediate @ weights['w_down']
    return h1 + output

# 2. FusionML Compiled
def _forward_compiled(x_mlx, w_qkv, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b):
    h = mxf.layer_norm(x_mlx, ln1_g, ln1_b, 1e-5)
    qkv = h @ w_qkv
    qkv_3d = mx.reshape(qkv, (-1, 3, 4096))
    q = qkv_3d[:, 0, :]
    k = qkv_3d[:, 1, :]
    v = qkv_3d[:, 2, :]
    
    kT = mx.transpose(k)
    scores = q @ kT
    attended = scores @ v
    projected = attended @ w_o
    h1 = x_mlx + projected
    
    h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
    gate = h2 @ w_gate
    up = h2 @ w_up
    silu_gate = mlx_nn.silu(gate)
    intermediate = silu_gate * up
    output = intermediate @ w_down
    return h1 + output

run_fusion_compiled = mx.compile(_forward_compiled)

# 3. FusionML Eager (Standard Separate + Fast LN)
def run_fusion_eager_sep(x, weights):
    h = mxf.layer_norm(x, weights['ln1_g'], weights['ln1_b'], 1e-5)
    q = h @ weights['w_q']
    k = h @ weights['w_k']
    v = h @ weights['w_v']
    kT = mx.transpose(k)
    scores = q @ kT
    attended = scores @ v
    projected = attended @ weights['w_o']
    h1 = x + projected

    h2 = mxf.layer_norm(h1, weights['ln2_g'], weights['ln2_b'], 1e-5)
    gate = h2 @ weights['w_gate']
    silu_gate = mlx_nn.silu(gate)
    up = h2 @ weights['w_up']
    intermediate = silu_gate * up
    output = intermediate @ weights['w_down']
    return h1 + output

# 4. FusionML Compiled (Standard Separate + Fast LN)
run_fusion_compiled_sep = mx.compile(run_fusion_eager_sep)

# Warmup
for _ in range(10):
    mx.eval(run_mlx_eager(x, weights))
    mx.eval(run_fusion_compiled(x, w_qkv, weights['w_o'], weights['w_gate'], weights['w_up'], weights['w_down'], weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']))
    mx.eval(run_fusion_eager_sep(x, weights))
    mx.eval(run_fusion_compiled_sep(x, weights))

runs = 50

# MLX Eager
t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(run_mlx_eager(x, weights))
print(f"MLX Eager (Separate + Manual LN): {(time.perf_counter() - t0)*1000/runs:.2f} ms")

# FusionML Compiled (Concat + Fast LN)
t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(run_fusion_compiled(x, w_qkv, weights['w_o'], weights['w_gate'], weights['w_up'], weights['w_down'], weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']))
print(f"FusionML Compiled (Concat + Fast LN): {(time.perf_counter() - t0)*1000/runs:.2f} ms")

# FusionML Eager (Separate + Fast LN)
t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(run_fusion_eager_sep(x, weights))
print(f"FusionML Eager (Separate + Fast LN): {(time.perf_counter() - t0)*1000/runs:.2f} ms")

# FusionML Compiled (Separate + Fast LN)
t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(run_fusion_compiled_sep(x, weights))
print(f"FusionML Compiled (Separate + Fast LN): {(time.perf_counter() - t0)*1000/runs:.2f} ms")
