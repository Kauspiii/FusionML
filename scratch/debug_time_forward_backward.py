import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

import mlx.core as mx
from fusionml.tensor import Tensor, layer_norm, gelu
from fusionml._metal.tri_scheduler import get_scheduler

# Setup GPT-2 Config
D, FFN = 1600, 6400
L = 1024
scale = 0.02

np.random.seed(123)
x_np = np.random.randn(L, D).astype(np.float32) * scale
w_q_np = np.random.randn(D, D).astype(np.float32) * scale
w_k_np = np.random.randn(D, D).astype(np.float32) * scale
w_v_np = np.random.randn(D, D).astype(np.float32) * scale
w_o_np = np.random.randn(D, D).astype(np.float32) * scale
w_fc1_np = np.random.randn(D, FFN).astype(np.float32) * scale
w_fc2_np = np.random.randn(FFN, D).astype(np.float32) * scale
ln1_g_np = np.ones(D, dtype=np.float32)
ln1_b_np = np.zeros(D, dtype=np.float32)
ln2_g_np = np.ones(D, dtype=np.float32)
ln2_b_np = np.zeros(D, dtype=np.float32)

# --- MLX Setup ---
x_mx = mx.array(x_np)
weights_mlx = {
    'w_q': mx.array(w_q_np), 'w_k': mx.array(w_k_np), 'w_v': mx.array(w_v_np),
    'w_o': mx.array(w_o_np), 'w_fc1': mx.array(w_fc1_np), 'w_fc2': mx.array(w_fc2_np),
    'ln1_g': mx.array(ln1_g_np), 'ln1_b': mx.array(ln1_b_np),
    'ln2_g': mx.array(ln2_g_np), 'ln2_b': mx.array(ln2_b_np),
}

def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / mx.sqrt(var + eps) + beta

# --- FusionML Setup ---
scheduler = get_scheduler()
shapes = [
    (1024, 1600, 1600),
    (1024, 1600, 4800),
    (1024, 1600, 6400),
    (1024, 6400, 1600),
    (1024, 1600, 1024),
    (1024, 1024, 1600),
    (1600, 1600, 1024),
    (6400, 1600, 1024),
    (1600, 6400, 1024)
]
scheduler.calibrate(shapes=shapes, verbose=False)

x_fs = Tensor(x_np, requires_grad=False).to_gpu()
weights_fs = {
    'w_q': Tensor(w_q_np, requires_grad=True).to_gpu(),
    'w_k': Tensor(w_k_np, requires_grad=True).to_gpu(),
    'w_v': Tensor(w_v_np, requires_grad=True).to_gpu(),
    'w_o': Tensor(w_o_np, requires_grad=True).to_gpu(),
    'w_fc1': Tensor(w_fc1_np, requires_grad=True).to_gpu(),
    'w_fc2': Tensor(w_fc2_np, requires_grad=True).to_gpu(),
    'ln1_g': Tensor(ln1_g_np, requires_grad=True).to_gpu(),
    'ln1_b': Tensor(ln1_b_np, requires_grad=True).to_gpu(),
    'ln2_g': Tensor(ln2_g_np, requires_grad=True).to_gpu(),
    'ln2_b': Tensor(ln2_b_np, requires_grad=True).to_gpu(),
}
for w in weights_fs.values():
    w.is_parameter = True

# 1. Timing MLX Forward and Backward
times_mlx_f = []
times_mlx_b = []
for _ in range(25):
    # MLX Value and Grad
    def loss_fn(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b):
        h = _layer_norm(x, ln1_g, ln1_b)
        q = h @ w_q
        k = h @ w_k
        v = h @ w_v
        kT = mx.transpose(k)
        scores = q @ kT
        attended = scores @ v
        projected = attended @ w_o
        h1 = x + projected
        h2 = _layer_norm(h1, ln2_g, ln2_b)
        fc1 = h2 @ w_fc1
        activated = 0.5 * fc1 * (1 + mx.tanh(
            0.7978845608028654 * (fc1 + 0.044715 * fc1 ** 3)
        ))
        output = activated @ w_fc2
        res = h1 + output
        return mx.mean(res)

    grad_fn = mx.value_and_grad(loss_fn, argnums=list(range(1, 11)))
    
    # Time Forward
    t0 = time.perf_counter()
    loss, grads = grad_fn(
        x_mx, weights_mlx['w_q'], weights_mlx['w_k'], weights_mlx['w_v'], weights_mlx['w_o'],
        weights_mlx['w_fc1'], weights_mlx['w_fc2'],
        weights_mlx['ln1_g'], weights_mlx['ln1_b'], weights_mlx['ln2_g'], weights_mlx['ln2_b']
    )
    mx.eval(loss) # Eval loss first to force forward
    times_mlx_f.append((time.perf_counter() - t0) * 1000)
    
    # Time Backward
    t0 = time.perf_counter()
    mx.eval(grads)
    times_mlx_b.append((time.perf_counter() - t0) * 1000)

print(f"MLX Forward: {np.median(times_mlx_f[5:]):.3f} ms")
print(f"MLX Backward: {np.median(times_mlx_b[5:]):.3f} ms")


# 2. Timing FusionML Forward and Backward
times_fs_f = []
times_fs_b = []
for _ in range(25):
    for w in weights_fs.values():
        w.grad = None
        
    # Time Forward
    t0 = time.perf_counter()
    h = layer_norm(x_fs, weights_fs['ln1_g'], weights_fs['ln1_b'])
    q = h @ weights_fs['w_q']
    k = h @ weights_fs['w_k']
    v = h @ weights_fs['w_v']
    kT = k.T
    scores = q @ kT
    attended = scores @ v
    projected = attended @ weights_fs['w_o']
    h1 = x_fs + projected
    
    h2 = layer_norm(h1, weights_fs['ln2_g'], weights_fs['ln2_b'])
    fc1 = h2 @ weights_fs['w_fc1']
    activated = gelu(fc1)
    output = activated @ weights_fs['w_fc2']
    res = h1 + output
    loss = res.mean()
    loss.eval() # Force forward eval
    times_fs_f.append((time.perf_counter() - t0) * 1000)
    
    # Time Backward
    t0 = time.perf_counter()
    loss.backward()
    for w in weights_fs.values():
        if w.grad is not None:
            w.grad.eval()
    times_fs_b.append((time.perf_counter() - t0) * 1000)

print(f"FusionML Forward: {np.median(times_fs_f[5:]):.3f} ms")
print(f"FusionML Backward: {np.median(times_fs_b[5:]):.3f} ms")
