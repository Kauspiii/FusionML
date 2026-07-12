import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

import mlx.core as mx
from fusionml.tensor import Tensor, layer_norm, gelu, batch_eval
from fusionml._metal.tri_scheduler import get_scheduler

def run_fusion_gpt2_forward(x, weights):
    h = layer_norm(x, weights['ln1_g'], weights['ln1_b'])
    q = h @ weights['w_q']
    k = h @ weights['w_k']
    v = h @ weights['w_v']
    kT = k.T
    scores = q @ kT
    attended = scores @ v
    projected = attended @ weights['w_o']
    h1 = x + projected

    h2 = layer_norm(h1, weights['ln2_g'], weights['ln2_b'])
    fc1 = h2 @ weights['w_fc1']
    activated = gelu(fc1)
    output = activated @ weights['w_fc2']
    res = h1 + output
    return res

def run_mlx_gpt2_forward(x, weights):
    def _layer_norm(x, gamma, beta, eps=1e-5):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / mx.sqrt(var + eps) + beta

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
    fc1 = h2 @ weights['w_fc1']
    activated = 0.5 * fc1 * (1 + mx.tanh(
        mx.sqrt(mx.array(2.0 / np.pi)) * (fc1 + 0.044715 * fc1 ** 3)
    ))
    output = activated @ weights['w_fc2']
    res = h1 + output
    return res

def main():
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
    print("Calibrating FusionML scheduler...")
    scheduler.calibrate(shapes=shapes, verbose=False)
    scheduler.print_status()

    # Inputs
    L, D, FFN = 1024, 1600, 6400
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

    # FusionML Setup
    x_fml = Tensor(x_np, requires_grad=False).to_gpu()
    weights_fml = {
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
    for w in weights_fml.values():
        w.is_parameter = True

    # MLX Setup
    x_mlx = mx.array(x_np)
    weights_mlx = {
        'w_q': mx.array(w_q_np), 'w_k': mx.array(w_k_np),
        'w_v': mx.array(w_v_np), 'w_o': mx.array(w_o_np),
        'w_fc1': mx.array(w_fc1_np), 'w_fc2': mx.array(w_fc2_np),
        'ln1_g': mx.array(ln1_g_np), 'ln1_b': mx.array(ln1_b_np),
        'ln2_g': mx.array(ln2_g_np), 'ln2_b': mx.array(ln2_b_np),
    }

    # Profile FusionML
    print("\n--- Profiling FusionML (Training) ---")
    fwd_times = []
    bwd_times = []
    for run in range(15):
        # Reset grads
        for w in weights_fml.values():
            w.grad = None
        
        t0 = time.perf_counter()
        res = run_fusion_gpt2_forward(x_fml, weights_fml)
        loss = res.mean()
        # Force lazy graph construction
        # mx.eval(loss._mlx) # we don't eval here to keep it matching benchmark
        t1 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000)
        
        t2 = time.perf_counter()
        loss.backward()
        grads = [w.grad for w in weights_fml.values() if w.grad is not None]
        batch_eval(loss, *grads)
        t3 = time.perf_counter()
        bwd_times.append((t3 - t2) * 1000)
        
    print(f"FusionML Forward - Median: {np.median(fwd_times[5:]):.2f} ms | Mean: {np.mean(fwd_times[5:]):.2f} ms")
    print(f"FusionML Backward - Median: {np.median(bwd_times[5:]):.2f} ms | Mean: {np.mean(bwd_times[5:]):.2f} ms")

    # Profile MLX
    print("\n--- Profiling MLX (Training) ---")
    mlx_fwd_times = []
    mlx_bwd_times = []
    
    def mlx_loss_fn(x, w):
        res = run_mlx_gpt2_forward(x, w)
        return mx.mean(res)
        
    grad_fn = mx.value_and_grad(mlx_loss_fn, argnums=1)
    
    for run in range(15):
        t0 = time.perf_counter()
        # MLX forward is part of value_and_grad usually, but let's measure forward alone
        res = run_mlx_gpt2_forward(x_mlx, weights_mlx)
        t1 = time.perf_counter()
        mlx_fwd_times.append((t1 - t0) * 1000)
        
        t2 = time.perf_counter()
        loss, grads = grad_fn(x_mlx, weights_mlx)
        mx.eval(loss, grads)
        t3 = time.perf_counter()
        mlx_bwd_times.append((t3 - t2) * 1000)
        
    print(f"MLX Forward - Median: {np.median(mlx_fwd_times[5:]):.2f} ms | Mean: {np.mean(mlx_fwd_times[5:]):.2f} ms")
    print(f"MLX Backward (value_and_grad) - Median: {np.median(mlx_bwd_times[5:]):.2f} ms | Mean: {np.mean(mlx_bwd_times[5:]):.2f} ms")

if __name__ == "__main__":
    main()
