import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

import mlx.core as mx
from fusionml.tensor import Tensor, layer_norm, silu, batch_eval
from fusionml._metal.tri_scheduler import get_scheduler

def run_fusion_llama_training(x, weights):
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
    gate = h2 @ weights['w_gate']
    silu_gate = silu(gate)
    up = h2 @ weights['w_up']
    intermediate = silu_gate * up
    output = intermediate @ weights['w_down']
    res = h1 + output
    
    loss = res.mean()
    loss.backward()
    grads = [w.grad for w in weights.values() if w.grad is not None]
    batch_eval(loss, *grads)
    return loss

def main():
    threads = os.environ.get("VECLIB_MAXIMUM_THREADS", "unknown")
    print(f"Running Llama Training profile. Thread count: {threads}")
    
    scheduler = get_scheduler()
    shapes = [
        (1024, 4096, 4096),
        (1024, 4096, 14336),
        (1024, 14336, 4096),
        (1024, 4096, 1024),
        (1024, 1024, 4096),
        (4096, 4096, 1024),
        (14336, 4096, 1024),
        (4096, 14336, 1024)
    ]
    scheduler.calibrate(shapes=shapes, verbose=False)
    scheduler.print_status()

    # Inputs
    L, D, FFN = 1024, 4096, 14336
    scale = 0.02
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

    # FusionML Setup
    x_fml = Tensor(x_np, requires_grad=False).to_gpu()
    weights_fml = {
        'w_q': Tensor(w_q_np, requires_grad=True).to_gpu(),
        'w_k': Tensor(w_k_np, requires_grad=True).to_gpu(),
        'w_v': Tensor(w_v_np, requires_grad=True).to_gpu(),
        'w_o': Tensor(w_o_np, requires_grad=True).to_gpu(),
        'w_gate': Tensor(w_gate_np, requires_grad=True).to_gpu(),
        'w_up': Tensor(w_up_np, requires_grad=True).to_gpu(),
        'w_down': Tensor(w_down_np, requires_grad=True).to_gpu(),
        'ln1_g': Tensor(ln1_g_np, requires_grad=True).to_gpu(),
        'ln1_b': Tensor(ln1_b_np, requires_grad=True).to_gpu(),
        'ln2_g': Tensor(ln2_g_np, requires_grad=True).to_gpu(),
        'ln2_b': Tensor(ln2_b_np, requires_grad=True).to_gpu(),
    }
    for w in weights_fml.values():
        w.is_parameter = True

    # Warmup
    for _ in range(3):
        for w in weights_fml.values():
            w.grad = None
        _ = run_fusion_llama_training(x_fml, weights_fml)
        mx.clear_cache()

    # Time runs
    times = []
    for _ in range(5):
        for w in weights_fml.values():
            w.grad = None
        mx.clear_cache()
        t0 = time.perf_counter()
        _ = run_fusion_llama_training(x_fml, weights_fml)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"FusionML Llama Training - Median: {np.median(times):.2f} ms | Mean: {np.mean(times):.2f} ms")

if __name__ == "__main__":
    main()
