import sys
import os
import time
import cProfile
import pstats
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

x = Tensor(x_np, requires_grad=False).to_gpu()
weights = {
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
for w in weights.values():
    w.is_parameter = True

def run_fusion_gpt2_train(x, weights):
    # Zero gradients
    for w in weights.values():
        w.grad = None
        
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
    
    loss = res.mean()
    loss.backward()
    loss.eval()
    for w in weights.values():
        if w.grad is not None:
            w.grad.eval()
    return loss

# Warmup
for _ in range(5):
    run_fusion_gpt2_train(x, weights)

print("Starting profile...")
pr = cProfile.Profile()
pr.enable()
for _ in range(20):
    run_fusion_gpt2_train(x, weights)
pr.disable()

ps = pstats.Stats(pr).sort_stats('cumulative')
ps.print_stats(30)
