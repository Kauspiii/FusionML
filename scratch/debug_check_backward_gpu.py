import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

import mlx.core as mx
from fusionml.tensor import Tensor, layer_norm, gelu

# Setup weights
D, FFN = 1600, 6400
L = 1024
x = Tensor(np.random.randn(L, D).astype(np.float32), requires_grad=False).to_gpu()
w_q = Tensor(np.random.randn(D, D).astype(np.float32), requires_grad=True).to_gpu()
w_q.is_parameter = True

h = layer_norm(x, Tensor(np.ones(D, dtype=np.float32), requires_grad=True).to_gpu(), Tensor(np.zeros(D, dtype=np.float32), requires_grad=True).to_gpu())
q = h @ w_q
loss = q.mean()

# Modify backward to print op and use_gpu
import fusionml.autograd.grad as grad_mod

orig_backward = grad_mod.backward

def debug_backward(tensor, grad=None):
    from fusionml.tensor import Tensor
    if grad is None:
        grad = grad_mod._make_grad_tensor(mx.ones(tensor.shape), on_gpu=True)
    topo = []
    visited = set()
    def build_topo(t):
        if id(t) not in visited and t._ctx is not None:
            visited.add(id(t))
            op, *inputs = t._ctx
            for inp in inputs:
                if isinstance(inp, Tensor):
                    build_topo(inp)
            topo.append(t)
    build_topo(tensor)
    tensor.grad = grad
    for t in reversed(topo):
        if t._ctx is None:
            continue
        op, *args = t._ctx
        use_gpu = grad_mod._is_gpu(t) and t.grad is not None and grad_mod._is_gpu(t.grad)
        print(f"OP: {op:12s} | use_gpu: {use_gpu} | t.grad: {t.grad is not None} | is_gpu(t): {grad_mod._is_gpu(t)} | is_gpu(t.grad): {t.grad is not None and grad_mod._is_gpu(t.grad)}")
        
    # Run original backward to check if it completes
    orig_backward(tensor, grad)

loss.backward()
