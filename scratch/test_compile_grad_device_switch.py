import mlx.core as mx
import numpy as np
import time

M, K, N = 1024, 4096, 4096
np.random.seed(42)
x_np = np.random.randn(M, K).astype(np.float32)
w_np = np.random.randn(K, N).astype(np.float32) * 0.02

x = mx.array(x_np)
w = mx.array(w_np)

def split_matmul(a, b):
    cpu_rows = int(a.shape[0] * 0.25)
    a_cpu = a[:cpu_rows]
    a_gpu = a[cpu_rows:]
    
    c_gpu = a_gpu @ b
    
    mx.set_default_device(mx.cpu)
    c_cpu = a_cpu @ b
    mx.set_default_device(mx.gpu)
    
    return mx.concatenate([c_cpu, c_gpu], axis=0)

def loss_fn(a, b):
    c = split_matmul(a, b)
    return mx.mean(c)

grad_fn = mx.compile(mx.value_and_grad(loss_fn, argnums=1))

# Warmup
print("Testing compiled gradients of split matmul...")
try:
    loss, grad = grad_fn(x, w)
    mx.eval(loss, grad)
    print("Success: Differentiating split matmul with device switch works!")
    print("Loss shape:", loss.shape, "Grad shape:", grad.shape)
    
    # Timing
    t0 = time.perf_counter()
    for _ in range(20):
        loss, grad = grad_fn(x, w)
        mx.eval(loss, grad)
    print(f"Compiled Split Grad Step Time: {(time.perf_counter() - t0)*1000/20:.2f} ms")
    
except Exception as e:
    print("Error during compiled split matmul grad execution:", e)
