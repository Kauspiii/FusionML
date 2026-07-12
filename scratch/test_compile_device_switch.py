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

def pure_gpu_matmul(a, b):
    return a @ b

split_compiled = mx.compile(split_matmul)
pure_compiled = mx.compile(pure_gpu_matmul)

# Warmup
for _ in range(20):
    mx.eval(split_compiled(x, w))
    mx.eval(pure_compiled(x, w))

runs = 100

t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(split_compiled(x, w))
split_time = (time.perf_counter() - t0)*1000/runs
print(f"Compiled Split Matmul Time: {split_time:.2f} ms")

t0 = time.perf_counter()
for _ in range(runs):
    mx.eval(pure_compiled(x, w))
pure_time = (time.perf_counter() - t0)*1000/runs
print(f"Compiled Pure GPU Matmul Time: {pure_time:.2f} ms")

print(f"Compiled Split Speedup: {pure_time/split_time:.2f}x")
