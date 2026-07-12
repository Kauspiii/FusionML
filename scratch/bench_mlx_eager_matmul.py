import mlx.core as mx
import time
import numpy as np

def benchmark_matmul(M, K, N, dtype, name):
    np.random.seed(42)
    a = mx.array(np.random.randn(M, K).astype(np.float32) * 0.02).astype(dtype)
    b = mx.array(np.random.randn(K, N).astype(np.float32) * 0.02).astype(dtype)
    mx.eval(a, b)
    
    # Warmup
    for _ in range(10):
        mx.eval(a @ b)
        
    t0 = time.perf_counter()
    for _ in range(50):
        mx.eval(a @ b)
    ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    print(f"  {name:<25} ({str(dtype):<15}): {ms:.3f} ms")

print("--------------------------------------------------")
print("1. GPT-2 XL Matmul Shapes (Eager)")
print("--------------------------------------------------")
# Attention shapes
benchmark_matmul(1024, 1600, 1600, mx.float32, "Attn (1024x1600x1600)")
benchmark_matmul(1024, 1600, 1600, mx.float16, "Attn (1024x1600x1600)")

# MLP Up
benchmark_matmul(1024, 1600, 6400, mx.float32, "MLP Up (1024x1600x6400)")
benchmark_matmul(1024, 1600, 6400, mx.float16, "MLP Up (1024x1600x6400)")

# MLP Down
benchmark_matmul(1024, 6400, 1600, mx.float32, "MLP Down (1024x6400x1600)")
benchmark_matmul(1024, 6400, 1600, mx.float16, "MLP Down (1024x6400x1600)")

print("\n--------------------------------------------------")
print("2. LLaMA-3-8B Matmul Shapes (Eager)")
print("--------------------------------------------------")
# Attention shapes
benchmark_matmul(1024, 4096, 4096, mx.float32, "Attn (1024x4096x4096)")
benchmark_matmul(1024, 4096, 4096, mx.float16, "Attn (1024x4096x4096)")

# MLP Up
benchmark_matmul(1024, 4096, 14336, mx.float32, "MLP Up (1024x4096x14336)")
benchmark_matmul(1024, 4096, 14336, mx.float16, "MLP Up (1024x4096x14336)")

# MLP Down
benchmark_matmul(1024, 14336, 4096, mx.float32, "MLP Down (1024x14336x4096)")
benchmark_matmul(1024, 14336, 4096, mx.float16, "MLP Down (1024x14336x4096)")
