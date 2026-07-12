import mlx.core as mx
import time

def measure(label, warmup=3, iterations=10, op=None):
    for _ in range(warmup):
        op()
        mx.eval(mx.zeros(1))
    
    start = time.perf_counter()
    for _ in range(iterations):
        op()
        mx.eval(mx.zeros(1))
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    return elapsed

B, S, H, FFN = 4, 1024, 4096, 14336
BS = B * S

for dtype in [mx.float32, mx.float16]:
    wq = mx.random.normal((H, H), dtype=dtype)
    wk = mx.random.normal((H, H), dtype=dtype)
    wv = mx.random.normal((H, H), dtype=dtype)
    wo = mx.random.normal((H, H), dtype=dtype)
    wg = mx.random.normal((H, FFN), dtype=dtype)
    wu = mx.random.normal((H, FFN), dtype=dtype)
    wd = mx.random.normal((FFN, H), dtype=dtype)
    gamma = mx.ones((H,), dtype=dtype)
    beta = mx.zeros((H,), dtype=dtype)
    x = mx.random.normal((BS, H), dtype=dtype)
    mx.eval(wq, wk, wv, wo, wg, wu, wd, gamma, beta, x)

    def decoder_block(x=x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        ln1 = (x - mean) / mx.sqrt(var + 1e-5) * gamma + beta
        
        q = ln1 @ wq
        k = ln1 @ wk
        v = ln1 @ wv
        attn_out = q @ wo
        attn_add = x + attn_out
        
        mean2 = mx.mean(attn_add, axis=-1, keepdims=True)
        var2 = mx.var(attn_add, axis=-1, keepdims=True)
        ln2 = (attn_add - mean2) / mx.sqrt(var2 + 1e-5) * gamma + beta
        
        gate = ln2 @ wg
        up = ln2 @ wu
        act = mx.maximum(gate, 0) * gate
        gated = act * up
        down = gated @ wd
        out = attn_add + down
        mx.eval(out)

    ms = measure("Decoder block", op=decoder_block)
    print(f"MLX {dtype}: {ms:.2f} ms")
