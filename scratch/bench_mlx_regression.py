import mlx.core as mx
import mlx.core.fast as mxf
import mlx.nn as mlx_nn
import time
import numpy as np

DIM = 4096
FFN_DIM = 14336
BATCH = 8
SEQ = 2048
GEN_STEPS = 50

def run_benchmark(compiled, name):
    np.random.seed(42); s = 0.02
    w = {
        'w_q':    mx.array(np.random.randn(DIM, DIM).astype(np.float32) * s).astype(mx.float16),
        'w_k':    mx.array(np.random.randn(DIM, DIM).astype(np.float32) * s).astype(mx.float16),
        'w_v':    mx.array(np.random.randn(DIM, DIM).astype(np.float32) * s).astype(mx.float16),
        'w_o':    mx.array(np.random.randn(DIM, DIM).astype(np.float32) * s).astype(mx.float16),
        'w_gate': mx.array(np.random.randn(DIM, FFN_DIM).astype(np.float32) * s).astype(mx.float16),
        'w_up':   mx.array(np.random.randn(DIM, FFN_DIM).astype(np.float32) * s).astype(mx.float16),
        'w_down': mx.array(np.random.randn(FFN_DIM, DIM).astype(np.float32) * s).astype(mx.float16),
        'ln1_g':  mx.ones((DIM,), dtype=mx.float16),
        'ln1_b':  mx.zeros((DIM,), dtype=mx.float16),
        'ln2_g':  mx.ones((DIM,), dtype=mx.float16),
        'ln2_b':  mx.zeros((DIM,), dtype=mx.float16),
    }
    mx.eval(*w.values())

    def _fwd(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b):
        x16 = x.astype(mx.float16)
        h   = mxf.layer_norm(x16, ln1_g, ln1_b, 1e-5)
        q   = h @ w_q; k = h @ w_k; v = h @ w_v
        h1  = x16 + (q @ mx.transpose(k) @ v) @ w_o
        h2  = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        gate = h2 @ w_gate
        out  = (mlx_nn.silu(gate) * (h2 @ w_up)) @ w_down
        return (h1 + out).astype(mx.float32)

    fwd_fn = mx.compile(_fwd) if compiled else _fwd

    # Warmup prefill
    mx.eval(fwd_fn(mx.zeros((BATCH * SEQ, DIM)), w['w_q'], w['w_k'], w['w_v'], w['w_o'],
                   w['w_gate'], w['w_up'], w['w_down'], w['ln1_g'], w['ln1_b'], w['ln2_g'], w['ln2_b']))
    # Warmup decode
    x_dec = mx.zeros((BATCH, DIM))
    for _ in range(5):
        mx.eval(fwd_fn(x_dec, w['w_q'], w['w_k'], w['w_v'], w['w_o'],
                       w['w_gate'], w['w_up'], w['w_down'], w['ln1_g'], w['ln1_b'], w['ln2_g'], w['ln2_b']))

    # Timed run
    t0 = time.perf_counter()
    for _ in range(GEN_STEPS):
        mx.eval(fwd_fn(x_dec, w['w_q'], w['w_k'], w['w_v'], w['w_o'],
                       w['w_gate'], w['w_up'], w['w_down'], w['ln1_g'], w['ln1_b'], w['ln2_g'], w['ln2_b']))
    tps = BATCH * GEN_STEPS / (time.perf_counter() - t0)
    print(f"  {name:<25} : {tps:.2f} tok/s")

print("--------------------------------------------------")
print("MLX Compiled Decode Throughput Verification (batch=8, seq=2048)")
print("--------------------------------------------------")
for i in range(5):
    run_benchmark(True, f"Run {i+1}")
