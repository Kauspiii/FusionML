#!/usr/bin/env python3
"""
minimal_vag_fp16_repro.py — self-contained repro: FP16 transformer-block
loss is finite in a plain forward pass but NaN when the same function is
wrapped in mx.value_and_grad. Suspected mlx 0.32.0 regression (0.30.1 ok).
No dependencies beyond mlx + numpy.
"""

import numpy as np
import mlx.core as mx
import mlx.core.fast as mxf

D, F, L = 4096, 14336, 1024
np.random.seed(42)


def _a(shape, scale=0.02):
    return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(mx.float16)


x = _a((L, D))
w_q, w_k, w_v, w_o = _a((D, D)), _a((D, D)), _a((D, D)), _a((D, D))
w_gate, w_up, w_down = _a((D, F)), _a((D, F)), _a((F, D))
ln1_g = mx.ones((D,), dtype=mx.float16); ln1_b = mx.zeros((D,), dtype=mx.float16)
ln2_g = mx.ones((D,), dtype=mx.float16); ln2_b = mx.zeros((D,), dtype=mx.float16)
mx.eval(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down)


def loss_fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b):
    h = mxf.layer_norm(x, ln1_g, ln1_b, 1e-5)
    q, k, v = h @ w_q, h @ w_k, h @ w_v
    scores = (q @ mx.transpose(k)) * (1.0 / (D ** 0.5))
    attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
    h1 = x + (attn @ v) @ w_o
    h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
    gate = h2 @ w_gate
    out = (gate * mx.sigmoid(gate) * (h2 @ w_up)) @ w_down
    return mx.mean(h1 + out)


args = (x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b)

fwd = loss_fn(*args)
mx.eval(fwd)
print(f"mlx {mx.__version__}")
print(f"  forward-only loss        : {float(fwd.item())}")

loss, grads = mx.value_and_grad(loss_fn, argnums=list(range(1, 12)))(*args)
mx.eval(loss, grads)
print(f"  value_and_grad loss      : {float(loss.item())}")
gn = [float(mx.sqrt(mx.sum(g.astype(mx.float32) ** 2)).item()) for g in grads]
print(f"  grad norms finite        : {all(np.isfinite(gn))}  (norms: {[round(g, 4) for g in gn[:4]]}...)")
print(f"\n  VERDICT: {'REGRESSION REPRODUCED — vag loss NaN, forward finite' if not np.isfinite(float(loss.item())) else 'no repro on this mlx version'}")
