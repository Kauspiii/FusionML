#!/usr/bin/env python3
"""
vag_nan_bisect.py — bisect the M4 FP16 value_and_grad NaN.

minimal_vag_fp16_repro.py (manual silu) is finite on M4, but the benchmark
path (mlx_nn.silu / mlx_nn.gelu_approx) NaNs. Variants differ by ONE element
each; every variant prints forward loss vs value_and_grad loss.

Variants:
  A manual_silu     gate * sigmoid(gate)             (known-finite control)
  B nn_silu         mlx_nn.silu                       (llama benchmark path)
  C nn_gelu         mlx_nn.gelu_approx, GPT-2 shapes  (gpt2 benchmark path)
  D nn_silu_small   mlx_nn.silu at L=64 (does size matter?)
  E exact_import    the actual run_mlx_llama from model_comparison.py
                    (skipped gracefully if fusionml deps unavailable)
"""

import numpy as np
import mlx.core as mx
import mlx.core.fast as mxf
import mlx.nn as mlx_nn

np.random.seed(42)


def _a(shape, scale=0.02):
    return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(mx.float16)


def build(D, F, L, gpt2=False):
    w = {'x': _a((L, D)),
         'w_q': _a((D, D)), 'w_k': _a((D, D)), 'w_v': _a((D, D)), 'w_o': _a((D, D)),
         'ln1_g': mx.ones((D,), dtype=mx.float16), 'ln1_b': mx.zeros((D,), dtype=mx.float16),
         'ln2_g': mx.ones((D,), dtype=mx.float16), 'ln2_b': mx.zeros((D,), dtype=mx.float16)}
    if gpt2:
        w.update({'w_fc1': _a((D, F)), 'w_fc2': _a((F, D))})
    else:
        w.update({'w_gate': _a((D, F)), 'w_up': _a((D, F)), 'w_down': _a((F, D))})
    mx.eval(*w.values())
    return w


def make_loss_fn(D, act, gpt2=False):
    def loss_fn(x, *ws):
        if gpt2:
            w_q, w_k, w_v, w_o, w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b = ws
        else:
            w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b = ws
        h = mxf.layer_norm(x, ln1_g, ln1_b, 1e-5)
        q, k, v = h @ w_q, h @ w_k, h @ w_v
        scores = (q @ mx.transpose(k)) * (1.0 / (D ** 0.5))
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        h1 = x + (attn @ v) @ w_o
        h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        if gpt2:
            out = act(h2 @ w_fc1) @ w_fc2
        else:
            out = (act(h2 @ w_gate) * (h2 @ w_up)) @ w_down
        return mx.mean(h1 + out)
    return loss_fn


def run_variant(name, D, F, L, act, gpt2=False):
    w = build(D, F, L, gpt2)
    keys = (['w_q', 'w_k', 'w_v', 'w_o'] +
            (['w_fc1', 'w_fc2'] if gpt2 else ['w_gate', 'w_up', 'w_down']) +
            ['ln1_g', 'ln1_b', 'ln2_g', 'ln2_b'])
    args = [w['x']] + [w[k] for k in keys]
    loss_fn = make_loss_fn(D, act, gpt2)

    fwd = loss_fn(*args); mx.eval(fwd)
    loss, grads = mx.value_and_grad(loss_fn, argnums=list(range(1, len(args))))(*args)
    mx.eval(loss, grads)
    fv, lv = float(fwd.item()), float(loss.item())
    gfin = all(bool(mx.all(mx.isfinite(g.astype(mx.float32))).item()) for g in grads)
    print(f"  {name:15s} fwd={fv:+.6f}  vag={lv:+.6f}  vag_finite={np.isfinite(lv)}  grads_finite={gfin}")


def main():
    print(f"mlx {mx.__version__}")
    manual_silu = lambda g: g * mx.sigmoid(g)
    run_variant("A manual_silu", 4096, 14336, 1024, manual_silu)
    run_variant("B nn_silu", 4096, 14336, 1024, mlx_nn.silu)
    run_variant("C nn_gelu(gpt2)", 1600, 6400, 1024, mlx_nn.gelu_approx, gpt2=True)
    run_variant("D nn_silu_small", 4096, 14336, 64, mlx_nn.silu)

    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python")))
        from model_comparison import run_mlx_llama
        w = build(4096, 14336, 1024)
        weights = {k: v for k, v in w.items() if k != 'x'}
        loss = run_mlx_llama(w['x'], weights, training=True)
        lv = float(loss.item()) if hasattr(loss, "item") else float(loss)
        print(f"  E exact_import   vag={lv:+.6f}  vag_finite={np.isfinite(lv)}")
    except Exception as e:
        print(f"  E exact_import   SKIPPED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
