#!/usr/bin/env python3
"""
vag_nan_bisect2.py — round 2. Round 1 (vag_nan_bisect.py) showed on M4:
reconstructed math finite (A-D), imported run_mlx_llama NaN (E). Two
remaining hypotheses, separated here:

  F exact_copy    : literal standalone copy of run_mlx_llama's training
                    branch (varargs loss_fn, same structure) — NO
                    model_comparison/fusionml import.
                    NaN here → the code shape (varargs vag etc.) triggers it.
  G math_after_import : import model_comparison FIRST (module-level
                    fusionml scheduler init), then run round-1's finite
                    variant-B math.
                    NaN here → import side effect corrupts MLX state.
  H exact_import  : round-1 E again as in-process control (expect NaN).
"""

import numpy as np
import mlx.core as mx
import mlx.core.fast as mxf
import mlx.nn as mlx_nn

np.random.seed(42)
D, F_DIM, L = 4096, 14336, 1024


def _a(shape, scale=0.02):
    return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(mx.float16)


def build():
    w = {'x': _a((L, D)),
         'w_q': _a((D, D)), 'w_k': _a((D, D)), 'w_v': _a((D, D)), 'w_o': _a((D, D)),
         'w_gate': _a((D, F_DIM)), 'w_up': _a((D, F_DIM)), 'w_down': _a((F_DIM, D)),
         'ln1_g': mx.ones((D,), dtype=mx.float16), 'ln1_b': mx.zeros((D,), dtype=mx.float16),
         'ln2_g': mx.ones((D,), dtype=mx.float16), 'ln2_b': mx.zeros((D,), dtype=mx.float16)}
    mx.eval(*w.values())
    return w


def report(name, lv):
    print(f"  {name:20s} vag={lv:+.6f}  finite={np.isfinite(lv)}")


# --- F: literal copy of run_mlx_llama training branch, no imports ---------

def exact_copy_training(x, weights):
    def forward_fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down,
                   ln1_g, ln1_b, ln2_g, ln2_b):
        D = x.shape[-1]
        h = mxf.layer_norm(x, ln1_g, ln1_b, 1e-5)
        q = h @ w_q
        k = h @ w_k
        v = h @ w_v
        kT = mx.transpose(k)
        scale = 1.0 / (D ** 0.5)
        scores = (q @ kT) * scale
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        attended = attn @ v
        projected = attended @ w_o
        h1 = x + projected
        h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        gate = h2 @ w_gate
        up = h2 @ w_up
        output = (mlx_nn.silu(gate) * up) @ w_down
        return h1 + output

    def loss_fn(*args):
        res = forward_fn(*args)
        return mx.mean(res)

    grad_fn = mx.value_and_grad(loss_fn, argnums=list(range(1, 12)))
    loss, grads = grad_fn(
        x, weights['w_q'], weights['w_k'], weights['w_v'], weights['w_o'],
        weights['w_gate'], weights['w_up'], weights['w_down'],
        weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b'])
    mx.eval(loss, grads)
    return float(loss.item())


# --- G/H helpers -----------------------------------------------------------

def variant_b_math(w):
    def loss_fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b):
        h = mxf.layer_norm(x, ln1_g, ln1_b, 1e-5)
        q, k, v = h @ w_q, h @ w_k, h @ w_v
        scores = (q @ mx.transpose(k)) * (1.0 / (D ** 0.5))
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        h1 = x + (attn @ v) @ w_o
        h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        out = (mlx_nn.silu(h2 @ w_gate) * (h2 @ w_up)) @ w_down
        return mx.mean(h1 + out)

    keys = ['w_q', 'w_k', 'w_v', 'w_o', 'w_gate', 'w_up', 'w_down',
            'ln1_g', 'ln1_b', 'ln2_g', 'ln2_b']
    args = [w['x']] + [w[k] for k in keys]
    loss, grads = mx.value_and_grad(loss_fn, argnums=list(range(1, 12)))(*args)
    mx.eval(loss, grads)
    return float(loss.item())


def main():
    print(f"mlx {mx.__version__}")

    w = build()
    report("F exact_copy", exact_copy_training(w['x'], w))

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python")))
    try:
        import model_comparison  # module-level fusionml scheduler init
        report("G math_after_import", variant_b_math(build()))
        wh = build()
        weights = {k: v for k, v in wh.items() if k != 'x'}
        loss = model_comparison.run_mlx_llama(wh['x'], weights, training=True)
        report("H exact_import", float(loss.item()) if hasattr(loss, "item") else float(loss))
    except Exception as e:
        print(f"  G/H SKIPPED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
