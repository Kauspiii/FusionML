#!/usr/bin/env python3
"""
split_point_diagnostic.py — why does per-layer stream splitting win 1.3x on
isolated matmuls but lose 0.92x at block level?

Times the GPT-2 block forward with splitting enabled at different subsets of
Linear layers, plus a pure-matmul-chain control (no attention, no layernorm)
to separate "dependency structure" from "stream overhead".
n=20, in-process (single diagnostic, one config at a time).
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

import mlx.core as mx
import mlx.core.fast as mxf
import mlx.nn as mlx_nn

L, D, F = 2048, 1600, 6400
RATIO = 0.35
RUNS = 20


def make_linear(split_set):
    def linear(h, w, tag):
        if tag not in split_set:
            return h @ w
        cpu_rows = int(h.shape[0] * RATIO)
        c_cpu = mx.matmul(h[:cpu_rows], w, stream=mx.cpu)
        c_gpu = mx.matmul(h[cpu_rows:], w, stream=mx.gpu)
        return mx.concatenate([c_cpu, c_gpu], axis=0)
    return linear


def block_forward(x, weights, linear):
    h = mxf.layer_norm(x, weights['ln1_g'], weights['ln1_b'], 1e-5)
    q = linear(h, weights['w_q'], 'q')
    k = linear(h, weights['w_k'], 'k')
    v = linear(h, weights['w_v'], 'v')
    scale = 1.0 / (D ** 0.5)
    scores = (q @ mx.transpose(k)) * scale
    attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
    attended = attn @ v
    projected = linear(attended, weights['w_o'], 'o')
    h1 = x + projected
    h2 = mxf.layer_norm(h1, weights['ln2_g'], weights['ln2_b'], 1e-5)
    fc1 = linear(h2, weights['w_fc1'], 'fc1')
    act = mlx_nn.gelu_approx(fc1)
    output = linear(act, weights['w_fc2'], 'fc2')
    return h1 + output


def matmul_chain(x, weights, linear):
    """Control: 6 chained matmuls, no attention/norm — pure Linear pipeline."""
    h = linear(x, weights['w_q'], 'q')
    h = linear(h, weights['w_k'], 'k')
    h = linear(h, weights['w_v'], 'v')
    h = linear(h, weights['w_o'], 'o')
    h = linear(h, weights['w_fc1'], 'fc1')
    h = linear(h, weights['w_fc2'], 'fc2')
    return h


def bench(fn):
    for _ in range(5):
        mx.eval(fn())
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def main():
    np.random.seed(42); scale = 0.02

    def _a(shape):
        return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(mx.float16)

    x = _a((L, D))
    weights = {
        'w_q': _a((D, D)), 'w_k': _a((D, D)), 'w_v': _a((D, D)), 'w_o': _a((D, D)),
        'w_fc1': _a((D, F)), 'w_fc2': _a((F, D)),
        'ln1_g': mx.ones((D,), dtype=mx.float16), 'ln1_b': mx.zeros((D,), dtype=mx.float16),
        'ln2_g': mx.ones((D,), dtype=mx.float16), 'ln2_b': mx.zeros((D,), dtype=mx.float16),
    }
    mx.eval(x, *weights.values())

    variants = {
        "none":      set(),
        "fc_only":   {'fc1', 'fc2'},
        "fc1_only":  {'fc1'},
        "qkv_only":  {'q', 'k', 'v'},
        "all":       {'q', 'k', 'v', 'o', 'fc1', 'fc2'},
    }

    print(f"GPT-2 block, L={L}, ratio={RATIO}, n={RUNS} (median ms)")
    results = {}
    base = None
    for name, s in variants.items():
        lin = make_linear(s)
        t = bench(lambda: block_forward(x, weights, lin))
        results[f"block_{name}"] = t
        if name == "none":
            base = t
        print(f"  block  {name:9s}: {t:8.2f} ms   vs none: {base/t:.3f}x")

    chain_base = None
    for name in ["none", "all"]:
        lin = make_linear(variants[name])
        t = bench(lambda: matmul_chain(x, weights, lin))
        results[f"chain_{name}"] = t
        if name == "none":
            chain_base = t
        print(f"  chain  {name:9s}: {t:8.2f} ms   vs none: {chain_base/t:.3f}x")

    print(json.dumps(results))


if __name__ == "__main__":
    main()
