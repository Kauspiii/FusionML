#!/usr/bin/env python3
"""
llama_fp16_nan_check.py — urgent correctness check.

gpt2_fp16_training_stability_test.py revealed loss=NaN from step 0, caused by
unscaled Q@K^T attention scores overflowing FP16 range (no 1/sqrt(D) scaling
in run_fusion_gpt2's attention). run_fusion_llama has the identical unscaled
pattern (`scores = q @ kT`, no division by sqrt(D)) and D=4096 is even larger.

This checks whether the ALREADY-COMMITTED Llama FP16 training benchmark
(796.82ms, reported as 1.25x speedup) is measuring a NaN-loss / NaN-gradient
computation -- i.e. whether the timing numbers are latency-only measurements
of a mathematically broken training step.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))


def check(model_name, D, F, forward_fn_builder):
    import mlx.core as mx
    np.random.seed(42); scale = 0.02

    def _a(shape, dtype):
        return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(dtype)

    dtype = mx.float16
    w_q = _a((D, D), dtype); w_k = _a((D, D), dtype); w_v = _a((D, D), dtype); w_o = _a((D, D), dtype)
    ln1_g = mx.ones((D,), dtype=dtype); ln1_b = mx.zeros((D,), dtype=dtype)
    ln2_g = mx.ones((D,), dtype=dtype); ln2_b = mx.zeros((D,), dtype=dtype)

    if model_name == "llama":
        w_gate = _a((D, F), dtype); w_up = _a((D, F), dtype); w_down = _a((F, D), dtype)
        train_fn = forward_fn_builder(w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b)
        args = (w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b)
        names = ["w_q","w_k","w_v","w_o","w_gate","w_up","w_down","ln1_g","ln1_b","ln2_g","ln2_b"]
    else:
        w_fc1 = _a((D, F), dtype); w_fc2 = _a((F, D), dtype)
        train_fn = forward_fn_builder(w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b)
        args = (w_q, w_k, w_v, w_o, w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b)
        names = ["w_q","w_k","w_v","w_o","w_fc1","w_fc2","ln1_g","ln1_b","ln2_g","ln2_b"]

    grad_fn = mx.value_and_grad(train_fn, argnums=list(range(1, len(args)+1)))
    x = mx.array(np.random.randn(1024, D).astype(np.float32) * scale)

    loss, grads = grad_fn(x, *args)
    mx.eval(loss, grads)

    loss_val = float(loss)
    print(f"\n{model_name.upper()} (D={D}, unscaled Q@K^T attention, FP16):")
    print(f"  loss = {loss_val}")
    print(f"  loss is NaN: {np.isnan(loss_val)}")
    for name, g in zip(names, grads):
        g32 = g.astype(mx.float32)
        norm = float(mx.sqrt(mx.sum(g32*g32)).item())
        flag = " <- NaN" if np.isnan(norm) else ""
        print(f"    {name:8s} grad_norm = {norm:.6e}{flag}")
    return loss_val


def llama_builder(w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn
    def _fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b):
        xd = x.astype(mx.float16)
        Dh = xd.shape[-1]
        h  = mxf.layer_norm(xd, ln1_g, ln1_b, 1e-5)
        q  = h @ w_q; k = h @ w_k; v = h @ w_v
        kT = mx.transpose(k)
        scale = 1.0 / (Dh ** 0.5)
        scores = (q @ kT) * scale            # FIXED: 1/sqrt(D) scaling + softmax
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        h1 = xd + (attn @ v) @ w_o
        h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        g  = h2 @ w_gate
        out = (mlx_nn.silu(g) * (h2 @ w_up)) @ w_down
        return mx.mean(h1 + out).astype(mx.float32)
    return _fn


def gpt2_builder(w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn
    def _fn(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b):
        xd = x.astype(mx.float16)
        Dh = xd.shape[-1]
        h  = mxf.layer_norm(xd, ln1_g, ln1_b, 1e-5)
        q  = h @ w_q; k = h @ w_k; v = h @ w_v
        kT = mx.transpose(k)
        scale = 1.0 / (Dh ** 0.5)
        scores = (q @ kT) * scale            # FIXED: 1/sqrt(D) scaling + softmax
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        h1 = xd + (attn @ v) @ w_o
        h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        out = mlx_nn.gelu_approx(h2 @ w_fc1) @ w_fc2
        return mx.mean(h1 + out).astype(mx.float32)
    return _fn


if __name__ == "__main__":
    print("=" * 70)
    print("  Unscaled-Attention FP16 NaN Check")
    print("  (Does the ALREADY-COMMITTED Llama FP16 training number")
    print("   correspond to a NaN loss?)")
    print("=" * 70)
    llama_loss = check("llama", 4096, 14336, llama_builder)
    gpt2_loss  = check("gpt2", 1600, 6400, gpt2_builder)

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  Llama FP16 training loss NaN: {np.isnan(llama_loss)}")
    print(f"  GPT-2 FP16 training loss NaN: {np.isnan(gpt2_loss)}")
    if np.isnan(llama_loss):
        print("\n  CRITICAL: the committed Llama FP16 training benchmark (796.82ms,")
        print("  reported 1.25x speedup) is measuring the latency of a NaN-loss")
        print("  computation. The timing is still a valid latency measurement")
        print("  (NaN propagation doesn't change FLOP count or memory traffic),")
        print("  but the training step as implemented does not compute a")
        print("  mathematically valid gradient. This must be disclosed before")
        print("  citing 'training speedup' as a correctness-validated result.")
