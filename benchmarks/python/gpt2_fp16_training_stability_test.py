#!/usr/bin/env python3
"""
gpt2_fp16_training_stability_test.py
=====================================
Checks whether FP16 backward is numerically safe for GPT-2 XL's training
step before adding FP16 to run_fusion_gpt2's training path (the 1.11x->1.41x
upgrade identified by the training ablation).

Failure mode under test: gradient underflow in w_fc1/w_fc2 (largest
parameter tensors, [1600,6400] and [6400,1600]) without loss scaling.

Runs 50 consecutive training steps in FP16 (no optimizer step — just
forward + value_and_grad, matching the ablation's measurement scope) and
reports, per step: gradient L2 norm for every parameter, and whether any
gradient contains NaN/Inf or is exactly zero (a proxy for total underflow).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

D, F, L = 1600, 6400, 1024
STEPS = 50


def main():
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn

    np.random.seed(42); scale = 0.02

    def _a(shape, dtype):
        return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(dtype)

    dtype = mx.float16
    w_q = _a((D, D), dtype); w_k = _a((D, D), dtype); w_v = _a((D, D), dtype); w_o = _a((D, D), dtype)
    w_fc1 = _a((D, F), dtype); w_fc2 = _a((F, D), dtype)
    ln1_g = mx.ones((D,), dtype=dtype); ln1_b = mx.zeros((D,), dtype=dtype)
    ln2_g = mx.ones((D,), dtype=dtype); ln2_b = mx.zeros((D,), dtype=dtype)

    param_names = ["w_q", "w_k", "w_v", "w_o", "w_fc1", "w_fc2", "ln1_g", "ln1_b", "ln2_g", "ln2_b"]

    def _train(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b):
        xd = x.astype(dtype)
        h  = mxf.layer_norm(xd, ln1_g, ln1_b, 1e-5)
        q  = h @ w_q; k = h @ w_k; v = h @ w_v
        h1 = xd + (q @ mx.transpose(k) @ v) @ w_o
        h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        out = mlx_nn.gelu_approx(h2 @ w_fc1) @ w_fc2
        return mx.mean(h1 + out).astype(mx.float32)

    grad_fn = mx.value_and_grad(_train, argnums=list(range(1, 11)))
    grad_fn = mx.compile(grad_fn)

    x_fp32 = mx.array(np.random.randn(L, D).astype(np.float32) * scale)

    print("=" * 70)
    print("  GPT-2 XL FP16 Training Stability Check")
    print(f"  Steps: {STEPS}  (forward + value_and_grad, no optimizer step)")
    print("=" * 70)

    any_nan = False
    any_zero_norm = False
    norms_over_time = {name: [] for name in param_names}

    for step in range(STEPS):
        loss, grads = grad_fn(x_fp32, w_q, w_k, w_v, w_o, w_fc1, w_fc2,
                               ln1_g, ln1_b, ln2_g, ln2_b)
        mx.eval(loss, grads)

        step_has_nan = False
        step_has_zero = False
        for name, g in zip(param_names, grads):
            g32 = g.astype(mx.float32)
            norm = float(mx.sqrt(mx.sum(g32 * g32)).item())
            norms_over_time[name].append(norm)
            if np.isnan(norm) or np.isinf(norm):
                step_has_nan = True
                any_nan = True
            if norm == 0.0:
                step_has_zero = True
                any_zero_norm = True

        if step == 0 or step == STEPS - 1 or step_has_nan or step_has_zero:
            flag = " ⚠ NaN/Inf" if step_has_nan else (" ⚠ ZERO" if step_has_zero else "")
            print(f"  step {step:3d}: loss={float(loss):.6f}  "
                  f"w_fc1_grad_norm={norms_over_time['w_fc1'][-1]:.6e}  "
                  f"w_fc2_grad_norm={norms_over_time['w_fc2'][-1]:.6e}{flag}")

    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"  NaN/Inf detected across {STEPS} steps: {any_nan}")
    print(f"  Zero-norm gradient detected: {any_zero_norm}")
    print(f"\n  Final gradient norms (step {STEPS-1}):")
    for name in param_names:
        print(f"    {name:8s}: {norms_over_time[name][-1]:.6e}")

    if not any_nan and not any_zero_norm:
        print("\n  CONCLUSION: FP16 training appears numerically stable over "
              f"{STEPS} steps without loss scaling. Safe to benchmark for "
              "paper inclusion, but this is not a substitute for a full "
              "convergence test over real training data.")
    else:
        print("\n  CONCLUSION: FP16 training shows instability. "
              "Do not add FP16 to run_fusion_gpt2's training path without "
              "a GradScaler (loss scaling).")


if __name__ == "__main__":
    main()
