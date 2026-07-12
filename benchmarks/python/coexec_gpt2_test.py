#!/usr/bin/env python3
"""
coexec_gpt2_test.py — Isolation test for GPT-2 co-execution 2× regression.

2×2 design: {compiled, no-compile} × {cpu_ratio=0.0, cpu_ratio=0.25}

Addressess the hypothesis: is the 2× regression a compile-interaction artifact,
or is it inherent to the CPU/GPU split overhead at GPT-2 FFN shapes?

If regression disappears without compile -> compile-interaction failure
If regression persists without compile -> inherent split overhead

Runs in-process (no subprocess isolation) — this is a diagnostic, not a paper-ready
benchmark.

Usage: python coexec_gpt2_test.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))


WARMUPS = 10
RUNS    = 50
D, F    = 1600, 6400    # GPT-2 XL dims
L       = 1024          # seq_len
CPU_RATIO = 0.25        # from tri_calibration_gpt2.json (all shapes: cpu=0.2-0.3)


def build_weights(dtype):
    import mlx.core as mx
    np.random.seed(42); s = 0.02
    def _a(shape):
        return mx.array(np.random.randn(*shape).astype(np.float32) * s).astype(dtype)
    return {
        "w_q": _a((D,D)), "w_k": _a((D,D)), "w_v": _a((D,D)), "w_o": _a((D,D)),
        "w_fc1": _a((D,F)), "w_fc2": _a((F,D)),
        "ln1_g": mx.ones((D,), dtype=dtype), "ln1_b": mx.zeros((D,), dtype=dtype),
        "ln2_g": mx.ones((D,), dtype=dtype), "ln2_b": mx.zeros((D,), dtype=dtype),
    }


def measure(step_fn, warmups=WARMUPS, runs=RUNS):
    for _ in range(warmups):
        step_fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        step_fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean":   float(np.mean(times)),
        "median": float(np.median(times)),
        "std":    float(np.std(times)),
        "ci95":   float(1.96 * np.std(times) / np.sqrt(runs)),
    }


def run_cell(cpu_ratio, use_compile):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn

    w = build_weights(mx.float32)
    x_fp32 = mx.array(np.zeros((L, D), dtype=np.float32))
    cpu_rows = int(L * cpu_ratio)

    def _forward(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b):
        Dh = x.shape[-1]
        h  = mxf.layer_norm(x, ln1_g, ln1_b, 1e-5)
        q  = h @ w_q;  k = h @ w_k;  v = h @ w_v
        kT = mx.transpose(k)
        scale = 1.0 / (Dh ** 0.5)
        scores = (q @ kT) * scale
        attn = mx.softmax(scores, axis=-1)
        h1 = x + (attn @ v) @ w_o

        h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)

        if cpu_rows > 0:
            x_cpu = h2[:cpu_rows]
            x_gpu = h2[cpu_rows:]

            out_gpu = mlx_nn.gelu_approx(x_gpu @ w_fc1) @ w_fc2

            mx.set_default_device(mx.cpu)
            out_cpu = mlx_nn.gelu_approx(x_cpu @ w_fc1) @ w_fc2
            mx.set_default_device(mx.gpu)

            return h1 + mx.concatenate([out_cpu, out_gpu], axis=0)
        else:
            return h1 + mlx_nn.gelu_approx(h2 @ w_fc1) @ w_fc2

    if use_compile:
        fn = mx.compile(_forward)
    else:
        fn = _forward

    def step():
        res = fn(x_fp32, w["w_q"], w["w_k"], w["w_v"], w["w_o"],
                 w["w_fc1"], w["w_fc2"],
                 w["ln1_g"], w["ln1_b"], w["ln2_g"], w["ln2_b"])
        mx.eval(res)

    return measure(step)


def main():
    print("=" * 65)
    print("  GPT-2 Co-Execution Isolation Test")
    print(f"  Shape: [{L}×{D}] (seq_len × dim), FFN dim={F}")
    print(f"  cpu_ratio={CPU_RATIO} (from calibration: all GPT-2 shapes cpu=0.2-0.3)")
    print(f"  Runs: {RUNS}, Warmup: {WARMUPS}")
    print("=" * 65)

    cells = [
        ("compiled=False, cpu_ratio=0.00 (GPU-only baseline)",   False, 0.00),
        ("compiled=False, cpu_ratio=0.25 (co-exec, no compile)", False, CPU_RATIO),
        ("compiled=True,  cpu_ratio=0.00 (GPU-only, compiled)",  True,  0.00),
        ("compiled=True,  cpu_ratio=0.25 (co-exec + compiled)",  True,  CPU_RATIO),
    ]

    results = {}
    baseline_med = None
    for label, use_compile, cpu_ratio in cells:
        print(f"\n▶ {label}")
        print("  (running...)", flush=True)
        stats = run_cell(cpu_ratio, use_compile)
        results[label] = stats
        if baseline_med is None:
            baseline_med = stats["median"]
        ratio = baseline_med / stats["median"]
        print(f"  median={stats['median']:.2f} ms  mean={stats['mean']:.2f} ms  "
              f"std={stats['std']:.2f} ms  vs baseline={ratio:.3f}×")

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("  (baseline = compiled=False, cpu_ratio=0.0)")
    print("=" * 65)
    print(f"  {'Config':<50} | {'Median':>8} | {'vs base':>8}")
    print(f"  {'-'*50}-+-{'-'*8}-+-{'-'*8}")
    base = results[cells[0][0]]["median"]
    for label, _, _ in cells:
        s = results[label]
        print(f"  {label:<50} | {s['median']:>7.2f}ms | {base/s['median']:>7.3f}×")

    print()
    # Diagnostic conclusions
    r_nocomp_coexec = results[cells[1][0]]["median"]
    r_comp_coexec   = results[cells[3][0]]["median"]
    regression_nocomp = r_nocomp_coexec / base
    regression_comp   = r_comp_coexec / base

    print("  DIAGNOSTIC:")
    if regression_nocomp < 0.95:
        print(f"  co-exec WITHOUT compile:  {1/regression_nocomp:.2f}× FASTER")
    elif regression_nocomp > 1.1:
        print(f"  co-exec WITHOUT compile:  {regression_nocomp:.2f}× SLOWER → overhead inherent, not compile")
    else:
        print(f"  co-exec WITHOUT compile:  neutral ({regression_nocomp:.2f}×)")

    if regression_comp > 1.1 and regression_nocomp < 1.1:
        print("  CONCLUSION: regression IS compile-interaction specific")
    elif regression_comp > 1.1 and regression_nocomp > 1.1:
        print("  CONCLUSION: regression is inherent (overhead exists with or without compile)")
    elif regression_comp < 0.95:
        print("  CONCLUSION: co-exec with compile is genuinely faster — revisit model_comparison")
    else:
        print("  CONCLUSION: ambiguous — both are within ±10%")


if __name__ == "__main__":
    main()
