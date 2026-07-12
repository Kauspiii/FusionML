#!/usr/bin/env python3
"""
mlx_fp16_baseline_test.py
==========================
Rigorous test of the most critical open question: does FusionML actually
outperform MLX, or only outperform MLX-in-FP32?

Uses the ACTUAL run_mlx_llama / run_mlx_gpt2 functions from model_comparison.py
(not a reimplementation) with weights cast to FP16 before the call — this is
the closest thing to "what if MLX benchmarked itself in FP16" without
modifying MLX's own code.

Compares against FusionML's current numbers (FP16 + compile) from the
committed model_comparison.json.

Each config runs in a subprocess for isolation. n=30, warmup=10.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

LLAMA = {"dim": 4096, "ffn_dim": 14336, "seq_len": 1024}
GPT2  = {"dim": 1600, "ffn_dim": 6400,  "seq_len": 1024}

WARMUPS = 10
RUNS    = 50


def run_worker(model, dtype_name):
    import mlx.core as mx
    from model_comparison import run_mlx_llama, run_mlx_gpt2
    import resource

    cfg = LLAMA if model == "llama" else GPT2
    D, F, L = cfg["dim"], cfg["ffn_dim"], cfg["seq_len"]
    dtype = mx.float16 if dtype_name == "fp16" else mx.float32
    np.random.seed(42); scale = 0.02

    def _a(shape):
        return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(dtype)

    x = _a((L, D))
    if model == "llama":
        weights = {
            'w_q': _a((D, D)), 'w_k': _a((D, D)), 'w_v': _a((D, D)), 'w_o': _a((D, D)),
            'w_gate': _a((D, F)), 'w_up': _a((D, F)), 'w_down': _a((F, D)),
            'ln1_g': mx.ones((D,), dtype=dtype), 'ln1_b': mx.zeros((D,), dtype=dtype),
            'ln2_g': mx.ones((D,), dtype=dtype), 'ln2_b': mx.zeros((D,), dtype=dtype),
        }
        fn = lambda: run_mlx_llama(x, weights, training=False)
    else:
        weights = {
            'w_q': _a((D, D)), 'w_k': _a((D, D)), 'w_v': _a((D, D)), 'w_o': _a((D, D)),
            'w_fc1': _a((D, F)), 'w_fc2': _a((F, D)),
            'ln1_g': mx.ones((D,), dtype=dtype), 'ln1_b': mx.zeros((D,), dtype=dtype),
            'ln2_g': mx.ones((D,), dtype=dtype), 'ln2_b': mx.zeros((D,), dtype=dtype),
        }
        fn = lambda: run_mlx_gpt2(x, weights, training=False)

    for _ in range(WARMUPS):
        fn()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)

    n = len(times)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    return {
        "mean": float(np.mean(times)), "std": float(np.std(times)),
        "median": float(np.median(times)), "ci95": float(1.96 * np.std(times) / np.sqrt(n)),
        "n_runs": n, "peak_mem_mb": peak_mb,
    }


def run_sub(model, dtype_name):
    cmd = [sys.executable, __file__, "--sub", "--model", model, "--dtype", dtype_name]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python")) \
        + os.pathsep + os.path.dirname(__file__)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if res.returncode != 0:
        print(f"  ⚠ {model} dtype={dtype_name}: {res.stderr[:500]}", file=sys.stderr)
        return None
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--model", choices=["llama", "gpt2"])
    parser.add_argument("--dtype", choices=["fp32", "fp16"])
    args = parser.parse_args()

    if args.sub:
        print(json.dumps(run_worker(args.model, args.dtype)))
        return

    # FusionML committed numbers (FP16 + compile, WITH corrected 1/sqrt(D) softmax
    # attention), from model_comparison.json -- regenerated after the attention fix.
    fusionml_current = {
        "llama": 301.72,  # ms, median
        "gpt2":  39.22,
    }

    print("=" * 70)
    print("  MLX-FP16 Baseline Test — Critical Section-2 Validity Check")
    print(f"  Runs: {RUNS}  Warmup: {WARMUPS}")
    print("=" * 70)

    all_results = {}
    for model, name in [("llama", "Llama-3-8B"), ("gpt2", "GPT-2 XL")]:
        print(f"\n▶ {name}")
        all_results[model] = {}
        for dtype in ["fp32", "fp16"]:
            print(f"   MLX-{dtype.upper():5s} (using run_mlx_{model}, actual model_comparison.py code) ... ", end="", flush=True)
            time.sleep(5.0)
            s = run_sub(model, dtype)
            all_results[model][dtype] = s
            if s:
                print(f"{s['median']:.2f} ms  ±{s['ci95']:.2f}")
            else:
                print("FAILED")

    print("\n" + "=" * 70)
    print("  RESULT: FusionML vs MLX, precision-matched")
    print("=" * 70)
    for model, name in [("llama", "Llama-3-8B"), ("gpt2", "GPT-2 XL")]:
        r = all_results[model]
        fml = fusionml_current[model]
        print(f"\n  {name}:")
        if r.get("fp32"):
            print(f"    MLX-FP32:  {r['fp32']['median']:>8.2f} ms   FusionML/MLX-FP32 speedup: {r['fp32']['median']/fml:.3f}×")
        if r.get("fp16"):
            print(f"    MLX-FP16:  {r['fp16']['median']:>8.2f} ms   FusionML/MLX-FP16 speedup: {r['fp16']['median']/fml:.3f}×")
        print(f"    FusionML:  {fml:>8.2f} ms  (FP16 + mx.compile, committed number)")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results", "Apple_M1_8GB_8CPU_7GPU_16ANE"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mlx_fp16_baseline_test.json"), "w") as f:
        json.dump({"fusionml_current": fusionml_current, "results": all_results}, f, indent=2)
    print(f"\n💾 Saved to {out_dir}/mlx_fp16_baseline_test.json")


if __name__ == "__main__":
    main()
