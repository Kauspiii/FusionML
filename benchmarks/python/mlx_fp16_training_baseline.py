#!/usr/bin/env python3
"""
mlx_fp16_training_baseline.py
==============================
Resolves the last dagger-footnoted claim: FusionML training speedups
(1.35x Llama / 1.13x GPT-2) were only ever measured against MLX-FP32.
This measures MLX training natively in FP16 — the fair baseline — and
re-measures FusionML training fresh in the same session, so both sides
share the same thermal state.

Uses the ACTUAL run_mlx_llama / run_mlx_gpt2 functions from
model_comparison.py (training=True) with FP16-cast weights, and the
actual model_comparison.py fusionml subprocess worker for the FusionML
side. No hardcoded comparison values.

Also verifies the training loss is finite for every configuration
(inspecting the loss scalar itself, per the FP16-NaN postmortem).

Each config runs in its own subprocess, strictly sequentially.
n=50, warmup=10.
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
        fn = lambda: run_mlx_llama(x, weights, training=True)
    else:
        weights = {
            'w_q': _a((D, D)), 'w_k': _a((D, D)), 'w_v': _a((D, D)), 'w_o': _a((D, D)),
            'w_fc1': _a((D, F)), 'w_fc2': _a((F, D)),
            'ln1_g': mx.ones((D,), dtype=dtype), 'ln1_b': mx.zeros((D,), dtype=dtype),
            'ln2_g': mx.ones((D,), dtype=dtype), 'ln2_b': mx.zeros((D,), dtype=dtype),
        }
        fn = lambda: run_mlx_gpt2(x, weights, training=True)

    loss_val = None
    for _ in range(WARMUPS):
        loss = fn()
        loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
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
        "final_loss": loss_val,
        "loss_finite": bool(np.isfinite(loss_val)) if loss_val is not None else None,
    }


def run_sub_mlx(model, dtype_name):
    cmd = [sys.executable, __file__, "--sub", "--model", model, "--dtype", dtype_name]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python")) \
        + os.pathsep + os.path.dirname(__file__)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if res.returncode != 0:
        print(f"  ⚠ {model} dtype={dtype_name}: {res.stderr[:500]}", file=sys.stderr)
        return None
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return None


def run_sub_fusionml(model):
    """Fresh FusionML training run via the actual model_comparison.py worker."""
    mc_path = os.path.join(os.path.dirname(__file__), "model_comparison.py")
    cmd = [sys.executable, mc_path, "--sub", "--fw", "fusionml", "--mode", "training", "--model", model]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python"))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if res.returncode != 0:
        print(f"  ⚠ fusionml {model}: {res.stderr[:500]}", file=sys.stderr)
        return None
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
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

    print("=" * 70)
    print("  MLX-FP16 TRAINING Baseline — resolves the dagger footnote")
    print(f"  Runs: {RUNS}  Warmup: {WARMUPS}")
    print("=" * 70)

    all_results = {}
    for model, name in [("llama", "Llama-3-8B"), ("gpt2", "GPT-2 XL")]:
        print(f"\n▶ {name} (training)")
        all_results[model] = {}
        for dtype in ["fp32", "fp16"]:
            print(f"   MLX-{dtype.upper():5s} ... ", end="", flush=True)
            time.sleep(5.0)
            s = run_sub_mlx(model, dtype)
            all_results[model][f"mlx_{dtype}"] = s
            if s:
                finite = "loss finite" if s["loss_finite"] else "LOSS NOT FINITE — INVALID"
                print(f"{s['median']:.2f} ms  ±{s['ci95']:.2f}  ({finite})")
            else:
                print("FAILED")
        print(f"   FusionML  ... ", end="", flush=True)
        time.sleep(5.0)
        s = run_sub_fusionml(model)
        all_results[model]["fusionml"] = s
        print(f"{s['median']:.2f} ms  ±{s.get('ci95', 0):.2f}" if s else "FAILED")

    print("\n" + "=" * 70)
    print("  RESULT: FusionML training vs MLX, precision-matched")
    print("=" * 70)
    for model, name in [("llama", "Llama-3-8B"), ("gpt2", "GPT-2 XL")]:
        r = all_results[model]
        fml = r.get("fusionml")
        if not fml:
            continue
        print(f"\n  {name}:")
        for key, label in [("mlx_fp32", "MLX-FP32"), ("mlx_fp16", "MLX-FP16")]:
            if r.get(key):
                sp = r[key]["median"] / fml["median"]
                print(f"    {label}: {r[key]['median']:>9.2f} ms   FusionML speedup vs {label}: {sp:.3f}x")
        print(f"    FusionML: {fml['median']:>9.2f} ms  (fresh run, same session)")

    from bench_hw import get_system_info, get_power_state
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results", get_system_info()["cpu_slug"]))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mlx_fp16_training_baseline.json")
    with open(out_path, "w") as f:
        json.dump({"warmups": WARMUPS, "runs": RUNS,
                   "environment": get_power_state(), "results": all_results}, f, indent=2)
    print(f"\n💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
