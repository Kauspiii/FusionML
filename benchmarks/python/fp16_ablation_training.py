#!/usr/bin/env python3
"""
fp16_ablation_training.py
=========================
2×2 ablation study: {FP32, FP16} × {no-compile, compiled}
on Llama-3-8B and GPT-2 XL decoder blocks — TRAINING (forward+backward) only.

Separates the FusionML training speedup into two orthogonal contributions:
  - FP16 precision alone        (half the memory bandwidth, smaller gradient tensors)
  - mx.compile(value_and_grad)  (JIT fusion of fwd+bwd graph)
  - Combined (FusionML default for training)

Each config runs in a subprocess for memory isolation.
30 timed runs, reports mean ± CI95 and speedup vs FP32-NoCompile baseline.

NOTE: FP16 backward pass is numerically unstable for real training (gradient
underflow without gradient scaling). This ablation measures performance only —
correctness of gradients is not asserted.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

# =============================================================================
# Config — mirrors model_comparison.py shapes exactly
# =============================================================================

LLAMA = {"dim": 4096, "ffn_dim": 14336, "seq_len": 1024}
GPT2  = {"dim": 1600, "ffn_dim": 6400,  "seq_len": 1024}

WARMUPS = 10
RUNS    = 50

MODES = [
    ("FP32-NoCompile", dict(fp16=False, compiled=False)),  # MLX baseline
    ("FP16-NoCompile", dict(fp16=True,  compiled=False)),  # FP16 alone
    ("FP32-Compiled",  dict(fp16=False, compiled=True)),   # compile alone
    ("FP16-Compiled",  dict(fp16=True,  compiled=True)),   # FusionML default
]


# =============================================================================
# Subprocess worker — training (value_and_grad)
# =============================================================================

def run_worker(model, fp16, compiled):
    """Run one (model, fp16, compiled) training config; return stats dict."""
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn
    import resource

    cfg   = LLAMA if model == "llama" else GPT2
    D, F  = cfg["dim"], cfg["ffn_dim"]
    L     = cfg["seq_len"]
    dtype = mx.float16 if fp16 else mx.float32
    np.random.seed(42); s = 0.02

    def _arr(shape):
        return mx.array(np.random.randn(*shape).astype(np.float32) * s).astype(dtype)

    w_q   = _arr((D, D));  w_k = _arr((D, D)); w_v = _arr((D, D)); w_o = _arr((D, D))
    ln1_g = mx.ones((D,), dtype=dtype); ln1_b = mx.zeros((D,), dtype=dtype)
    ln2_g = mx.ones((D,), dtype=dtype); ln2_b = mx.zeros((D,), dtype=dtype)

    if model == "llama":
        w_gate = _arr((D, F)); w_up = _arr((D, F)); w_down = _arr((F, D))

        def _llama_train(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down,
                         ln1_g, ln1_b, ln2_g, ln2_b):
            xd = x.astype(dtype)
            Dh = xd.shape[-1]
            h  = mxf.layer_norm(xd, ln1_g, ln1_b, 1e-5)
            q  = h @ w_q; k = h @ w_k; v = h @ w_v
            scale = 1.0 / (Dh ** 0.5)
            scores = (q @ mx.transpose(k)) * scale
            attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
            h1 = xd + (attn @ v) @ w_o
            h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
            g  = h2 @ w_gate
            out = (mlx_nn.silu(g) * (h2 @ w_up)) @ w_down
            return mx.mean(h1 + out).astype(mx.float32)

        grad_fn = mx.value_and_grad(_llama_train, argnums=list(range(1, 12)))
        if compiled:
            grad_fn = mx.compile(grad_fn)

        def step(x_in):
            loss, grads = grad_fn(
                x_in, w_q, w_k, w_v, w_o, w_gate, w_up, w_down,
                ln1_g, ln1_b, ln2_g, ln2_b
            )
            mx.eval(loss, grads)

    else:  # gpt2
        w_fc1 = _arr((D, F)); w_fc2 = _arr((F, D))

        def _gpt2_train(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2,
                        ln1_g, ln1_b, ln2_g, ln2_b):
            xd = x.astype(dtype)
            Dh = xd.shape[-1]
            h  = mxf.layer_norm(xd, ln1_g, ln1_b, 1e-5)
            q  = h @ w_q; k = h @ w_k; v = h @ w_v
            scale = 1.0 / (Dh ** 0.5)
            scores = (q @ mx.transpose(k)) * scale
            attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
            h1 = xd + (attn @ v) @ w_o
            h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
            out = mlx_nn.gelu_approx(h2 @ w_fc1) @ w_fc2
            return mx.mean(h1 + out).astype(mx.float32)

        grad_fn = mx.value_and_grad(_gpt2_train, argnums=list(range(1, 11)))
        if compiled:
            grad_fn = mx.compile(grad_fn)

        def step(x_in):
            loss, grads = grad_fn(
                x_in, w_q, w_k, w_v, w_o, w_fc1, w_fc2,
                ln1_g, ln1_b, ln2_g, ln2_b
            )
            mx.eval(loss, grads)

    x_fp32 = mx.array(np.zeros((L, D), dtype=np.float32))

    for _ in range(WARMUPS):
        step(x_fp32)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        step(x_fp32)
        times.append((time.perf_counter() - t0) * 1000.0)

    n = len(times)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    return {
        "mean":        float(np.mean(times)),
        "std":         float(np.std(times)),
        "median":      float(np.median(times)),
        "ci95":        float(1.96 * np.std(times) / np.sqrt(n)),
        "n_runs":      n,
        "peak_mem_mb": peak_mb,
    }


def run_sub(model, fp16, compiled):
    cmd = [
        sys.executable, __file__,
        "--sub", "--model", model,
        "--fp16",    "1" if fp16    else "0",
        "--compiled","1" if compiled else "0",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..", "python")
    )
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if res.returncode != 0:
        print(f"  ⚠ {model} fp16={fp16} compiled={compiled}: {res.stderr[:300]}",
              file=sys.stderr)
        return None
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# =============================================================================
# Orchestrator
# =============================================================================

def get_hw_slug():
    import re
    cpu = "Unknown"
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True)
        cpu = r.stdout.strip()
    except Exception:
        pass
    mem_gb = "?GB"
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True)
        mem_gb = f"{int(r.stdout.strip()) // (1024 ** 3)}GB"
    except Exception:
        pass
    cpu_cores = "?"
    try:
        r = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                           capture_output=True, text=True)
        v = r.stdout.strip()
        if v.isdigit():
            cpu_cores = v
    except Exception:
        pass
    gpu_cores = "?"
    try:
        r = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator"],
                           capture_output=True, text=True)
        m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', r.stdout)
        if m:
            gpu_cores = m.group(1)
    except Exception:
        pass
    _ANE = [
        ("M1 Ultra","32"),("M2 Ultra","32"),("M3 Ultra","36"),("M4 Ultra","64"),
        ("M3 Pro","18"),("M3 Max","18"),("M4 Pro","20"),("M4 Max","32"),
    ]
    ane = "16"
    for k, v in _ANE:
        if k in cpu: ane = v; break
    return f"{cpu}_{mem_gb}_{cpu_cores}CPU_{gpu_cores}GPU_{ane}ANE".replace(" ", "_"), cpu


def print_ablation_table(model_name, results):
    baseline = results.get("FP32-NoCompile", {}).get("median", 1.0)
    print(f"\n  {model_name} — Training Ablation (forward + backward)")
    print(f"  {'Config':<20} | {'Mean ± CI95 (ms)':>20} | {'Median (ms)':>12} | {'vs FP32-Base':>13} | {'Mem (MB)':>9}")
    print(f"  {'-'*20}-+-{'-'*20}-+-{'-'*12}-+-{'-'*13}-+-{'-'*9}")
    for label, _ in MODES:
        s = results.get(label)
        if not s:
            print(f"  {label:<20} | {'N/A':>20} | {'N/A':>12} | {'N/A':>13} | {'N/A':>9}")
            continue
        ratio = baseline / s["median"] if s["median"] > 0 else 0
        marker = " ← FusionML" if label == "FP16-Compiled" else ""
        print(f"  {label:<20} | {s['mean']:>9.2f} ± {s['ci95']:>6.2f}   | {s['median']:>12.2f} | "
              f"{ratio:>12.2f}× | {s['peak_mem_mb']:>9.0f}{marker}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub",      action="store_true")
    parser.add_argument("--model",    type=str, choices=["llama", "gpt2"])
    parser.add_argument("--fp16",     type=str, choices=["0", "1"])
    parser.add_argument("--compiled", type=str, choices=["0", "1"])
    args = parser.parse_args()

    if args.sub:
        stats = run_worker(args.model, args.fp16 == "1", args.compiled == "1")
        print(json.dumps(stats))
        return

    hw_slug, cpu_name = get_hw_slug()
    print("=" * 70)
    print("  FusionML FP16 × Compile Ablation Study — TRAINING")
    print(f"  Hardware: {cpu_name}")
    print(f"  Axes: {{FP32, FP16}} × {{no-compile, compiled}}")
    print(f"  Runs: {RUNS}  |  Warmup: {WARMUPS}")
    print(f"  Task: forward + value_and_grad (backward), no optimizer step")
    print("=" * 70)

    all_results = {}
    for model_key, model_name in [("llama", "Llama-3-8B"), ("gpt2", "GPT-2 XL")]:
        print(f"\n▶ {model_name}")
        all_results[model_key] = {}
        for label, flags in MODES:
            print(f"   {label:<20} ...", end=" ", flush=True)
            time.sleep(5.0)  # cooldown between modes
            s = run_sub(model_key, flags["fp16"], flags["compiled"])
            if s:
                all_results[model_key][label] = s
                base = all_results[model_key].get("FP32-NoCompile", {}).get("median")
                ratio = (base / s["median"]) if (base and label != "FP32-NoCompile") else 1.0
                print(f"{s['median']:.2f} ms  (×{ratio:.2f})")
            else:
                all_results[model_key][label] = None
                print("FAILED")

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    for model_key, model_name in [("llama", "Llama-3-8B"), ("gpt2", "GPT-2 XL")]:
        print_ablation_table(model_name, all_results[model_key])

    # Attribution breakdown
    print("\n" + "=" * 70)
    print("  GAIN ATTRIBUTION")
    print("=" * 70)
    for model_key, model_name in [("llama", "Llama-3-8B"), ("gpt2", "GPT-2 XL")]:
        r = all_results[model_key]
        b  = r.get("FP32-NoCompile", {}).get("median")
        fp = r.get("FP16-NoCompile", {}).get("median")
        cp = r.get("FP32-Compiled",  {}).get("median")
        fc = r.get("FP16-Compiled",  {}).get("median")
        if b and fp and cp and fc:
            fp16_gain    = b / fp
            compile_gain = b / cp
            interact     = (b / fc) / (fp16_gain * compile_gain)
            print(f"\n  {model_name}:")
            print(f"    FP16 alone:      {fp16_gain:.3f}×  ({b:.1f} → {fp:.1f} ms)")
            print(f"    Compile alone:   {compile_gain:.3f}×  ({b:.1f} → {cp:.1f} ms)")
            print(f"    Combined:        {b/fc:.3f}×  ({b:.1f} → {fc:.1f} ms)")
            print(f"    Interaction:     {interact:.3f}×  (combined / fp16×compile)")

    # Save
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../results", hw_slug)
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fp16_ablation_training.json")
    with open(out_path, "w") as f:
        json.dump({
            "hardware_slug": hw_slug,
            "task":    "training",
            "modes":   [m[0] for m in MODES],
            "warmups": WARMUPS,
            "runs":    RUNS,
            "results": all_results,
        }, f, indent=2)
    print(f"\n💾 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
