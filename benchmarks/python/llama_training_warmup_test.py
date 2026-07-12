#!/usr/bin/env python3
"""
llama_training_warmup_test.py
==============================
Tests whether the discrepancy between:
  - fp16_ablation_training.py's FusionML Llama training number (758.99ms, n=30/warmup=10)
  - model_comparison.py's committed FusionML Llama training number (796.82ms, n=15/warmup=5)
is explained by insufficient warmup for the mx.compile-traced training step,
rather than a structural difference in what's being measured.

Runs run_fusion_llama's actual training path (imported from model_comparison.py,
not reimplemented) at both warmup counts back to back in the same subprocess,
holding n=30 fixed for both so only warmup count varies.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

RUNS = 30
D, F, L = 4096, 14336, 1024


def run_worker(warmups):
    from model_comparison import run_fusion_llama
    from fusionml.tensor import Tensor
    from fusionml._metal.tri_scheduler import get_scheduler
    import resource

    scheduler = get_scheduler()
    cache_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../results/tri_calibration_llama.json")
    )
    if os.path.exists(cache_path):
        scheduler.load_calibration(cache_path)

    np.random.seed(42); scale = 0.02
    x_np = np.random.randn(L, D).astype(np.float32) * scale
    w_q_np = np.random.randn(D, D).astype(np.float32) * scale
    w_k_np = np.random.randn(D, D).astype(np.float32) * scale
    w_v_np = np.random.randn(D, D).astype(np.float32) * scale
    w_o_np = np.random.randn(D, D).astype(np.float32) * scale
    w_gate_np = np.random.randn(D, F).astype(np.float32) * scale
    w_up_np = np.random.randn(D, F).astype(np.float32) * scale
    w_down_np = np.random.randn(F, D).astype(np.float32) * scale
    ln1_g_np = np.ones(D, dtype=np.float32); ln1_b_np = np.zeros(D, dtype=np.float32)
    ln2_g_np = np.ones(D, dtype=np.float32); ln2_b_np = np.zeros(D, dtype=np.float32)

    x = Tensor(x_np, requires_grad=False).to_gpu()
    weights = {
        'w_q': Tensor(w_q_np, requires_grad=True).to_gpu(),
        'w_k': Tensor(w_k_np, requires_grad=True).to_gpu(),
        'w_v': Tensor(w_v_np, requires_grad=True).to_gpu(),
        'w_o': Tensor(w_o_np, requires_grad=True).to_gpu(),
        'w_gate': Tensor(w_gate_np, requires_grad=True).to_gpu(),
        'w_up': Tensor(w_up_np, requires_grad=True).to_gpu(),
        'w_down': Tensor(w_down_np, requires_grad=True).to_gpu(),
        'ln1_g': Tensor(ln1_g_np, requires_grad=True).to_gpu(),
        'ln1_b': Tensor(ln1_b_np, requires_grad=True).to_gpu(),
        'ln2_g': Tensor(ln2_g_np, requires_grad=True).to_gpu(),
        'ln2_b': Tensor(ln2_b_np, requires_grad=True).to_gpu(),
    }
    for w in weights.values():
        w.is_parameter = True

    fn = lambda: run_fusion_llama(x, weights, training=True)

    for _ in range(warmups):
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
        "n_runs": n, "warmups": warmups, "peak_mem_mb": peak_mb,
        "first_5_times": times[:5],  # to visually inspect for residual JIT contamination
    }


def run_sub(warmups):
    cmd = [sys.executable, __file__, "--sub", "--warmups", str(warmups)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python")) \
        + os.pathsep + os.path.dirname(__file__)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if res.returncode != 0:
        print(f"  ⚠ warmups={warmups}: {res.stderr[:500]}", file=sys.stderr)
        return None
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--warmups", type=int)
    args = parser.parse_args()

    if args.sub:
        print(json.dumps(run_worker(args.warmups)))
        return

    print("=" * 70)
    print("  Llama Training Warmup Sensitivity Test")
    print("  Comparing warmup=5 (main benchmark) vs warmup=10 (ablation)")
    print(f"  n={RUNS} timed runs for both")
    print("=" * 70)

    results = {}
    for warmups in [5, 10]:
        print(f"\n▶ warmup={warmups} ... ", end="", flush=True)
        time.sleep(5.0)
        s = run_sub(warmups)
        results[warmups] = s
        if s:
            print(f"median={s['median']:.2f}ms  first_5_times={[f'{t:.1f}' for t in s['first_5_times']]}")
        else:
            print("FAILED")

    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"  Ablation reported:        758.99 ms (n=30, warmup=10)")
    print(f"  Main benchmark reported:  796.82 ms (n=15, warmup=5)")
    if results.get(5) and results.get(10):
        m5, m10 = results[5]['median'], results[10]['median']
        print(f"  This test @ warmup=5:     {m5:.2f} ms (n=30)")
        print(f"  This test @ warmup=10:    {m10:.2f} ms (n=30)")
        gap_explained = abs(m5 - m10) / abs(796.82 - 758.99) if (796.82 - 758.99) != 0 else 0
        print(f"\n  Warmup-attributable gap: {m5-m10:+.2f} ms")
        print(f"  Original gap to explain: {796.82-758.99:+.2f} ms")
        if abs(m5 - m10) > 0.5 * abs(796.82 - 758.99):
            print("  CONCLUSION: warmup count explains the majority of the discrepancy.")
        else:
            print("  CONCLUSION: warmup count does NOT fully explain the discrepancy — "
                  "residual gap likely from n=15 vs n=30 sample-size noise, or a "
                  "structural difference between the two scripts' forward math.")


if __name__ == "__main__":
    main()
