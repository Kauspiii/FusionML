#!/usr/bin/env python3
"""
ane_spike_test.py — empirical spike: is ANE viable at all for GPT-2/Llama-shaped
matmuls on this M1 8GB, before investing engineering into a real 3-way scheduler?

Measures ane_matmul() (real CoreML/ANE execution, from fusionml._metal.ane_backend)
against CPU (numpy/Accelerate) and GPU (MLX) for representative row-chunk shapes
that a CPU+GPU+ANE split would actually produce.

Run in a subprocess (isolated) to avoid polluting the main process with CoreML
model cache state, and to contain any OOM/crash to a disposable process.
"""

import os
import sys
import time
import subprocess
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

# Representative shapes: GPT-2 FFN chunk sizes for a 3-way split of 1024 rows
# (e.g. 60% GPU / 25% CPU / 15% ANE of a 1024x1600x6400 FFN matmul)
SHAPES = [
    ("ane_chunk_gpt2_ffn",   154, 1600, 6400),   # ~15% of 1024 rows
    ("cpu_chunk_gpt2_ffn",   256, 1600, 6400),   # ~25% of 1024 rows (current CPU split size)
    ("full_gpt2_ffn",       1024, 1600, 6400),   # full shape, for reference
]


def run_worker(label, M, K, N):
    from fusionml._metal.ane_backend import ane_matmul, ane_available, HAS_COREML

    result = {"label": label, "M": M, "K": K, "N": N, "has_coreml": HAS_COREML}

    if not HAS_COREML:
        result["error"] = "coremltools not installed"
        return result

    np.random.seed(42)
    a = (np.random.randn(M, K) * 0.02).astype(np.float32)
    b = (np.random.randn(K, N) * 0.02).astype(np.float32)

    # First call: includes compilation
    t0 = time.perf_counter()
    try:
        out = ane_matmul(a, b, compute_units="CPU_AND_NE")
        first_call_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        result["error"] = f"ane_matmul failed: {type(e).__name__}: {e}"
        return result

    # Correctness sanity check vs numpy
    expected = a @ b
    max_abs_err = float(np.max(np.abs(out.astype(np.float64) - expected.astype(np.float64))))
    rel_err = max_abs_err / (float(np.max(np.abs(expected))) + 1e-8)

    # Steady-state calls (model now cached)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        ane_matmul(a, b, compute_units="CPU_AND_NE")
        times.append((time.perf_counter() - t0) * 1000.0)

    # CPU (numpy/Accelerate) reference for the same shape
    for _ in range(3):
        _ = a @ b
    cpu_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = a @ b
        cpu_times.append((time.perf_counter() - t0) * 1000.0)

    # GPU (MLX) reference for the same shape
    gpu_times = []
    try:
        import mlx.core as mx
        a_mx, b_mx = mx.array(a), mx.array(b)
        for _ in range(3):
            mx.eval(a_mx @ b_mx)
        for _ in range(20):
            t0 = time.perf_counter()
            mx.eval(a_mx @ b_mx)
            gpu_times.append((time.perf_counter() - t0) * 1000.0)
    except Exception as e:
        result["gpu_error"] = str(e)

    result.update({
        "first_call_ms": first_call_ms,
        "ane_steady_median_ms": float(np.median(times)),
        "ane_steady_std_ms": float(np.std(times)),
        "cpu_median_ms": float(np.median(cpu_times)),
        "gpu_median_ms": float(np.median(gpu_times)) if gpu_times else None,
        "max_abs_err_vs_numpy": max_abs_err,
        "rel_err_vs_numpy": rel_err,
    })
    return result


def run_sub(label, M, K, N):
    cmd = [sys.executable, __file__, "--sub", "--label", label,
           "--M", str(M), "--K", str(K), "--N", str(N)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python"))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    if res.returncode != 0:
        return {"label": label, "error": f"subprocess crashed: {res.stderr[-800:]}"}
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {"label": label, "error": "no JSON output", "stdout": res.stdout[-500:]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--label", type=str)
    parser.add_argument("--M", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--N", type=int)
    args = parser.parse_args()

    if args.sub:
        print(json.dumps(run_worker(args.label, args.M, args.K, args.N)))
        return

    print("=" * 70)
    print("  ANE Viability Spike Test")
    print("  Real CoreML/ANE execution vs CPU (Accelerate) vs GPU (MLX)")
    print("=" * 70)

    results = []
    for label, M, K, N in SHAPES:
        print(f"\n▶ {label}: [{M}x{K}] @ [{K}x{N}]")
        r = run_sub(label, M, K, N)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  First call (incl. compile): {r['first_call_ms']:.1f} ms")
        print(f"  ANE steady-state:  median={r['ane_steady_median_ms']:.3f} ms  std={r['ane_steady_std_ms']:.3f} ms")
        print(f"  CPU (Accelerate):  median={r['cpu_median_ms']:.3f} ms")
        gpu_str = f"{r['gpu_median_ms']:.3f} ms" if r.get('gpu_median_ms') else "N/A"
        print(f"  GPU (MLX):         median={gpu_str}")
        print(f"  Correctness: max_abs_err={r['max_abs_err_vs_numpy']:.2e}  rel_err={r['rel_err_vs_numpy']:.2e}")
        if r['gpu_median_ms']:
            print(f"  ANE/GPU ratio: {r['ane_steady_median_ms']/r['gpu_median_ms']:.2f}x "
                  f"({'ANE faster' if r['ane_steady_median_ms'] < r['gpu_median_ms'] else 'ANE slower'})")
        print(f"  ANE/CPU ratio: {r['ane_steady_median_ms']/r['cpu_median_ms']:.2f}x "
              f"({'ANE faster' if r['ane_steady_median_ms'] < r['cpu_median_ms'] else 'ANE slower'})")

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    viable = [r for r in results if "error" not in r and r.get("gpu_median_ms")
              and r["ane_steady_median_ms"] < r["gpu_median_ms"]]
    if not viable:
        print("  ANE is SLOWER than GPU for every tested shape, even at steady-state")
        print("  (model compiled and cached). Adding ANE to the model-level scheduler")
        print("  would not help for these matmul sizes on this hardware.")
    else:
        print(f"  ANE beat GPU for: {[r['label'] for r in viable]}")
        print("  ANE may be viable for these shapes -- worth prototyping in the scheduler.")

    from bench_hw import get_system_info
    out_path = os.path.join(os.path.dirname(__file__), "../results",
                             get_system_info()["cpu_slug"], "ane_spike_test.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
