#!/usr/bin/env python3
"""
prefill_scale_benchmark.py
===========================
The headline experiment: does per-layer concurrent CPU+GPU splitting beat a
fair MLX-FP16 baseline at prefill/batch scale (large row counts), where
tricompute_adaptive_search already proved 1.6-1.8x wins on raw matmuls?

Ports the Swift IntelligentRouter strategy to Python: EVERY Linear-layer
matmul (qkv/o projections, FFN gate/up/down or fc1/fc2) is split row-wise
across mx.cpu / mx.gpu streams — MLX executes independent work on different
streams concurrently — while attention math stays GPU-only (forceGPU, same
as Swift). Ratios are calibrated per distinct (M,K,N) shape by measuring the
ACTUAL concurrent split (contention-aware), not per-unit solo rates.

Three arms per (model, seq_len), each in its own subprocess, strictly
sequential:
  mlx_fp16      — fair baseline: run_mlx_* code path, FP16 weights
  fusion_nosplit— current FusionML (FP16 + mx.compile, no split)
  fusion_split  — FP16 + per-layer stream split (this experiment)

Correctness: split forward is checked against the GPU-only forward once per
config (max rel err reported). n=50, warmup=10.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

LLAMA = {"dim": 4096, "ffn_dim": 14336}
GPT2  = {"dim": 1600, "ffn_dim": 6400}

SEQ_LENS = [1024, 2048, 4096, 8192]
WARMUPS = 10
RUNS    = int(os.environ.get("PREFILL_RUNS", 50))
RATIO_CANDIDATES = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
CALIB_RUNS = 7

# fusion_dynamic gate parameters. First tuning pass (3/25/0.3) made two
# mistakes under thermal drift: 3-sample probation picked the wrong mode on
# noisy cells, and the 25-call probe cadence left the inactive-mode EMA so
# stale that a bad choice persisted for most of a 50-run window.
PROBATION_SAMPLES = 7    # per-mode samples to pick the starting mode (median)
PROBE_EVERY = 10         # re-probe the inactive mode every N calls
HYSTERESIS = 0.97        # inactive mode must be >3% faster to trigger a switch
EMA_ALPHA = 0.5


# =============================================================================
# Per-layer stream split (the ported Swift IntelligentRouter strategy)
# =============================================================================

def make_split_linear(ratios):
    import mlx.core as mx

    def split_linear(h, w):
        M, K = h.shape[0], h.shape[1]
        N = w.shape[1]
        cpu_ratio = ratios.get((M, K, N), 0.0)
        if cpu_ratio < 0.02:
            return h @ w
        # Eager boundary: MLX serializes cross-stream execution when the CPU
        # op's input is an unmaterialized GPU output inside the same eval
        # graph (stream_parallelism_probe.py: 0.66x lazy vs 1.34x eager).
        # Materializing the input first restores true CPU/GPU concurrency —
        # the same eager-per-layer structure Swift's IntelligentRouter uses.
        mx.eval(h)
        cpu_rows = int(M * cpu_ratio)
        c_cpu = mx.matmul(h[:cpu_rows], w, stream=mx.cpu)
        c_gpu = mx.matmul(h[cpu_rows:], w, stream=mx.gpu)
        return mx.concatenate([c_cpu, c_gpu], axis=0)

    return split_linear


def calibrate_shape(M, K, N):
    """Contention-aware ratio search for one matmul shape: measure the actual
    concurrent cpu/gpu stream split, pick the ratio with lowest median wall time."""
    import mlx.core as mx

    np.random.seed(7)
    a = mx.array((np.random.randn(M, K) * 0.02).astype(np.float32)).astype(mx.float16)
    w = mx.array((np.random.randn(K, N) * 0.02).astype(np.float32)).astype(mx.float16)
    mx.eval(a, w)

    results = {}
    for r in RATIO_CANDIDATES:
        cpu_rows = int(M * r)

        def run_once():
            if cpu_rows == 0:
                out = a @ w
            else:
                c_cpu = mx.matmul(a[:cpu_rows], w, stream=mx.cpu)
                c_gpu = mx.matmul(a[cpu_rows:], w, stream=mx.gpu)
                out = mx.concatenate([c_cpu, c_gpu], axis=0)
            mx.eval(out)

        run_once(); run_once()  # warm
        times = []
        for _ in range(CALIB_RUNS):
            t0 = time.perf_counter()
            run_once()
            times.append((time.perf_counter() - t0) * 1000.0)
        results[r] = float(np.median(times))

    best_ratio = min(results, key=results.get)
    return best_ratio, results


# =============================================================================
# Forward passes (FP16, mirror run_mlx_llama/run_mlx_gpt2 math exactly)
# =============================================================================

def llama_forward(x, weights, linear):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn

    D = x.shape[-1]
    h = mxf.layer_norm(x, weights['ln1_g'], weights['ln1_b'], 1e-5)
    q = linear(h, weights['w_q'])
    k = linear(h, weights['w_k'])
    v = linear(h, weights['w_v'])
    kT = mx.transpose(k)
    scale = 1.0 / (D ** 0.5)
    scores = (q @ kT) * scale                       # forceGPU (default stream)
    attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
    attended = attn @ v
    projected = linear(attended, weights['w_o'])
    h1 = x + projected

    h2 = mxf.layer_norm(h1, weights['ln2_g'], weights['ln2_b'], 1e-5)
    gate = linear(h2, weights['w_gate'])
    up = linear(h2, weights['w_up'])
    output = linear(mlx_nn.silu(gate) * up, weights['w_down'])
    return h1 + output


def gpt2_forward(x, weights, linear):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn

    D = x.shape[-1]
    h = mxf.layer_norm(x, weights['ln1_g'], weights['ln1_b'], 1e-5)
    q = linear(h, weights['w_q'])
    k = linear(h, weights['w_k'])
    v = linear(h, weights['w_v'])
    kT = mx.transpose(k)
    scale = 1.0 / (D ** 0.5)
    scores = (q @ kT) * scale
    attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
    attended = attn @ v
    projected = linear(attended, weights['w_o'])
    h1 = x + projected

    h2 = mxf.layer_norm(h1, weights['ln2_g'], weights['ln2_b'], 1e-5)
    fc1 = linear(h2, weights['w_fc1'])
    act = mlx_nn.gelu_approx(fc1)
    output = linear(act, weights['w_fc2'])
    return h1 + output


def build_weights(model, L):
    import mlx.core as mx
    cfg = LLAMA if model == "llama" else GPT2
    D, F = cfg["dim"], cfg["ffn_dim"]
    np.random.seed(42); scale = 0.02

    def _a(shape):
        return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(mx.float16)

    x = _a((L, D))
    weights = {
        'w_q': _a((D, D)), 'w_k': _a((D, D)), 'w_v': _a((D, D)), 'w_o': _a((D, D)),
        'ln1_g': mx.ones((D,), dtype=mx.float16), 'ln1_b': mx.zeros((D,), dtype=mx.float16),
        'ln2_g': mx.ones((D,), dtype=mx.float16), 'ln2_b': mx.zeros((D,), dtype=mx.float16),
    }
    if model == "llama":
        weights.update({'w_gate': _a((D, F)), 'w_up': _a((D, F)), 'w_down': _a((F, D))})
    else:
        weights.update({'w_fc1': _a((D, F)), 'w_fc2': _a((F, D))})
    mx.eval(x, *weights.values())
    return x, weights, D, F


class DynamicGate:
    """Block-level dynamic split/nosplit controller.

    Probation (during warmup) measures both modes and starts with the faster
    one. Every PROBE_EVERY calls, one call runs the inactive mode instead;
    EMAs of both modes are maintained and the gate switches when the inactive
    mode is faster past HYSTERESIS. Probe calls are real forward passes on the
    caller's timed path, so the benchmark's reported latency includes the full
    amortized cost of staying adaptive."""

    def __init__(self, fns):
        self.fns = fns  # {"split": fn, "nosplit": fn}
        self.ema = {}
        self.mode = None
        self.calls = 0
        self.switches = 0
        self.timeline = []  # (mode, ms) per call, for convergence analysis

    def _timed(self, mode):
        t0 = time.perf_counter()
        self.fns[mode]()
        dt = (time.perf_counter() - t0) * 1000.0
        prev = self.ema.get(mode)
        self.ema[mode] = dt if prev is None else (EMA_ALPHA * dt + (1 - EMA_ALPHA) * prev)
        self.timeline.append((mode, round(dt, 3)))
        return dt

    def probation(self):
        med = {}
        for mode in self.fns:
            ts = [self._timed(mode) for _ in range(PROBATION_SAMPLES)]
            med[mode] = float(np.median(ts))
            self.ema[mode] = med[mode]
        self.mode = min(med, key=med.get)

    def __call__(self):
        self.calls += 1
        others = [m for m in self.fns if m != self.mode]
        if others and self.calls % PROBE_EVERY == 0:
            # Round-robin probe of inactive modes
            probe = others[(self.calls // PROBE_EVERY) % len(others)]
            self._timed(probe)
            best_other = min(others, key=lambda m: self.ema.get(m, float("inf")))
            if self.ema[best_other] < HYSTERESIS * self.ema[self.mode]:
                self.mode = best_other
                self.switches += 1
        else:
            self._timed(self.mode)


# =============================================================================
# Subprocess worker
# =============================================================================

def run_worker(model, L, arm):
    import mlx.core as mx
    import resource

    x, weights, D, F = build_weights(model, L)
    forward = llama_forward if model == "llama" else gpt2_forward
    gpu_linear = lambda h, w: h @ w

    calib_info = None
    if arm == "mlx_fp16":
        # Fair baseline: identical math, plain MLX FP16, eager (matches
        # mlx_fp16_baseline_test methodology / run_mlx_* code path)
        fn = lambda: mx.eval(forward(x, weights, gpu_linear))

    elif arm == "fusion_nosplit":
        # Current FusionML: FP16 + mx.compile, no split
        wkeys = sorted(weights.keys())

        def _fwd(x_, *ws):
            wd = dict(zip(wkeys, ws))
            return forward(x_, wd, gpu_linear)

        compiled = mx.compile(_fwd)
        wvals = [weights[k] for k in wkeys]
        fn = lambda: mx.eval(compiled(x, *wvals))

    elif arm in ("fusion_split", "fusion_dynamic"):
        # Per-layer contention-aware stream split (uncompiled — mx.compile
        # does not support multi-stream graphs)
        shapes = [(L, D, D), (L, D, F), (L, F, D)]
        ratios, calib_detail = {}, {}
        for (M, K, N) in shapes:
            best, detail = calibrate_shape(M, K, N)
            ratios[(M, K, N)] = best
            calib_detail[f"{M}x{K}x{N}"] = {"best_cpu_ratio": best, "medians_ms": detail}
        calib_info = calib_detail

        split_linear = make_split_linear(ratios)
        split_fn = lambda: mx.eval(forward(x, weights, split_linear))

        # One-time correctness check vs GPU-only forward
        ref = forward(x, weights, gpu_linear)
        out = forward(x, weights, split_linear)
        mx.eval(ref, out)
        ref64 = np.array(ref.astype(mx.float32), copy=False).astype(np.float64)
        out64 = np.array(out.astype(mx.float32), copy=False).astype(np.float64)
        rel_err = float(np.max(np.abs(out64 - ref64)) / (np.max(np.abs(ref64)) + 1e-8))

        if arm == "fusion_split":
            fn = split_fn
        else:
            # Dynamic gate over THREE modes: split, compiled-nosplit, and plain
            # eager (identical to the mlx_fp16 baseline). Eager must be a gate
            # mode because mx.compile itself loses to eager MLX at large seq
            # (llama@8192: compiled ~0.94x — dynamic_convergence_test.json), so
            # a split/compiled-only gate has no true baseline-level floor.
            wkeys = sorted(weights.keys())

            def _fwd(x_, *ws):
                return forward(x_, dict(zip(wkeys, ws)), gpu_linear)

            compiled = mx.compile(_fwd)
            wvals = [weights[k] for k in wkeys]
            nosplit_fn = lambda: mx.eval(compiled(x, *wvals))
            eager_fn = lambda: mx.eval(forward(x, weights, gpu_linear))
            for _ in range(3):  # warm all paths before probation measures them
                split_fn(); nosplit_fn(); eager_fn()
            gate = DynamicGate({"split": split_fn, "nosplit": nosplit_fn, "eager": eager_fn})
            gate.probation()
            fn = gate
    else:
        raise ValueError(arm)

    for _ in range(WARMUPS):
        fn()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)

    n = len(times)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    out = {
        "mean": float(np.mean(times)), "std": float(np.std(times)),
        "median": float(np.median(times)), "ci95": float(1.96 * np.std(times) / np.sqrt(n)),
        "n_runs": n, "peak_mem_mb": peak_mb,
        "tokens_per_sec": float(L * 1000.0 / np.mean(times)),
    }
    if arm in ("fusion_split", "fusion_dynamic"):
        out["calibration"] = calib_info
        out["max_rel_err_vs_gpu_only"] = rel_err
    if arm == "fusion_dynamic":
        out["final_mode"] = fn.mode
        out["mode_switches"] = fn.switches
        out["ema_ms"] = {k: round(v, 3) for k, v in fn.ema.items()}
        out["timeline"] = fn.timeline
    return out


def run_sub(model, L, arm):
    cmd = [sys.executable, __file__, "--sub", "--model", model, "--seq", str(L), "--arm", arm]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python")) \
        + os.pathsep + os.path.dirname(__file__)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)
    if res.returncode != 0:
        print(f"\n  ⚠ {model} L={L} {arm}: rc={res.returncode} {res.stderr[-600:]}", file=sys.stderr)
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
    parser.add_argument("--seq", type=int)
    parser.add_argument("--arm", choices=["mlx_fp16", "fusion_nosplit", "fusion_split", "fusion_dynamic"])
    parser.add_argument("--models", default="llama,gpt2")
    parser.add_argument("--seqs", default=",".join(str(s) for s in SEQ_LENS))
    parser.add_argument("--arms", default="mlx_fp16,fusion_nosplit,fusion_split,fusion_dynamic",
                        help="comma-separated arms to run; existing results for other arms are preserved")
    parser.add_argument("--cooldown", type=float, default=8.0,
                        help="seconds of idle between subprocesses (fanless machines need 600+ after heavy cells)")
    args = parser.parse_args()

    if args.sub:
        print(json.dumps(run_worker(args.model, args.seq, args.arm)))
        return

    models = args.models.split(",")
    seqs = [int(s) for s in args.seqs.split(",")]
    arms = args.arms.split(",")

    print("=" * 78)
    print("  Prefill-Scale Benchmark — per-layer stream split vs fair MLX-FP16")
    print(f"  Runs: {RUNS}  Warmup: {WARMUPS}  Seq lengths: {seqs}  Arms: {arms}")
    print("=" * 78)

    from bench_hw import get_system_info
    sys_info = get_system_info()
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results", sys_info["cpu_slug"]))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prefill_scale_benchmark.json")
    print(f"  System: {sys_info['cpu']} {sys_info['memory']} "
          f"({'passively cooled' if sys_info['passively_cooled'] else 'actively cooled'}) → {sys_info['cpu_slug']}")

    # Merge into existing results so arms can be (re)run independently
    all_results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                all_results = json.load(f).get("results", {})
        except Exception:
            all_results = {}

    for model in models:
        name = "Llama-3-8B" if model == "llama" else "GPT-2 XL"
        all_results.setdefault(model, {})
        for L in seqs:
            print(f"\n▶ {name}  seq_len={L}")
            cell = all_results[model].setdefault(str(L), {})
            for arm in arms:
                print(f"   {arm:15s} ... ", end="", flush=True)
                time.sleep(args.cooldown)  # thermal cooldown between subprocesses
                s = run_sub(model, L, arm)
                cell[arm] = s
                if s:
                    extra = ""
                    if arm in ("fusion_split", "fusion_dynamic"):
                        extra = f"  rel_err={s['max_rel_err_vs_gpu_only']:.2e}"
                    if arm == "fusion_dynamic":
                        extra += f"  mode={s['final_mode']} switches={s['mode_switches']}"
                    print(f"{s['median']:8.2f} ms  ±{s['ci95']:.2f}{extra}")
                else:
                    print("FAILED")
            r = cell
            if r.get("mlx_fp16"):
                base = r["mlx_fp16"]["median"]
                parts = []
                for a in ["fusion_split", "fusion_dynamic", "fusion_nosplit"]:
                    if r.get(a):
                        parts.append(f"{a.replace('fusion_', '')} {base / r[a]['median']:.3f}x")
                print(f"   → vs MLX-FP16: " + "   ".join(parts))

            from bench_hw import get_power_state
            with open(out_path, "w") as f:
                json.dump({"warmups": WARMUPS, "runs": RUNS, "seq_lens": seqs,
                           "ratio_candidates": RATIO_CANDIDATES,
                           "environment": get_power_state(),
                           "results": all_results}, f, indent=2)

    print(f"\n💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
