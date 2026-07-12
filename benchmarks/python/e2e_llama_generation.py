#!/usr/bin/env python3
"""
e2e_llama_generation.py
=======================
End-to-end autoregressive generation benchmark on a Llama-3-8B decoder block.

Sweeps batch_sizes × seq_lens and reports for each (framework, batch, seq):
  - Prefill latency (ms)        — forward pass on full prompt
  - Generation tokens/sec       — batch_size * GEN_STEPS decode steps / elapsed
  - Peak unified memory (MB)    — process RSS after model loaded

Uses 2D tensors [batch*seq, dim] throughout (consistent with model_comparison.py).
Each framework benchmark runs in a separate subprocess to avoid memory contamination.
"""

import os
import sys
import re
import time
import gc
import json
import subprocess
import resource

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

# =============================================================================
# Configuration
# =============================================================================

DIM     = 4096
FFN_DIM = 14336

BATCH_SIZES = [1, 4, 8]
SEQ_LENS    = [512, 1024, 2048]
GEN_STEPS   = 50   # decode steps per config (kept lower to cap wall time)
PREFILL_REPS = 10  # repeated prefill runs; report median
WARMUP_REPS  = 3


# =============================================================================
# Hardware slug (shared with model_comparison.py)
# =============================================================================

def get_hw_slug():
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
        ("M1 Ultra", "32"), ("M2 Ultra", "32"), ("M3 Ultra", "36"), ("M4 Ultra", "64"),
        ("M3 Pro", "18"), ("M3 Max", "18"), ("M4 Pro", "20"), ("M4 Max", "32"),
    ]
    ane_cores = "16"
    for k, v in _ANE:
        if k in cpu:
            ane_cores = v
            break
    slug = f"{cpu}_{mem_gb}_{cpu_cores}CPU_{gpu_cores}GPU_{ane_cores}ANE".replace(" ", "_")
    return slug, cpu, mem_gb


# =============================================================================
# Subprocess worker — one (framework, batch, seq) per invocation
# =============================================================================

def _build_mlx_weights():
    import mlx.core as mx
    mx.random.seed(42)
    return {
        'w_q':    mx.random.normal((DIM, DIM), dtype=mx.float32) * 0.02,
        'w_k':    mx.random.normal((DIM, DIM), dtype=mx.float32) * 0.02,
        'w_v':    mx.random.normal((DIM, DIM), dtype=mx.float32) * 0.02,
        'w_o':    mx.random.normal((DIM, DIM), dtype=mx.float32) * 0.02,
        'w_gate': mx.random.normal((DIM, FFN_DIM), dtype=mx.float32) * 0.02,
        'w_up':   mx.random.normal((DIM, FFN_DIM), dtype=mx.float32) * 0.02,
        'w_down': mx.random.normal((FFN_DIM, DIM), dtype=mx.float32) * 0.02,
        'ln1_g':  mx.ones((DIM,)),  'ln1_b': mx.zeros((DIM,)),
        'ln2_g':  mx.ones((DIM,)),  'ln2_b': mx.zeros((DIM,)),
    }


def _mlx_forward(x, w):
    import mlx.core as mx

    def _ln(x, g, b, eps=1e-5):
        mu = mx.mean(x, axis=-1, keepdims=True)
        va = mx.var(x,  axis=-1, keepdims=True)
        return g * (x - mu) / mx.sqrt(va + eps) + b

    h  = _ln(x, w['ln1_g'], w['ln1_b'])
    q  = h @ w['w_q']; k = h @ w['w_k']; v = h @ w['w_v']
    s  = q @ mx.transpose(k); a = s @ v
    h1 = x + a @ w['w_o']

    h2   = _ln(h1, w['ln2_g'], w['ln2_b'])
    gate = h2 @ w['w_gate']
    out  = (gate * mx.sigmoid(gate)) * (h2 @ w['w_up'])
    return h1 + out @ w['w_down']


def run_mlx(batch, seq):
    import mlx.core as mx
    w = _build_mlx_weights()

    # Warmup
    for _ in range(WARMUP_REPS):
        mx.eval(_mlx_forward(mx.zeros((batch * seq, DIM)), w))

    # Prefill (median over PREFILL_REPS)
    x_pre = mx.array(np.zeros((batch * seq, DIM), dtype=np.float32))
    times = []
    for _ in range(PREFILL_REPS):
        t0 = time.perf_counter()
        mx.eval(_mlx_forward(x_pre, w))
        times.append((time.perf_counter() - t0) * 1000.0)
    prefill_ms = float(np.median(times))

    # Decode — batch_size × 1 token per step
    x_dec = mx.array(np.zeros((batch, DIM), dtype=np.float32))
    t0 = time.perf_counter()
    for _ in range(GEN_STEPS):
        mx.eval(_mlx_forward(x_dec, w))
    gen_tps = batch * GEN_STEPS / (time.perf_counter() - t0)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    return {"prefill_ms": prefill_ms, "gen_tps": gen_tps, "peak_mem_mb": peak_mb}


def run_pytorch(batch, seq):
    import torch
    dev = "mps"
    np.random.seed(42); s = 0.02
    w = {
        'w_q':    torch.from_numpy(np.random.randn(DIM, DIM).astype(np.float32) * s).to(dev),
        'w_k':    torch.from_numpy(np.random.randn(DIM, DIM).astype(np.float32) * s).to(dev),
        'w_v':    torch.from_numpy(np.random.randn(DIM, DIM).astype(np.float32) * s).to(dev),
        'w_o':    torch.from_numpy(np.random.randn(DIM, DIM).astype(np.float32) * s).to(dev),
        'w_gate': torch.from_numpy(np.random.randn(DIM, FFN_DIM).astype(np.float32) * s).to(dev),
        'w_up':   torch.from_numpy(np.random.randn(DIM, FFN_DIM).astype(np.float32) * s).to(dev),
        'w_down': torch.from_numpy(np.random.randn(FFN_DIM, DIM).astype(np.float32) * s).to(dev),
        'ln1_g':  torch.ones(DIM, device=dev),  'ln1_b': torch.zeros(DIM, device=dev),
        'ln2_g':  torch.ones(DIM, device=dev),  'ln2_b': torch.zeros(DIM, device=dev),
    }

    def fwd(x):
        import torch.nn.functional as F
        h  = F.layer_norm(x, (DIM,), w['ln1_g'], w['ln1_b'])
        q  = h @ w['w_q']; k = h @ w['w_k']; v = h @ w['w_v']
        h1 = x + (q @ k.T @ v) @ w['w_o']
        h2   = F.layer_norm(h1, (DIM,), w['ln2_g'], w['ln2_b'])
        gate = h2 @ w['w_gate']
        out  = F.silu(gate) * (h2 @ w['w_up'])
        return h1 + out @ w['w_down']

    with torch.no_grad():
        for _ in range(WARMUP_REPS):
            _ = fwd(torch.zeros(batch * seq, DIM, device=dev)); torch.mps.synchronize()

        x_pre = torch.zeros(batch * seq, DIM, device=dev)
        times = []
        for _ in range(PREFILL_REPS):
            t0 = time.perf_counter()
            fwd(x_pre); torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        prefill_ms = float(np.median(times))

        x_dec = torch.zeros(batch, DIM, device=dev)
        t0 = time.perf_counter()
        for _ in range(GEN_STEPS):
            fwd(x_dec); torch.mps.synchronize()
        gen_tps = batch * GEN_STEPS / (time.perf_counter() - t0)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    return {"prefill_ms": prefill_ms, "gen_tps": gen_tps, "peak_mem_mb": peak_mb}


def run_fusionml(batch, seq):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn

    np.random.seed(42); s = 0.02
    # Pre-cast weights to FP16 once — avoids per-call cast overhead that
    # dominates at decode sizes ([batch, 4096]) even after mx.compile tracing.
    w = {
        'w_q':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
        'w_k':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
        'w_v':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
        'w_o':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
        'w_gate': mx.random.normal((DIM, FFN_DIM), dtype=mx.float16) * 0.02,
        'w_up':   mx.random.normal((DIM, FFN_DIM), dtype=mx.float16) * 0.02,
        'w_down': mx.random.normal((FFN_DIM, DIM), dtype=mx.float16) * 0.02,
        'ln1_g':  mx.ones((DIM,), dtype=mx.float16),
        'ln1_b':  mx.zeros((DIM,), dtype=mx.float16),
        'ln2_g':  mx.ones((DIM,), dtype=mx.float16),
        'ln2_b':  mx.zeros((DIM,), dtype=mx.float16),
    }
    mx.eval(*w.values())

    # Compiled FP16 forward — weights arrive already FP16, only input cast needed.
    def _fwd(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b):
        x16 = x.astype(mx.float16)
        h   = mxf.layer_norm(x16, ln1_g, ln1_b, 1e-5)
        q   = h @ w_q; k = h @ w_k; v = h @ w_v
        h1  = x16 + (q @ mx.transpose(k) @ v) @ w_o
        h2  = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        gate = h2 @ w_gate
        out  = (mlx_nn.silu(gate) * (h2 @ w_up)) @ w_down
        return (h1 + out).astype(mx.float32)

    compiled = mx.compile(_fwd)

    def fwd(x):
        res = compiled(x, w['w_q'], w['w_k'], w['w_v'], w['w_o'],
                       w['w_gate'], w['w_up'], w['w_down'],
                       w['ln1_g'], w['ln1_b'], w['ln2_g'], w['ln2_b'])
        mx.eval(res)
        return res

    # Warm up prefill shape first, then decode shape separately.
    # mx.compile is shape-specific — missing decode warmup forces retrace on
    # the first timed decode call, inflating gen_tps measurement.
    for _ in range(WARMUP_REPS):
        fwd(mx.zeros((batch * seq, DIM)))
    x_dec_warm = mx.zeros((batch, DIM))
    for _ in range(WARMUP_REPS):
        fwd(x_dec_warm)

    x_pre = mx.zeros((batch * seq, DIM))
    times = []
    for _ in range(PREFILL_REPS):
        t0 = time.perf_counter()
        fwd(x_pre)
        times.append((time.perf_counter() - t0) * 1000.0)
    prefill_ms = float(np.median(times))

    x_dec = mx.zeros((batch, DIM))
    t0 = time.perf_counter()
    for _ in range(GEN_STEPS):
        fwd(x_dec)
    gen_tps = batch * GEN_STEPS / (time.perf_counter() - t0)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    return {"prefill_ms": prefill_ms, "gen_tps": gen_tps, "peak_mem_mb": peak_mb}


def run_mlx_fp16(batch, seq):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn

    np.random.seed(42); s = 0.02
    w = {
        'w_q':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
        'w_k':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
        'w_v':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
        'w_o':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
        'w_gate': mx.random.normal((DIM, FFN_DIM), dtype=mx.float16) * 0.02,
        'w_up':   mx.random.normal((DIM, FFN_DIM), dtype=mx.float16) * 0.02,
        'w_down': mx.random.normal((FFN_DIM, DIM), dtype=mx.float16) * 0.02,
        'ln1_g':  mx.ones((DIM,), dtype=mx.float16),
        'ln1_b':  mx.zeros((DIM,), dtype=mx.float16),
        'ln2_g':  mx.ones((DIM,), dtype=mx.float16),
        'ln2_b':  mx.zeros((DIM,), dtype=mx.float16),
    }
    mx.eval(*w.values())

    def _fwd(x):
        x16 = x.astype(mx.float16)
        h   = mxf.layer_norm(x16, w['ln1_g'], w['ln1_b'], 1e-5)
        q   = h @ w['w_q']; k = h @ w['w_k']; v = h @ w['w_v']
        h1  = x16 + (q @ mx.transpose(k) @ v) @ w['w_o']
        h2  = mxf.layer_norm(h1, w['ln2_g'], w['ln2_b'], 1e-5)
        gate = h2 @ w['w_gate']
        out  = (mlx_nn.silu(gate) * (h2 @ w['w_up'])) @ w['w_down']
        return (h1 + out).astype(mx.float32)

    # Warmup
    for _ in range(WARMUP_REPS):
        mx.eval(_fwd(mx.zeros((batch * seq, DIM))))

    # Prefill
    x_pre = mx.zeros((batch * seq, DIM))
    times = []
    for _ in range(PREFILL_REPS):
        t0 = time.perf_counter()
        mx.eval(_fwd(x_pre))
        times.append((time.perf_counter() - t0) * 1000.0)
    prefill_ms = float(np.median(times))

    # Decode
    x_dec = mx.zeros((batch, DIM))
    t0 = time.perf_counter()
    for _ in range(GEN_STEPS):
        mx.eval(_fwd(x_dec))
    gen_tps = batch * GEN_STEPS / (time.perf_counter() - t0)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    return {"prefill_ms": prefill_ms, "gen_tps": gen_tps, "peak_mem_mb": peak_mb}


def run_mlx_fp16_compiled(batch, seq):
    return run_fusionml(batch, seq)


# =============================================================================
# Subprocess launcher — isolate each (fw, batch, seq) to avoid memory bleed
# =============================================================================

def run_sub(fw, batch, seq):
    cmd = [sys.executable, __file__, "--sub", "--fw", fw,
           "--batch", str(batch), "--seq", str(seq)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..", "python")
    )
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if res.returncode != 0:
        print(f"  ⚠ {fw} batch={batch} seq={seq}: {res.stderr[:300]}", file=sys.stderr)
        return None
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub",   action="store_true")
    parser.add_argument("--fw",    type=str, choices=["mlx", "pytorch", "fusionml", "mlx_fp16", "mlx_fp16_compiled"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq",   type=int, default=512)
    args = parser.parse_args()

    if args.sub:
        fn = {
            "mlx": run_mlx,
            "pytorch": run_pytorch,
            "fusionml": run_fusionml,
            "mlx_fp16": run_mlx_fp16,
            "mlx_fp16_compiled": run_mlx_fp16_compiled
        }[args.fw]
        result = fn(args.batch, args.seq)
        print(json.dumps(result))
        return

    # ── Orchestrator ──────────────────────────────────────────────────────────
    hw_slug, cpu_name, mem_gb = get_hw_slug()

    print("=" * 95)
    print("  E2E Llama-3-8B Block — Prefill & Generation Sweep (Full Grid)")
    print(f"  Hardware: {cpu_name} ({mem_gb})")
    print(f"  Batch sizes: {BATCH_SIZES}  |  Seq lengths: {SEQ_LENS}")
    print(f"  Prefill reps: {PREFILL_REPS}  |  Decode steps: {GEN_STEPS}")
    print("=" * 95)

    frameworks = ["MLX FP32 Eager", "MLX FP16 Eager", "MLX FP16 Compiled", "PyTorch (MPS)", "FusionML (Compiled)"]
    fw_args    = {
        "MLX FP32 Eager": "mlx",
        "MLX FP16 Eager": "mlx_fp16",
        "MLX FP16 Compiled": "mlx_fp16_compiled",
        "PyTorch (MPS)": "pytorch",
        "FusionML (Compiled)": "fusionml"
    }

    # results[fw][batch][seq] = {prefill_ms, gen_tps, peak_mem_mb}
    results = {fw: {} for fw in frameworks}

    for fw in frameworks:
        print(f"\n▶ {fw}")
        results[fw] = {}
        for batch in BATCH_SIZES:
            results[fw][batch] = {}
            for seq in SEQ_LENS:
                print(f"   batch={batch} seq={seq} ...", end=" ", flush=True)
                r = run_sub(fw_args[fw], batch, seq)
                if r:
                    results[fw][batch][seq] = r
                    print(f"prefill={r['prefill_ms']:.1f}ms  "
                          f"gen={r['gen_tps']:.0f} tok/s  "
                          f"mem={r['peak_mem_mb']:.0f}MB")
                else:
                    results[fw][batch][seq] = None
                    print("FAILED")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("  PREFILL LATENCY (ms, median) — lower is better")
    print(f"{'='*110}")
    hdr = f"  {'Batch':>5} {'Seq':>6} | {'MLX FP32':>10} {'MLX FP16':>10} {'MLX Comp':>10} {'PyTorch':>10} {'FusionML':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for batch in BATCH_SIZES:
        for seq in SEQ_LENS:
            m32_r = results["MLX FP32 Eager"][batch].get(seq)
            m16_r = results["MLX FP16 Eager"][batch].get(seq)
            mcp_r = results["MLX FP16 Compiled"][batch].get(seq)
            pt_r  = results["PyTorch (MPS)"][batch].get(seq)
            fml_r = results["FusionML (Compiled)"][batch].get(seq)
            
            m32_v = m32_r["prefill_ms"] if m32_r else None
            m16_v = m16_r["prefill_ms"] if m16_r else None
            mcp_v = mcp_r["prefill_ms"] if mcp_r else None
            pt_v  = pt_r["prefill_ms"]  if pt_r  else None
            fml_v = fml_r["prefill_ms"] if fml_r else None
            
            print(f"  {batch:>5} {seq:>6} | "
                  f"{(f'{m32_v:.1f}' if m32_v else 'N/A'):>10} "
                  f"{(f'{m16_v:.1f}' if m16_v else 'N/A'):>10} "
                  f"{(f'{mcp_v:.1f}' if mcp_v else 'N/A'):>10} "
                  f"{(f'{pt_v:.1f}'  if pt_v  else 'N/A'):>10} "
                  f"{(f'{fml_v:.1f}' if fml_v else 'N/A'):>10}")

    print(f"\n{'='*110}")
    print("  GENERATION TOKENS/SEC — higher is better")
    print(f"{'='*110}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for batch in BATCH_SIZES:
        for seq in SEQ_LENS:
            m32_r = results["MLX FP32 Eager"][batch].get(seq)
            m16_r = results["MLX FP16 Eager"][batch].get(seq)
            mcp_r = results["MLX FP16 Compiled"][batch].get(seq)
            pt_r  = results["PyTorch (MPS)"][batch].get(seq)
            fml_r = results["FusionML (Compiled)"][batch].get(seq)
            
            m32_v = m32_r["gen_tps"] if m32_r else None
            m16_v = m16_r["gen_tps"] if m16_r else None
            mcp_v = mcp_r["gen_tps"] if mcp_r else None
            pt_v  = pt_r["gen_tps"]  if pt_r  else None
            fml_v = fml_r["gen_tps"] if fml_r else None
            
            print(f"  {batch:>5} {seq:>6} | "
                  f"{(f'{m32_v:.0f}' if m32_v else 'N/A'):>10} "
                  f"{(f'{m16_v:.0f}' if m16_v else 'N/A'):>10} "
                  f"{(f'{mcp_v:.0f}' if mcp_v else 'N/A'):>10} "
                  f"{(f'{pt_v:.0f}'  if pt_v  else 'N/A'):>10} "
                  f"{(f'{fml_v:.0f}' if fml_v else 'N/A'):>10}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../results", hw_slug)
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e2e_llama_generation.json")
    payload = {
        "hardware_slug": hw_slug,
        "batch_sizes":   BATCH_SIZES,
        "seq_lens":      SEQ_LENS,
        "gen_steps":     GEN_STEPS,
        "prefill_reps":  PREFILL_REPS,
        "results":       results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n💾 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
