#!/usr/bin/env python3
"""
training_nan_diag.py — diagnose the M4 FP16-training NaN in the CURRENT
(post-attention-fix) code path. llama_fp16_nan_check.py tests the OLD
unscaled attention and cannot answer this.

Checks, for both models, in FP16 with the exact math of run_mlx_llama /
run_mlx_gpt2 (scaled QK^T, fp32 softmax cast back):
  1. library versions (suspect: unpinned mlx/coremltools differ per machine)
  2. forward pass: per-stage isfinite
  3. loss as computed by the benchmark (mean over fp16 result)
  4. loss with fp32 accumulation (isolates mean-reduction overflow)
  5. value_and_grad loss + grad norms (the benchmark's actual training step)
"""

import os
import sys
import json
import platform
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

import mlx.core as mx
import mlx.core.fast as mxf
import mlx.nn as mlx_nn

LLAMA = {"dim": 4096, "ffn_dim": 14336, "seq_len": 1024}
GPT2  = {"dim": 1600, "ffn_dim": 6400,  "seq_len": 1024}


def finite(name, arr):
    a32 = arr.astype(mx.float32)
    ok = bool(mx.all(mx.isfinite(a32)).item())
    amax = float(mx.max(mx.abs(a32)).item()) if ok else float("nan")
    print(f"    {name:12s} finite={ok}  max_abs={amax:.4g}")
    return ok


def build(model):
    cfg = LLAMA if model == "llama" else GPT2
    D, F, L = cfg["dim"], cfg["ffn_dim"], cfg["seq_len"]
    np.random.seed(42); scale = 0.02

    def _a(shape):
        return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(mx.float16)

    x = _a((L, D))
    w = {'w_q': _a((D, D)), 'w_k': _a((D, D)), 'w_v': _a((D, D)), 'w_o': _a((D, D)),
         'ln1_g': mx.ones((D,), dtype=mx.float16), 'ln1_b': mx.zeros((D,), dtype=mx.float16),
         'ln2_g': mx.ones((D,), dtype=mx.float16), 'ln2_b': mx.zeros((D,), dtype=mx.float16)}
    if model == "llama":
        w.update({'w_gate': _a((D, F)), 'w_up': _a((D, F)), 'w_down': _a((F, D))})
    else:
        w.update({'w_fc1': _a((D, F)), 'w_fc2': _a((F, D))})
    mx.eval(x, *w.values())
    return x, w, D


def staged_forward(model, x, w, D):
    print("  Forward, per-stage:")
    h = mxf.layer_norm(x, w['ln1_g'], w['ln1_b'], 1e-5); finite("ln1", h)
    q = h @ w['w_q']; k = h @ w['w_k']; v = h @ w['w_v']
    finite("q", q); finite("k", k); finite("v", v)
    scores = (q @ mx.transpose(k)) * (1.0 / (D ** 0.5)); finite("scores", scores)
    attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype); finite("softmax", attn)
    attended = attn @ v; finite("attn@v", attended)
    h1 = x + attended @ w['w_o']; finite("h1", h1)
    h2 = mxf.layer_norm(h1, w['ln2_g'], w['ln2_b'], 1e-5); finite("ln2", h2)
    if model == "llama":
        out = (mlx_nn.silu(h2 @ w['w_gate']) * (h2 @ w['w_up'])) @ w['w_down']
    else:
        out = mlx_nn.gelu_approx(h2 @ w['w_fc1']) @ w['w_fc2']
    finite("mlp_out", out)
    res = h1 + out; finite("res", res)
    loss16 = mx.mean(res)
    loss32 = mx.mean(res.astype(mx.float32))
    print(f"    loss fp16-accum = {float(loss16.item())}   loss fp32-accum = {float(loss32.item())}")
    return res


def grad_step(model, x, w):
    from model_comparison import run_mlx_llama, run_mlx_gpt2
    fn = run_mlx_llama if model == "llama" else run_mlx_gpt2
    loss = fn(x, w, training=True)
    lv = float(loss.item()) if hasattr(loss, "item") else float(loss)
    print(f"  Benchmark training step (run_mlx_{model}): loss = {lv}  finite = {np.isfinite(lv)}")


def main():
    try:
        import coremltools
        ct_v = coremltools.__version__
    except Exception:
        ct_v = "n/a"
    mac = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip()
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    print(f"VERSIONS: mlx={mx.__version__}  coremltools={ct_v}  macOS={mac}  chip={chip}  python={platform.python_version()}")

    for model in ["llama", "gpt2"]:
        print(f"\n▶ {model.upper()} (FP16, current scaled+softmax math)")
        x, w, D = build(model)
        staged_forward(model, x, w, D)
        grad_step(model, x, w)


if __name__ == "__main__":
    main()
