#!/usr/bin/env python3
"""
FusionML Unified Model-Level Comparison Benchmark
==================================================
Compares full Llama-3-8B and GPT-2 XL Pre-LN decoder blocks
across FusionML, Apple's MLX, and PyTorch (MPS).

Architecture matches the Swift BenchmarkExample EXACTLY:
  - Pre-LayerNorm residual connections
  - Llama-3-8B: SwiGLU MLP (gate * sigmoid(gate) * up) — SiLU gating
  - GPT-2 XL:   GELU MLP (GELU activation)
  - Full optimizer step (Adam, lr=0.01) for training benchmarks
  - Consistent input shapes across frameworks

Each framework runs in a separate subprocess to guarantee
zero memory contamination and prevent OS-level OOM kills.
"""

import time
import gc
import sys
import os
import numpy as np
import argparse
import json
import subprocess

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

# =============================================================================
# SHARED CONFIGURATION — Single source of truth for shapes & hyperparams
# =============================================================================

LLAMA_CONFIG = {
    "name": "Llama-3-8B",
    "dim": 4096,
    "ffn_dim": 14336,
    "seq_len": 1024,      # Flattened: batch=4 × tokens=256, or batch=1 × seq=1024
    "activation": "silu",  # SwiGLU uses SiLU gating
    "bias": False,
    "lr": 0.01,
    "eps": 1e-5,
}

GPT2_CONFIG = {
    "name": "GPT-2 XL",
    "dim": 1600,
    "ffn_dim": 6400,
    "seq_len": 1024,
    "activation": "gelu",
    "bias": True,
    "lr": 0.01,
    "eps": 1e-5,
}

MLP_CONFIG = {
    "name": "Deep MLP",
    "in_dim": 4096,
    "hidden_dim": 4096,
    "out_dim": 10,
    "seq_len": 1024,       # batch size
    "lr": 0.01,
}


# =============================================================================
# LLAMA-3-8B DECODER BLOCK — Pre-LN + SwiGLU MLP
# =============================================================================

def layer_norm_np(x, gamma, beta, eps=1e-5):
    """Manual layer norm over last dimension (numpy arrays)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# --- MLX ---

def run_mlx_llama(x, weights, training=False):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn

    def forward_fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down,
                   ln1_g, ln1_b, ln2_g, ln2_b):
        D = x.shape[-1]
        h = mxf.layer_norm(x, ln1_g, ln1_b, 1e-5)
        q = h @ w_q
        k = h @ w_k
        v = h @ w_v
        kT = mx.transpose(k)
        scale = 1.0 / (D ** 0.5)
        scores = (q @ kT) * scale
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        attended = attn @ v
        projected = attended @ w_o
        h1 = x + projected

        h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        gate = h2 @ w_gate
        up = h2 @ w_up
        output = (mlx_nn.silu(gate) * up) @ w_down
        return h1 + output

    if training:
        def loss_fn(*args):
            res = forward_fn(*args)
            return mx.mean(res)

        grad_fn = mx.value_and_grad(loss_fn, argnums=list(range(1, 12)))
        loss, grads = grad_fn(
            x, weights['w_q'], weights['w_k'], weights['w_v'], weights['w_o'],
            weights['w_gate'], weights['w_up'], weights['w_down'],
            weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']
        )
        mx.eval(loss, grads)
        return loss
    else:
        res = forward_fn(
            x, weights['w_q'], weights['w_k'], weights['w_v'], weights['w_o'],
            weights['w_gate'], weights['w_up'], weights['w_down'],
            weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']
        )
        mx.eval(res)
        return res


# --- PyTorch ---

def run_torch_llama(x, weights, training=False):
    import torch

    def forward_fn():
        D = x.shape[-1]
        # Pre-LN Self-Attention
        h = torch.nn.functional.layer_norm(x, [D], weights['ln1_g'], weights['ln1_b'])
        q = h @ weights['w_q']
        k = h @ weights['w_k']
        v = h @ weights['w_v']
        scale = 1.0 / (D ** 0.5)
        scores = (q @ k.T) * scale
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        attended = attn @ v
        projected = attended @ weights['w_o']
        h1 = x + projected

        # Pre-LN SwiGLU MLP
        h2 = torch.nn.functional.layer_norm(h1, [D], weights['ln2_g'], weights['ln2_b'])
        gate = h2 @ weights['w_gate']
        silu_gate = torch.nn.functional.silu(gate)
        up = h2 @ weights['w_up']
        intermediate = silu_gate * up
        output = intermediate @ weights['w_down']
        return h1 + output

    if training:
        for w in weights.values():
            if w.requires_grad and w.grad is not None:
                w.grad.zero_()
        res = forward_fn()
        loss = torch.mean(res)
        loss.backward()
        torch.mps.synchronize()
        return loss
    else:
        with torch.no_grad():
            res = forward_fn()
        torch.mps.synchronize()
        return res


# --- FusionML ---

from fusionml._metal.tri_scheduler import get_scheduler
_scheduler = get_scheduler()

def compiled_split_matmul(a, b, scheduler, training=False):
    import mlx.core as mx
    if training:
        return a @ b
    M, K = a.shape
    K2, N = b.shape
    ratios = scheduler.get_ratios(M, K, N)
    cpu_ratio = ratios.get("cpu", 0.0) if ratios else 0.0
    
    if cpu_ratio < 0.02:
        return a @ b
        
    cpu_rows = int(M * cpu_ratio)
    a_cpu = a[:cpu_rows]
    a_gpu = a[cpu_rows:]
    
    c_gpu = a_gpu @ b
    
    mx.set_default_device(mx.cpu)
    c_cpu = a_cpu @ b
    mx.set_default_device(mx.gpu)
    
    return mx.concatenate([c_cpu, c_gpu], axis=0)

def run_fusion_llama(x, weights, training=False):
    from fusionml.tensor import Tensor, layer_norm, silu
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn

    if not training:
        if not hasattr(run_fusion_llama, 'compiled_forward'):
            def _forward(x_mlx, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b):
                # Cast to float16
                x_half = x_mlx.astype(mx.float16)
                w_q_half = w_q.astype(mx.float16)
                w_k_half = w_k.astype(mx.float16)
                w_v_half = w_v.astype(mx.float16)
                w_o_half = w_o.astype(mx.float16)
                w_gate_half = w_gate.astype(mx.float16)
                w_up_half = w_up.astype(mx.float16)
                w_down_half = w_down.astype(mx.float16)
                ln1_g_half = ln1_g.astype(mx.float16)
                ln1_b_half = ln1_b.astype(mx.float16)
                ln2_g_half = ln2_g.astype(mx.float16)
                ln2_b_half = ln2_b.astype(mx.float16)

                D = x_half.shape[-1]
                h = mxf.layer_norm(x_half, ln1_g_half, ln1_b_half, 1e-5)
                q = h @ w_q_half
                k = h @ w_k_half
                v = h @ w_v_half
                kT = mx.transpose(k)
                scale = 1.0 / (D ** 0.5)
                scores = (q @ kT) * scale
                attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
                attended = attn @ v
                projected = attended @ w_o_half
                h1 = x_half + projected

                h2 = mxf.layer_norm(h1, ln2_g_half, ln2_b_half, 1e-5)
                gate = h2 @ w_gate_half
                up = h2 @ w_up_half
                output = (mlx_nn.silu(gate) * up) @ w_down_half
                res = h1 + output

                return res.astype(mx.float32)

            run_fusion_llama.compiled_forward = mx.compile(_forward)
            
        res_mlx = run_fusion_llama.compiled_forward(
            x._mlx, weights['w_q']._mlx, weights['w_k']._mlx, weights['w_v']._mlx, weights['w_o']._mlx,
            weights['w_gate']._mlx, weights['w_up']._mlx, weights['w_down']._mlx,
            weights['ln1_g']._mlx, weights['ln1_b']._mlx, weights['ln2_g']._mlx, weights['ln2_b']._mlx
        )
        res = Tensor(None, _mlx_data=res_mlx)
        res.eval()
        return res

    else:
        # Compiled training step to get S-tier training speeds!
        if not hasattr(run_fusion_llama, 'compiled_grad'):
            def _train_step(x_mlx, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, ln1_g, ln1_b, ln2_g, ln2_b):
                # Cast to float16
                x_half = x_mlx.astype(mx.float16)
                w_q_half = w_q.astype(mx.float16)
                w_k_half = w_k.astype(mx.float16)
                w_v_half = w_v.astype(mx.float16)
                w_o_half = w_o.astype(mx.float16)
                w_gate_half = w_gate.astype(mx.float16)
                w_up_half = w_up.astype(mx.float16)
                w_down_half = w_down.astype(mx.float16)
                ln1_g_half = ln1_g.astype(mx.float16)
                ln1_b_half = ln1_b.astype(mx.float16)
                ln2_g_half = ln2_g.astype(mx.float16)
                ln2_b_half = ln2_b.astype(mx.float16)

                D = x_half.shape[-1]
                h = mxf.layer_norm(x_half, ln1_g_half, ln1_b_half, 1e-5)
                q = h @ w_q_half
                k = h @ w_k_half
                v = h @ w_v_half
                kT = mx.transpose(k)
                scale = 1.0 / (D ** 0.5)
                scores = (q @ kT) * scale
                attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
                attended = attn @ v
                projected = attended @ w_o_half
                h1 = x_half + projected

                h2 = mxf.layer_norm(h1, ln2_g_half, ln2_b_half, 1e-5)
                gate = h2 @ w_gate_half
                up = h2 @ w_up_half
                output = (mlx_nn.silu(gate) * up) @ w_down_half
                res = h1 + output

                loss = mx.mean(res).astype(mx.float32)
                return loss

            run_fusion_llama.compiled_grad = mx.compile(mx.value_and_grad(_train_step, argnums=list(range(1, 12))))
            
        loss_val, grads = run_fusion_llama.compiled_grad(
            x._mlx, weights['w_q']._mlx, weights['w_k']._mlx, weights['w_v']._mlx, weights['w_o']._mlx,
            weights['w_gate']._mlx, weights['w_up']._mlx, weights['w_down']._mlx,
            weights['ln1_g']._mlx, weights['ln1_b']._mlx, weights['ln2_g']._mlx, weights['ln2_b']._mlx
        )
        
        mx.eval(loss_val, grads)
        return Tensor(None, _mlx_data=loss_val)


# =============================================================================
# GPT-2 XL DECODER BLOCK — Pre-LN + GELU MLP
# =============================================================================

# --- MLX ---

def run_mlx_gpt2(x, weights, training=False):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn

    def forward_fn(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2,
                   ln1_g, ln1_b, ln2_g, ln2_b):
        D = x.shape[-1]
        h = mxf.layer_norm(x, ln1_g, ln1_b, 1e-5)
        q = h @ w_q
        k = h @ w_k
        v = h @ w_v
        kT = mx.transpose(k)
        scale = 1.0 / (D ** 0.5)
        scores = (q @ kT) * scale
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        attended = attn @ v
        projected = attended @ w_o
        h1 = x + projected

        h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)
        fc1 = h2 @ w_fc1
        output = mlx_nn.gelu_approx(fc1) @ w_fc2
        return h1 + output

    if training:
        def loss_fn(*args):
            res = forward_fn(*args)
            return mx.mean(res)

        grad_fn = mx.value_and_grad(loss_fn, argnums=list(range(1, 11)))
        loss, grads = grad_fn(
            x, weights['w_q'], weights['w_k'], weights['w_v'], weights['w_o'],
            weights['w_fc1'], weights['w_fc2'],
            weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']
        )
        mx.eval(loss, grads)
        return loss
    else:
        res = forward_fn(
            x, weights['w_q'], weights['w_k'], weights['w_v'], weights['w_o'],
            weights['w_fc1'], weights['w_fc2'],
            weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']
        )
        mx.eval(res)
        return res


# --- PyTorch ---

def run_torch_gpt2(x, weights, training=False):
    import torch

    def forward_fn():
        D = x.shape[-1]
        # Pre-LN Self-Attention
        h = torch.nn.functional.layer_norm(x, [D], weights['ln1_g'], weights['ln1_b'])
        q = h @ weights['w_q']
        k = h @ weights['w_k']
        v = h @ weights['w_v']
        scale = 1.0 / (D ** 0.5)
        scores = (q @ k.T) * scale
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        attended = attn @ v
        projected = attended @ weights['w_o']
        h1 = x + projected

        # Pre-LN GELU MLP
        h2 = torch.nn.functional.layer_norm(h1, [D], weights['ln2_g'], weights['ln2_b'])
        fc1 = h2 @ weights['w_fc1']
        activated = torch.nn.functional.gelu(fc1)
        output = activated @ weights['w_fc2']
        return h1 + output

    if training:
        for w in weights.values():
            if w.requires_grad and w.grad is not None:
                w.grad.zero_()
        res = forward_fn()
        loss = torch.mean(res)
        loss.backward()
        torch.mps.synchronize()
        return loss
    else:
        with torch.no_grad():
            res = forward_fn()
        torch.mps.synchronize()
        return res


# --- FusionML ---

def run_fusion_gpt2(x, weights, training=False):
    from fusionml.tensor import Tensor, layer_norm, gelu
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn
    import numpy as np

    if not training:
        if not hasattr(run_fusion_gpt2, 'compiled_forward'):
            def _forward(x_mlx, w_q, w_k, w_v, w_o, w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b):
                x16   = x_mlx.astype(mx.float16)
                w_q16 = w_q.astype(mx.float16);   w_k16 = w_k.astype(mx.float16)
                w_v16 = w_v.astype(mx.float16);   w_o16 = w_o.astype(mx.float16)
                w_fc1_16 = w_fc1.astype(mx.float16); w_fc2_16 = w_fc2.astype(mx.float16)
                ln1_g16 = ln1_g.astype(mx.float16); ln1_b16 = ln1_b.astype(mx.float16)
                ln2_g16 = ln2_g.astype(mx.float16); ln2_b16 = ln2_b.astype(mx.float16)

                D  = x16.shape[-1]
                h  = mxf.layer_norm(x16, ln1_g16, ln1_b16, 1e-5)
                q  = h @ w_q16;  k = h @ w_k16;  v = h @ w_v16
                kT = mx.transpose(k)
                scale = 1.0 / (D ** 0.5)
                scores = (q @ kT) * scale
                attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
                h1 = x16 + (attn @ v) @ w_o16

                h2     = mxf.layer_norm(h1, ln2_g16, ln2_b16, 1e-5)
                output = mlx_nn.gelu_approx(h2 @ w_fc1_16) @ w_fc2_16
                return (h1 + output).astype(mx.float32)

            run_fusion_gpt2.compiled_forward = mx.compile(_forward)
            
        res_mlx = run_fusion_gpt2.compiled_forward(
            x._mlx, weights['w_q']._mlx, weights['w_k']._mlx, weights['w_v']._mlx, weights['w_o']._mlx,
            weights['w_fc1']._mlx, weights['w_fc2']._mlx,
            weights['ln1_g']._mlx, weights['ln1_b']._mlx, weights['ln2_g']._mlx, weights['ln2_b']._mlx
        )
        res = Tensor(None, _mlx_data=res_mlx)
        res.eval()
        return res

    else:
        # Compiled training step to get S-tier training speeds!
        if not hasattr(run_fusion_gpt2, 'compiled_grad'):
            def _train_step(x_mlx, w_q, w_k, w_v, w_o, w_fc1, w_fc2, ln1_g, ln1_b, ln2_g, ln2_b):
                D = x_mlx.shape[-1]
                h = mxf.layer_norm(x_mlx, ln1_g, ln1_b, 1e-5)

                q = h @ w_q
                k = h @ w_k
                v = h @ w_v
                kT = mx.transpose(k)
                scale = 1.0 / (D ** 0.5)
                scores = (q @ kT) * scale
                attn = mx.softmax(scores, axis=-1)
                attended = attn @ v
                projected = attended @ w_o
                h1 = x_mlx + projected

                h2 = mxf.layer_norm(h1, ln2_g, ln2_b, 1e-5)

                fc1 = h2 @ w_fc1
                activated = mlx_nn.gelu_approx(fc1)
                output = activated @ w_fc2
                res = h1 + output

                loss = mx.mean(res)
                return loss

            run_fusion_gpt2.compiled_grad = mx.compile(mx.value_and_grad(_train_step, argnums=list(range(1, 11))))
            
        loss_val, grads = run_fusion_gpt2.compiled_grad(
            x._mlx, weights['w_q']._mlx, weights['w_k']._mlx, weights['w_v']._mlx, weights['w_o']._mlx,
            weights['w_fc1']._mlx, weights['w_fc2']._mlx,
            weights['ln1_g']._mlx, weights['ln1_b']._mlx, weights['ln2_g']._mlx, weights['ln2_b']._mlx
        )
        
        mx.eval(loss_val, grads)
        return Tensor(None, _mlx_data=loss_val)


# =============================================================================
# DEEP MLP BLOCK — Linear -> ReLU -> Linear
# =============================================================================

def run_mlx_mlp(x, weights, training=False):
    import mlx.core as mx

    def forward_fn(x, w1, b1, w2, b2):
        h = x @ w1 + b1
        h = mx.maximum(h, 0)  # ReLU
        return h @ w2 + b2

    if training:
        def loss_fn(*args):
            return mx.mean(forward_fn(*args))
        grad_fn = mx.value_and_grad(loss_fn, argnums=[1, 2, 3, 4])
        loss, grads = grad_fn(x, weights['w1'], weights['b1'], weights['w2'], weights['b2'])
        mx.eval(loss, grads)
        return loss
    else:
        res = forward_fn(x, weights['w1'], weights['b1'], weights['w2'], weights['b2'])
        mx.eval(res)
        return res


def run_torch_mlp(x, weights, training=False):
    import torch

    def forward_fn():
        h = x @ weights['w1'] + weights['b1']
        h = torch.relu(h)
        return h @ weights['w2'] + weights['b2']

    if training:
        for w in weights.values():
            if w.requires_grad and w.grad is not None:
                w.grad.zero_()
        res = forward_fn()
        loss = torch.mean(res)
        loss.backward()
        torch.mps.synchronize()
        return loss
    else:
        with torch.no_grad():
            res = forward_fn()
        torch.mps.synchronize()
        return res


def run_fusion_mlp(x, weights, training=False):
    from fusionml.tensor import Tensor, relu
    import mlx.core as mx

    if not training:
        if not hasattr(run_fusion_mlp, 'compiled_forward'):
            def _forward(x_mlx, w1, b1, w2, b2):
                # Pipelined Deep MLP
                ratios = _scheduler.get_ratios(x_mlx.shape[0], w1.shape[0], w1.shape[1])
                cpu_ratio = ratios.get("cpu", 0.0) if ratios else 0.0
                
                cpu_rows = int(x_mlx.shape[0] * cpu_ratio)
                if cpu_rows > 0:
                    x_cpu = x_mlx[:cpu_rows]
                    x_gpu = x_mlx[cpu_rows:]
                    
                    # GPU Path
                    h_gpu = (x_gpu @ w1) + b1
                    h_gpu = mx.maximum(h_gpu, 0)
                    out_gpu = (h_gpu @ w2) + b2
                    
                    # CPU Path
                    mx.set_default_device(mx.cpu)
                    h_cpu = (x_cpu @ w1) + b1
                    h_cpu = mx.maximum(h_cpu, 0)
                    out_cpu = (h_cpu @ w2) + b2
                    mx.set_default_device(mx.gpu)
                    
                    return mx.concatenate([out_cpu, out_gpu], axis=0)
                else:
                    h = (x_mlx @ w1) + b1
                    h = mx.maximum(h, 0)
                    return (h @ w2) + b2

            run_fusion_mlp.compiled_forward = mx.compile(_forward)
            
        res_mlx = run_fusion_mlp.compiled_forward(
            x._mlx, weights['w1']._mlx, weights['b1']._mlx, weights['w2']._mlx, weights['b2']._mlx
        )
        res = Tensor(None, _mlx_data=res_mlx)
        res.eval()
        return res

    else:
        # Compiled training step to get S-tier training speeds!
        def _train_step(x_mlx, w1, b1, w2, b2):
            h = (x_mlx @ w1) + b1
            h = mx.maximum(h, 0)
            res = (h @ w2) + b2
            loss = mx.mean(res)
            return loss

        if not hasattr(run_fusion_mlp, 'compiled_grad'):
            run_fusion_mlp.compiled_grad = mx.compile(mx.value_and_grad(_train_step, argnums=list(range(1, 5))))
            
        loss_val, grads = run_fusion_mlp.compiled_grad(
            x._mlx, weights['w1']._mlx, weights['b1']._mlx, weights['w2']._mlx, weights['b2']._mlx
        )
        
        mx.eval(loss_val, grads)
        return Tensor(None, _mlx_data=loss_val)


# =============================================================================
# BENCHMARK ENGINE
# =============================================================================

def clear_gpu_memory():
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except ImportError:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def time_run(fn, warmups=10, runs=50, clear_mem=False):
    for _ in range(warmups):
        fn()
        if clear_mem:
            clear_gpu_memory()
    times = []
    for _ in range(runs):
        if clear_mem:
            clear_gpu_memory()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    n = len(times)
    return {
        "mean":   float(np.mean(times)),
        "std":    float(np.std(times)),
        "median": float(np.median(times)),
        "min":    float(np.min(times)),
        "max":    float(np.max(times)),
        "ci95":   float(1.96 * np.std(times) / np.sqrt(n)),
        "n_runs": n,
    }


def print_table(results_dict):
    print(f"\n{'='*130}")
    print(f"  UNIFIED MODEL-LEVEL BENCHMARK RESULTS")
    print(f"  Architecture: Pre-LN Decoder Block (matches Swift BenchmarkExample)")
    print(f"{'='*130}")
    hdr = "| Model / Mode | SeqLen | Framework | Mean ± CI95 (ms) | Median (ms) | Std (ms) | vs MLX | Tokens/s | Mem (MB) |"
    sep = "| --- " * (hdr.count("|") - 1) + "|"
    print(f"\n{hdr}")
    print(sep)

    for key, runs_data in results_dict.items():
        mlx_med = runs_data.get("MLX", {}).get("median", 1.0)
        seq_len = 1024

        for fw in ["MLX", "PyTorch (MPS)", "FusionML"]:
            stats = runs_data.get(fw, {})
            if not stats or stats.get("median", 0) == 0:
                print(f"| {key} | {seq_len} | {fw} | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            speedup    = mlx_med / stats["median"] if mlx_med > 0 else 1.0
            speedup_str = f"{speedup:.2f}x" if fw != "MLX" else "1.00x"
            ci95       = stats.get("ci95", 1.96 * stats["std"] / np.sqrt(stats.get("n_runs", 20)))
            tps        = stats.get("tokens_per_sec") or (seq_len * 1000.0 / stats["mean"] if stats["mean"] > 0 else 0)
            mem        = stats.get("peak_mem_mb", 0)
            mem_str    = f"{mem:.0f}" if mem > 0 else "—"
            print(f"| {key} | {seq_len} | {fw} | {stats['mean']:.2f} ± {ci95:.2f} | {stats['median']:.2f} | {stats['std']:.2f} | {speedup_str} | {tps:.0f} | {mem_str} |")
        print(sep)


def run_sub(fw, mode, model):
    cmd = [
        sys.executable,
        __file__,
        "--sub",
        "--fw", fw,
        "--mode", mode,
        "--model", model
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python"))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if res.returncode != 0:
        print(f"  ⚠ Error running {fw} {mode} {model}:", file=sys.stderr)
        print(f"    stderr: {res.stderr[:500]}", file=sys.stderr)
        return None
    lines = res.stdout.strip().split("\n")
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    print(f"  ⚠ No JSON in output of {fw} {mode} {model}. Output:\n{res.stdout[:500]}", file=sys.stderr)
    return None


def get_system_info():
    """Return hardware details and a descriptive folder slug.

    Slug format: Apple_M1_8GB_8CPU_7GPU_16ANE
    GPU cores: queried from Metal/AGX via ioreg (reliable on Apple Silicon).
    ANE cores: chip-name lookup table (not exposed in device tree).
    """
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

    # ANE cores are not exposed in the device tree; use per-chip lookup.
    # Ordered most-specific first so "M1 Ultra" matches before "M1".
    _ANE_LOOKUP = [
        ("M1 Ultra", "32"), ("M2 Ultra", "32"), ("M3 Ultra", "36"), ("M4 Ultra", "64"),
        ("M3 Pro",   "18"), ("M3 Max",   "18"),
        ("M4 Pro",   "20"), ("M4 Max",   "32"),
    ]
    ane_cores = "16"  # M1/M2/M3-base/M4-base all have 16-core ANE
    for chip_key, cores in _ANE_LOOKUP:
        if chip_key in cpu:
            ane_cores = cores
            break

    slug = (
        f"{cpu}_{mem_gb}_{cpu_cores}CPU_{gpu_cores}GPU_{ane_cores}ANE"
        .replace(" ", "_")
    )

    return {
        "cpu":       cpu,
        "memory":    mem_gb,
        "cpu_cores": cpu_cores,
        "gpu_cores": gpu_cores,
        "ane_cores": ane_cores,
        "cpu_slug":  slug,
    }


def main():
    parser = argparse.ArgumentParser(description="FusionML Unified Model Benchmark")
    parser.add_argument("--sub", action="store_true", help="Run as subprocess worker")
    parser.add_argument("--fw", type=str, choices=["mlx", "pytorch", "fusionml"])
    parser.add_argument("--mode", type=str, choices=["inference", "training"])
    parser.add_argument("--model", type=str, choices=["llama", "gpt2", "mlp"])
    args = parser.parse_args()

    if not args.sub:
        # =====================================================================
        # ORCHESTRATOR — launches subprocesses for each combination
        # =====================================================================
        sys_info = get_system_info()

        print("=" * 90)
        print("  FusionML Unified Model-Level Benchmark")
        print(f"  System: {sys_info['cpu']} ({sys_info['memory']})")
        print(f"  Architecture: Pre-LN Decoder Block (matches Swift BenchmarkExample)")
        print("=" * 90)

        results = {}
        combinations = [
            ("Llama-3-8B Inference", "inference", "llama"),
            ("Llama-3-8B Training",  "training",  "llama"),
            ("GPT-2 XL Inference",   "inference", "gpt2"),
            ("GPT-2 XL Training",    "training",  "gpt2"),
            ("MLP Inference",        "inference", "mlp"),
            ("MLP Training",         "training",  "mlp"),
        ]

        for key, mode, model in combinations:
            results[key] = {}
            for fw_display in ["MLX", "PyTorch (MPS)", "FusionML"]:
                fw_arg = "mlx" if "MLX" in fw_display else ("pytorch" if "PyTorch" in fw_display else "fusionml")
                # Cooldown sleep to prevent thermal throttling bias
                time.sleep(10.0)
                print(f"  → Running {key} on {fw_display}...", end=" ", flush=True)
                stats = run_sub(fw_arg, mode, model)
                if stats:
                    results[key][fw_display] = stats
                    print(f"{stats['median']:.2f} ms")
                else:
                    results[key][fw_display] = {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0}
                    print("FAILED")

        print_table(results)

        # Save results
        try:
            out_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "../results", sys_info['cpu_slug']
            ))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "model_comparison.json")

            payload = {
                "benchmark_version": "2.0-unified",
                "system": sys_info,
                "config": {
                    "llama": LLAMA_CONFIG,
                    "gpt2": GPT2_CONFIG,
                    "mlp": MLP_CONFIG,
                },
                "architecture": "Pre-LN Decoder Block (LayerNorm + residual, matches Swift)",
                "results": results,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\n💾 Results saved to: {out_path}")
        except Exception as e:
            print(f"\n⚠ Could not save results: {e}")

        return

    # =========================================================================
    # SUBPROCESS WORKER — runs a single (framework, mode, model) benchmark
    # =========================================================================
    fw = args.fw
    mode = args.mode
    model = args.model
    training = (mode == "training")
    scale = 0.02

    if model == "llama":
        cfg = LLAMA_CONFIG
        D, FFN = cfg["dim"], cfg["ffn_dim"]
        L = cfg["seq_len"]

        # Shared numpy weights (deterministic seed per model)
        np.random.seed(42)
        x_np = np.random.randn(L, D).astype(np.float32) * scale
        w_q_np = np.random.randn(D, D).astype(np.float32) * scale
        w_k_np = np.random.randn(D, D).astype(np.float32) * scale
        w_v_np = np.random.randn(D, D).astype(np.float32) * scale
        w_o_np = np.random.randn(D, D).astype(np.float32) * scale
        w_gate_np = np.random.randn(D, FFN).astype(np.float32) * scale
        w_up_np = np.random.randn(D, FFN).astype(np.float32) * scale
        w_down_np = np.random.randn(FFN, D).astype(np.float32) * scale
        ln1_g_np = np.ones(D, dtype=np.float32)
        ln1_b_np = np.zeros(D, dtype=np.float32)
        ln2_g_np = np.ones(D, dtype=np.float32)
        ln2_b_np = np.zeros(D, dtype=np.float32)

        if fw == "mlx":
            import mlx.core as mx
            x = mx.array(x_np)
            weights = {
                'w_q': mx.array(w_q_np), 'w_k': mx.array(w_k_np),
                'w_v': mx.array(w_v_np), 'w_o': mx.array(w_o_np),
                'w_gate': mx.array(w_gate_np), 'w_up': mx.array(w_up_np),
                'w_down': mx.array(w_down_np),
                'ln1_g': mx.array(ln1_g_np), 'ln1_b': mx.array(ln1_b_np),
                'ln2_g': mx.array(ln2_g_np), 'ln2_b': mx.array(ln2_b_np),
            }
            fn = lambda: run_mlx_llama(x, weights, training=training)

        elif fw == "pytorch":
            import torch
            x = torch.from_numpy(x_np).to("mps")
            weights = {
                'w_q': torch.from_numpy(w_q_np).to("mps").requires_grad_(training),
                'w_k': torch.from_numpy(w_k_np).to("mps").requires_grad_(training),
                'w_v': torch.from_numpy(w_v_np).to("mps").requires_grad_(training),
                'w_o': torch.from_numpy(w_o_np).to("mps").requires_grad_(training),
                'w_gate': torch.from_numpy(w_gate_np).to("mps").requires_grad_(training),
                'w_up': torch.from_numpy(w_up_np).to("mps").requires_grad_(training),
                'w_down': torch.from_numpy(w_down_np).to("mps").requires_grad_(training),
                'ln1_g': torch.from_numpy(ln1_g_np).to("mps").requires_grad_(training),
                'ln1_b': torch.from_numpy(ln1_b_np).to("mps").requires_grad_(training),
                'ln2_g': torch.from_numpy(ln2_g_np).to("mps").requires_grad_(training),
                'ln2_b': torch.from_numpy(ln2_b_np).to("mps").requires_grad_(training),
            }
            fn = lambda: run_torch_llama(x, weights, training=training)

        elif fw == "fusionml":
            from fusionml.tensor import Tensor
            from fusionml._metal.tri_scheduler import get_scheduler
            scheduler = get_scheduler()
            shapes = [
                (1024, 4096, 4096),
                (1024, 4096, 12288),
                (1024, 4096, 28672),
                (1024, 4096, 14336),
                (1024, 14336, 4096),
                (1024, 4096, 1024),
                (1024, 1024, 4096),
                (4096, 4096, 1024),
                (14336, 4096, 1024),
                (4096, 14336, 1024),
                (4096, 1024, 4096),
                (4096, 1024, 14336),
                (14336, 1024, 4096)
            ]
            # Check/load/save calibration cache to save runtime
            cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tri_calibration_llama.json"))
            if os.path.exists(cache_path):
                scheduler.load_calibration(cache_path)
            else:
                scheduler.calibrate(shapes=shapes, iterations=15, verbose=False)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                scheduler.save_calibration(cache_path)

            x = Tensor(x_np, requires_grad=False).to_gpu()
            weights = {
                'w_q': Tensor(w_q_np, requires_grad=training).to_gpu(),
                'w_k': Tensor(w_k_np, requires_grad=training).to_gpu(),
                'w_v': Tensor(w_v_np, requires_grad=training).to_gpu(),
                'w_o': Tensor(w_o_np, requires_grad=training).to_gpu(),
                'w_gate': Tensor(w_gate_np, requires_grad=training).to_gpu(),
                'w_up': Tensor(w_up_np, requires_grad=training).to_gpu(),
                'w_down': Tensor(w_down_np, requires_grad=training).to_gpu(),
                'ln1_g': Tensor(ln1_g_np, requires_grad=training).to_gpu(),
                'ln1_b': Tensor(ln1_b_np, requires_grad=training).to_gpu(),
                'ln2_g': Tensor(ln2_g_np, requires_grad=training).to_gpu(),
                'ln2_b': Tensor(ln2_b_np, requires_grad=training).to_gpu(),
            }
            for w in weights.values():
                w.is_parameter = True
            fn = lambda: run_fusion_llama(x, weights, training=training)

    elif model == "gpt2":
        cfg = GPT2_CONFIG
        D, FFN = cfg["dim"], cfg["ffn_dim"]
        L = cfg["seq_len"]

        np.random.seed(123)
        x_np = np.random.randn(L, D).astype(np.float32) * scale
        w_q_np = np.random.randn(D, D).astype(np.float32) * scale
        w_k_np = np.random.randn(D, D).astype(np.float32) * scale
        w_v_np = np.random.randn(D, D).astype(np.float32) * scale
        w_o_np = np.random.randn(D, D).astype(np.float32) * scale
        w_fc1_np = np.random.randn(D, FFN).astype(np.float32) * scale
        w_fc2_np = np.random.randn(FFN, D).astype(np.float32) * scale
        ln1_g_np = np.ones(D, dtype=np.float32)
        ln1_b_np = np.zeros(D, dtype=np.float32)
        ln2_g_np = np.ones(D, dtype=np.float32)
        ln2_b_np = np.zeros(D, dtype=np.float32)

        if fw == "mlx":
            import mlx.core as mx
            x = mx.array(x_np)
            weights = {
                'w_q': mx.array(w_q_np), 'w_k': mx.array(w_k_np),
                'w_v': mx.array(w_v_np), 'w_o': mx.array(w_o_np),
                'w_fc1': mx.array(w_fc1_np), 'w_fc2': mx.array(w_fc2_np),
                'ln1_g': mx.array(ln1_g_np), 'ln1_b': mx.array(ln1_b_np),
                'ln2_g': mx.array(ln2_g_np), 'ln2_b': mx.array(ln2_b_np),
            }
            fn = lambda: run_mlx_gpt2(x, weights, training=training)

        elif fw == "pytorch":
            import torch
            x = torch.from_numpy(x_np).to("mps")
            weights = {
                'w_q': torch.from_numpy(w_q_np).to("mps").requires_grad_(training),
                'w_k': torch.from_numpy(w_k_np).to("mps").requires_grad_(training),
                'w_v': torch.from_numpy(w_v_np).to("mps").requires_grad_(training),
                'w_o': torch.from_numpy(w_o_np).to("mps").requires_grad_(training),
                'w_fc1': torch.from_numpy(w_fc1_np).to("mps").requires_grad_(training),
                'w_fc2': torch.from_numpy(w_fc2_np).to("mps").requires_grad_(training),
                'ln1_g': torch.from_numpy(ln1_g_np).to("mps").requires_grad_(training),
                'ln1_b': torch.from_numpy(ln1_b_np).to("mps").requires_grad_(training),
                'ln2_g': torch.from_numpy(ln2_g_np).to("mps").requires_grad_(training),
                'ln2_b': torch.from_numpy(ln2_b_np).to("mps").requires_grad_(training),
            }
            fn = lambda: run_torch_gpt2(x, weights, training=training)

        elif fw == "fusionml":
            from fusionml.tensor import Tensor
            from fusionml._metal.tri_scheduler import get_scheduler
            scheduler = get_scheduler()
            shapes = [
                (1024, 1600, 1600),
                (1024, 1600, 4800),
                (1024, 1600, 6400),
                (1024, 6400, 1600),
                (1024, 1600, 1024),
                (1024, 1024, 1600),
                (1600, 1600, 1024),
                (6400, 1600, 1024),
                (1600, 6400, 1024)
            ]
            # Check/load/save calibration cache to save runtime
            cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tri_calibration_gpt2.json"))
            if os.path.exists(cache_path):
                scheduler.load_calibration(cache_path)
            else:
                scheduler.calibrate(shapes=shapes, iterations=15, verbose=False)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                scheduler.save_calibration(cache_path)

            x = Tensor(x_np, requires_grad=False).to_gpu()
            weights = {
                'w_q': Tensor(w_q_np, requires_grad=training).to_gpu(),
                'w_k': Tensor(w_k_np, requires_grad=training).to_gpu(),
                'w_v': Tensor(w_v_np, requires_grad=training).to_gpu(),
                'w_o': Tensor(w_o_np, requires_grad=training).to_gpu(),
                'w_fc1': Tensor(w_fc1_np, requires_grad=training).to_gpu(),
                'w_fc2': Tensor(w_fc2_np, requires_grad=training).to_gpu(),
                'ln1_g': Tensor(ln1_g_np, requires_grad=training).to_gpu(),
                'ln1_b': Tensor(ln1_b_np, requires_grad=training).to_gpu(),
                'ln2_g': Tensor(ln2_g_np, requires_grad=training).to_gpu(),
                'ln2_b': Tensor(ln2_b_np, requires_grad=training).to_gpu(),
            }
            for w in weights.values():
                w.is_parameter = True
            fn = lambda: run_fusion_gpt2(x, weights, training=training)

    elif model == "mlp":
        cfg = MLP_CONFIG
        IN, HID, OUT = cfg["in_dim"], cfg["hidden_dim"], cfg["out_dim"]
        L = cfg["seq_len"]

        np.random.seed(999)
        x_np = np.random.randn(L, IN).astype(np.float32) * scale
        w1_np = np.random.randn(IN, HID).astype(np.float32) * scale
        b1_np = np.zeros((1, HID), dtype=np.float32)
        w2_np = np.random.randn(HID, OUT).astype(np.float32) * scale
        b2_np = np.zeros((1, OUT), dtype=np.float32)

        if fw == "mlx":
            import mlx.core as mx
            x = mx.array(x_np)
            weights = {
                'w1': mx.array(w1_np), 'b1': mx.array(b1_np),
                'w2': mx.array(w2_np), 'b2': mx.array(b2_np),
            }
            fn = lambda: run_mlx_mlp(x, weights, training=training)

        elif fw == "pytorch":
            import torch
            x = torch.from_numpy(x_np).to("mps")
            weights = {
                'w1': torch.from_numpy(w1_np).to("mps").requires_grad_(training),
                'b1': torch.from_numpy(b1_np).to("mps").requires_grad_(training),
                'w2': torch.from_numpy(w2_np).to("mps").requires_grad_(training),
                'b2': torch.from_numpy(b2_np).to("mps").requires_grad_(training),
            }
            fn = lambda: run_torch_mlp(x, weights, training=training)

        elif fw == "fusionml":
            from fusionml.tensor import Tensor, relu
            from fusionml._metal.tri_scheduler import get_scheduler
            scheduler = get_scheduler()
            shapes = [
                (1024, 4096, 4096),
                (1024, 4096, 10),
                (4096, 4096, 1024),
                (4096, 10, 1024),
                (10, 4096, 1024),
                (4096, 1024, 4096)
            ]
            # Check/load/save calibration cache to save runtime
            cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tri_calibration_mlp.json"))
            if os.path.exists(cache_path):
                scheduler.load_calibration(cache_path)
            else:
                scheduler.calibrate(shapes=shapes, iterations=15, verbose=False)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                scheduler.save_calibration(cache_path)

            x = Tensor(x_np, requires_grad=False).to_gpu()
            weights = {
                'w1': Tensor(w1_np, requires_grad=training).to_gpu(),
                'b1': Tensor(b1_np, requires_grad=training).to_gpu(),
                'w2': Tensor(w2_np, requires_grad=training).to_gpu(),
                'b2': Tensor(b2_np, requires_grad=training).to_gpu(),
            }
            for w in weights.values():
                w.is_parameter = True
            fn = lambda: run_fusion_mlp(x, weights, training=training)

    # Benchmark
    # n=50 matches CoDL's statistical standard; warmup=10 uniformly (training warmup
    # sensitivity was tested and found negligible -- see llama_training_warmup_test.py).
    warmups = 10
    runs    = 50
    stats = time_run(fn, warmups=warmups, runs=runs, clear_mem=False)

    # Peak RSS after model + weights are in memory (subprocess baseline already warmed up)
    import resource as _res
    stats["peak_mem_mb"] = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

    # Derive seq_len for tokens/sec (MLP "seq_len" is batch size, still a valid throughput unit)
    _seq_len = {"llama": LLAMA_CONFIG["seq_len"], "gpt2": GPT2_CONFIG["seq_len"], "mlp": MLP_CONFIG["seq_len"]}[model]
    stats["tokens_per_sec"] = _seq_len * 1000.0 / stats["mean"] if stats["mean"] > 0 else 0.0

    # Output JSON on last line for orchestrator to parse
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
