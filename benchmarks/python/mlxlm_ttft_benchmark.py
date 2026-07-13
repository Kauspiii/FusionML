#!/usr/bin/env python3
"""
mlxlm_ttft_benchmark.py — the "survives contact with reality" experiment.

Loads a REAL checkpoint through mlx-lm (the major MLX LLM runner), then
measures time-to-first-token (TTFT = prefill) and decode tok/s for:
  stock    — unmodified mlx-lm
  fusionml — mlx-lm with nn.Linear patched to use FusionML's per-layer
             eager-boundary CPU+GPU stream split for large-row (prefill)
             matmuls. Decode calls (1 row) fall through untouched — the
             mechanism is prefill-only by design (decode is bandwidth-bound
             on unified memory; co-execution cannot add bandwidth).

Greedy decoding (temp=0) on identical prompts; the generated outputs are
recorded verbatim (prompt + reply transcripts) and compared token-for-token
between arms — same tokens, faster first token, is the proof artifact.

Requires: pip install mlx-lm   (the script tells you if it's missing)
Model default: mlx-community/Qwen2.5-7B-Instruct-bf16 — ungated, Apache-2.0,
full-precision weights (~15 GB; needs the 24GB machines). Override with
--model; Meta-Llama repos are HF-gated and need a logged-in account.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-bf16"
PROMPT_TOKENS = [2048, 4096, 8192]
NEW_TOKENS = 64
TTFT_REPEATS = 5
MIN_SPLIT_ROWS = 1024
CPU_RATIO = 0.30  # untuned fixed ratio from the block-level suite findings

BASE_PASSAGE = (
    "The study of heterogeneous computing on unified memory architectures "
    "raises questions about when concurrent execution across processing "
    "units yields real speedups and when shared resources such as memory "
    "bandwidth make additional compute units irrelevant. "
)


# ---------------------------------------------------------------------------
# nn.Linear patch
# ---------------------------------------------------------------------------

_split_stats = {"split_calls": 0, "passthrough_calls": 0}


def install_patch():
    import mlx.core as mx
    import mlx.nn as nn

    original_call = nn.Linear.__call__

    def patched_call(self, x):
        w = self.weight  # (out, in) — computes x @ w.T
        rows = 1
        for s in x.shape[:-1]:
            rows *= s
        if rows < MIN_SPLIT_ROWS or w.shape[0] * w.shape[1] < 1024 * 1024:
            _split_stats["passthrough_calls"] += 1
            return original_call(self, x)

        _split_stats["split_calls"] += 1
        x2 = x.reshape(rows, x.shape[-1])
        mx.eval(x2)  # eager boundary — required for true cross-stream concurrency
        cpu_rows = int(rows * CPU_RATIO)
        wT = mx.transpose(w)
        c = mx.matmul(x2[:cpu_rows], wT, stream=mx.cpu)
        g = mx.matmul(x2[cpu_rows:], wT, stream=mx.gpu)
        y = mx.concatenate([c, g], axis=0)
        if "bias" in self:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], w.shape[0])

    nn.Linear.__call__ = patched_call
    return lambda: setattr(nn.Linear, "__call__", original_call)


# ---------------------------------------------------------------------------
# Generation + timing
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, target_tokens):
    text = BASE_PASSAGE
    while len(tokenizer.encode(text)) < target_tokens:
        text += BASE_PASSAGE
    ids = tokenizer.encode(text)[:target_tokens - 16]
    question = " Question: summarize the key trade-off in one sentence. Answer:"
    return tokenizer.decode(ids) + question


def timed_generate(model, tokenizer, prompt):
    """Returns (ttft_ms, decode_tok_s, output_text, output_token_count)."""
    from mlx_lm import stream_generate

    t0 = time.perf_counter()
    ttft_ms = None
    n_tokens = 0
    pieces = []
    t_first = None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=NEW_TOKENS):
        now = time.perf_counter()
        if ttft_ms is None:
            ttft_ms = (now - t0) * 1000.0
            t_first = now
        n_tokens += 1
        pieces.append(resp.text)
    decode_s = (time.perf_counter() - t_first) if t_first else 0.0
    decode_tok_s = (n_tokens - 1) / decode_s if decode_s > 0 and n_tokens > 1 else None
    return ttft_ms, decode_tok_s, "".join(pieces), n_tokens


def bench_arm(model, tokenizer, prompt):
    ttfts, decode_rates, output = [], [], None
    for i in range(TTFT_REPEATS):
        ttft, rate, text, _ = timed_generate(model, tokenizer, prompt)
        ttfts.append(ttft)
        if rate:
            decode_rates.append(rate)
        if output is None:
            output = text
        elif text != output:
            print("    note: output varied between repeats (greedy should be deterministic)",
                  file=sys.stderr)
    return {"ttft_median_ms": float(np.median(ttfts)),
            "ttft_all_ms": [round(t, 2) for t in ttfts],
            "decode_tok_s_median": float(np.median(decode_rates)) if decode_rates else None,
            "output_text": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default=",".join(str(p) for p in PROMPT_TOKENS))
    args = parser.parse_args()

    try:
        from mlx_lm import load
    except ImportError:
        print("✗ mlx-lm not installed. Run:  .venv/bin/pip install mlx-lm")
        sys.exit(1)

    from bench_hw import get_system_info, get_power_state
    slug = get_system_info()["cpu_slug"]
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results", slug))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.model} (first run downloads from HuggingFace)...")
    model, tokenizer = load(args.model)

    # Sanity: warn if the checkpoint is quantized (patch only touches nn.Linear)
    import mlx.nn as nn
    n_linear = sum(1 for _, m in model.named_modules() if type(m) is nn.Linear)
    n_quant = sum(1 for _, m in model.named_modules() if "Quantized" in type(m).__name__)
    print(f"  modules: {n_linear} nn.Linear, {n_quant} quantized")
    if n_quant > 0 and n_linear == 0:
        print("✗ Fully quantized checkpoint — the split patch has nothing to attach to. "
              "Use a bf16/fp16 model (e.g. append '-bf16' community variant).")
        sys.exit(1)

    results, transcripts = {}, []
    for target in [int(p) for p in args.prompt_tokens.split(",")]:
        prompt = build_prompt(tokenizer, target)
        n_prompt = len(tokenizer.encode(prompt))
        print(f"\n▶ prompt ≈ {n_prompt} tokens (target {target}), {NEW_TOKENS} new tokens, "
              f"greedy, {TTFT_REPEATS} repeats")

        print("   stock mlx-lm  ... ", end="", flush=True)
        stock = bench_arm(model, tokenizer, prompt)
        print(f"TTFT {stock['ttft_median_ms']:8.1f} ms   decode {stock['decode_tok_s_median']:.1f} tok/s")

        _split_stats.update(split_calls=0, passthrough_calls=0)
        restore = install_patch()
        try:
            print("   fusionml-split... ", end="", flush=True)
            fused = bench_arm(model, tokenizer, prompt)
            print(f"TTFT {fused['ttft_median_ms']:8.1f} ms   decode {fused['decode_tok_s_median']:.1f} tok/s")
        finally:
            restore()

        match = stock["output_text"] == fused["output_text"]
        speedup = stock["ttft_median_ms"] / fused["ttft_median_ms"]
        print(f"   → TTFT speedup {speedup:.3f}x   outputs token-identical: {match}   "
              f"(split matmul calls during patched run: {_split_stats['split_calls']})")

        results[str(target)] = {
            "prompt_tokens": n_prompt, "new_tokens": NEW_TOKENS,
            "stock": {k: v for k, v in stock.items() if k != "output_text"},
            "fusionml": {k: v for k, v in fused.items() if k != "output_text"},
            "ttft_speedup": speedup,
            "outputs_identical": match,
            "split_calls": _split_stats["split_calls"],
        }
        transcripts.append({
            "prompt_tokens": n_prompt,
            "prompt_text": prompt,
            "stock_output": stock["output_text"],
            "fusionml_output": fused["output_text"],
            "outputs_identical": match,
        })

        with open(os.path.join(out_dir, "mlxlm_ttft_benchmark.json"), "w") as f:
            json.dump({"model": args.model, "greedy": True, "repeats": TTFT_REPEATS,
                       "cpu_ratio": CPU_RATIO, "min_split_rows": MIN_SPLIT_ROWS,
                       "environment": get_power_state(), "results": results}, f, indent=2)
        with open(os.path.join(out_dir, "mlxlm_transcripts.json"), "w") as f:
            json.dump(transcripts, f, indent=2)

    # Human-readable proof artifact
    md = [f"# MLX-LM TTFT proof transcripts — {args.model}\n",
          "Greedy decoding; stock vs FusionML-patched runs on identical prompts.\n"]
    for t in transcripts:
        md.append(f"\n## Prompt ({t['prompt_tokens']} tokens; shown truncated)\n")
        md.append("```\n..." + t["prompt_text"][-400:] + "\n```\n")
        md.append(f"\n**Stock output:**\n\n> {t['stock_output']}\n")
        md.append(f"\n**FusionML output:**\n\n> {t['fusionml_output']}\n")
        md.append(f"\n**Token-identical: {t['outputs_identical']}**\n")
    with open(os.path.join(out_dir, "mlxlm_transcripts.md"), "w") as f:
        f.write("".join(md))

    print(f"\n💾 Saved results + transcripts to {out_dir}/")


if __name__ == "__main__":
    main()
