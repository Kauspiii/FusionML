# FusionML Benchmarks

Community-driven, fair-baseline benchmarks for FusionML across Apple Silicon
generations. All comparisons are **precision-matched** (FusionML FP16 vs MLX
FP16 — never FP16-vs-FP32), correctness-checked, and every result JSON embeds
the machine's hardware, power, and thermal state.

## Contribute results from your machine (M1/M2/M3/M4, any tier)

One command. Roughly 2–3 hours, unattended:

```bash
git clone https://github.com/ommo007/FusionML.git && cd FusionML
git checkout benchmarks
./benchmarks/run_clean_suite.sh
```

The suite creates its own Python venv, verifies you are on AC power, runs all
phases strictly sequentially with cooldowns, and writes results to
`benchmarks/results/<your-hardware-slug>/` (e.g.
`Apple_M3_Pro_18GB_11CPU_14GPU_18ANE`). Then:

```bash
git checkout -b results/<your-chip>
git add benchmarks/results/
git commit -m "results: <your chip> clean-suite run"
git push -u origin results/<your-chip>
```

and open a PR — the template asks only for what the JSONs can't capture
(what else was running, laptop vs desktop, anything unusual).

**Ground rules** (the suite enforces most of these):
- AC power connected. On laptops: lid open, no clamshell.
- Quit other apps; don't use the machine during the run.
- Fanless machines (MacBook Air): expect noisier numbers — that's itself
  useful data; note it in the PR.
- Don't hand-edit result JSONs. If a phase fails, include the console output.

## What the suite measures

| Phase | Script | Question it answers |
|---|---|---|
| 1 | `python/prefill_scale_benchmark.py` | Does per-layer CPU+GPU stream splitting beat a fair MLX-FP16 baseline at seq 1024–8192? Four arms: baseline / FusionML-nosplit / split / dynamic gate. |
| 2 | `python/mlx_fp16_training_baseline.py` | Precision-matched training comparison (with loss-finiteness checks). |
| 3 | `python/dynamic_convergence_test.py` | Does the runtime split/nosplit gate converge to baseline-or-better, and what does probing cost? (n=200 per-call timeline) |
| 4 | `python/ane_spike_test.py` + ANE crash repros | ANE dispatch overhead on this stack, and whether the CoreML/MLX coexistence segfaults (present on macOS ≤25.x) are fixed. |

## Script index

**Current measurement suite** (run by `run_clean_suite.sh`): the four scripts
above, plus `python/bench_hw.py` (hardware/power detection, used by all).

**Model-level comparison** (`python/model_comparison.py`): the original
MLX / PyTorch-MPS / FusionML decoder-block benchmark (`./run_benchmark.sh
--cross-framework`). Its results feed `python/collate_results.py`, which
generates the paper's LaTeX tables.

**Mechanism probes** (single-question diagnostics, kept for reproducibility):
- `python/stream_parallelism_probe.py` — MLX serializes cross-stream work on
  in-graph dependencies (0.66×); an eager boundary restores concurrency (1.34×).
- `python/split_point_diagnostic.py` — which Linear layers pay for splitting.
- `python/vag_nan_bisect.py`, `python/vag_nan_bisect2.py`,
  `python/minimal_vag_fp16_repro.py`, `python/training_nan_diag.py` — the
  FP16 value_and_grad NaN investigation (resolved: uncommitted attention fix).
- `python/tricompute_adaptive_search.py` / `tricompute_scale_test.py` /
  `tricompute_sustained_test.py` — contention-aware 3-way (CPU+GPU+ANE)
  batch-scale split search: 1.6–1.8× burst, 1.16–1.29× sustained on M1.
- `python/ane_mlx_minimal_repro.py`, `ane_multi_tier_repro.py` — minimal
  CoreML/MLX segfault repros (fixed by macOS 26.3).
- `python/coexec_gpt2_test.py`, `mlp_cold_calibration_test.py`,
  `mlx_fp16_baseline_test.py`, `llama_fp16_nan_check.py`,
  `cholesky_benchmark.py`, `smart_matmul_test.py`,
  `throttle_guard_unit_test.py`, `fp16_ablation*.py` — historical
  experiments; see `PAPER_READINESS.md` for which claims each supports.

**Swift benchmark**: `examples/benchmark/main.swift`
(`swift run -c release BenchmarkExample`) — per-layer `IntelligentRouter`
co-execution on the Swift side.

## Reading results

`PAPER_READINESS.md` is the authoritative, ranked summary of every claim,
its verification status, and every retired/invalid number. Read it before
citing anything. Headline verified results as of 2026-07-12:

- Per-layer split vs MLX-FP16, decoder blocks: **1.13–1.25× (M4 24GB)**,
  1.06–1.18× (M1 8GB), replicated across sessions.
- Dynamic gate: never below baseline (≥1.0× every cell on M4).
- Batch-scale 3-way matmul: up to **1.79×** burst (M1).
- Honest negatives: training loses to MLX-FP16 on both chips (0.86–0.97×);
  ANE not viable at layer granularity (~20 ms dispatch overhead).

## Hall of Fame

Contributors who submitted verified clean-suite runs:

| Chip | Cooling | Contributor |
|---|---|---|
| Apple M1 8GB (MacBook Air) | passive | @ommo007 |
| Apple M4 24GB (Mac mini) | active | @ommo007 |
| *your machine here* | | |
