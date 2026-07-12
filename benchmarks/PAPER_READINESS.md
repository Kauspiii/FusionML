# FusionML — Paper Readiness Status

Last updated: 2026-07-10 (M1 8GB fanless + M4 24GB Mac mini; all numbers
n=50/warmup=10 unless noted; every comparison below is against a
precision-matched MLX-FP16 baseline unless explicitly labeled FP32).

## M4 Mac mini replication (24GB, actively cooled, AC) — HEADLINE TABLE

Clean-protocol suite (`run_clean_suite.sh`). Speedup vs adjacent MLX-FP16:

| Cell | split | dynamic | nosplit | gate mode |
|---|---|---|---|---|
| Llama 1024 | 1.148× | 1.146× | 1.003× | split |
| Llama 2048 | 1.166× | 1.171× | 1.002× | split |
| Llama 4096 | **1.249×** | **1.251×** | 1.006× | split |
| Llama 8192 | 1.170× | 1.167× | 1.000× | split |
| GPT-2 1024 | 1.053× | 1.006× | 1.019× | nosplit |
| GPT-2 2048 | 1.025× | 1.022× | 1.025× | nosplit |
| GPT-2 4096 | 1.136× | 1.169× | 1.024× | split |
| GPT-2 8192 | 1.054× | 1.053× | 1.025× | split |

- **Floor requirement met: every dynamic cell ≥ 1.006×.** Gate picked split
  where split wins, nosplit where parity — zero mid-run switches needed.
- GPT-2 instability GONE on cooled/desktop hardware — confirms the M1 swings
  were thermal/power, not architectural.
- Llama/8192 memory-pressure hypothesis CONFIRMED: 0.97× on 8GB → 1.17× on
  24GB. Convergence test (n=200): dynamic holds 1.179–1.195× across the whole
  window, baseline drift zero (1432.9→1431.2 ms), probe reads within 1–4% of
  true baseline — **observer effect is memory-pressure-specific and vanishes
  at 24GB**, completing the cross-machine ablation.
- Llama split wins are LARGER on M4 (1.15–1.25× vs 1.06–1.18× on M1).

**M4 training-NaN RESOLVED (2026-07-12): version skew, not a platform bug.**
The June attention fix (1/√D scaling + fp32 softmax) had never been
committed — it existed only in the M1 machine's working tree. The mini ran
the committed (unscaled, softmax-free) code, which overflows FP16 training
to NaN, exactly as the June postmortem predicted. Bisected conclusively
(`vag_nan_bisect.py`/`vag_nan_bisect2.py`: identical math copied from the
working tree = finite; imported committed function = NaN; same process,
weights, versions). Fix committed in 5506d46; full framework synced in b3fd136 (the entire
Python+Swift framework had been working-tree-only). Prefill/split results
were never affected (self-contained fixed math).

**Corrected M4 training numbers (valid, losses finite):** Llama — MLX-FP32
572.3 ms, MLX-FP16 434.7 ms, FusionML 457.7 ms → **0.950× vs fair baseline**;
GPT-2 — 81.4 / 65.4 / 72.1 ms → **0.907×**. Matches the M1 result
(0.969×/0.858×): FusionML training loses to MLX-FP16 on BOTH hardware
generations. The training honest-negative is now dual-hardware verified. Side finding kept: MLX 0.30.1
vs 0.32.0 both compute this workload finitely on both chips, and the ANE
crash conditions are absent on the mini's stack (macOS 26.3, same
coremltools 9.0 — repros run clean there, segfault on macOS 25.5).

## Headline claims, ranked by strength

### 1. Contention-aware heterogeneous scheduling wins at batch scale (STRONG)
Raw FFN-shaped matmuls, true concurrent CPU+GPU(+ANE) execution with
contention-aware adaptive split search (`tricompute_adaptive_search.py`):
- 2048 rows: **1.70×** vs GPU-only (2-way, search self-excluded ANE)
- 4096 rows: **1.61×** (3-way) 8192 rows: **1.79×** (3-way)
- Sustained (90s continuous): 1.16–1.29× — burst vs sustained gap is itself
  a disclosed finding.
- Key insight: per-unit throughput degrades under concurrent multi-unit load
  (GPU ~13.2 → 15.5 μs/row); solo-unit calibration is systematically wrong.

### 2. Lazy-graph frameworks serialize cross-stream heterogeneous work (STRONG, mechanism)
`stream_parallelism_probe.py`, fp16, 2048×1600 @ 1600×6400, 35% CPU split:
- Input pre-materialized: split = **1.376×** vs GPU-only
- Input = unmaterialized GPU output in same eval graph: split = **0.658×** (loses)
- Eager `mx.eval` boundary before split: **1.342×** (win restored)

This mechanistically explains why Swift's eager per-layer router shows
co-execution wins while every lazy single-graph Python attempt failed.
Generalizable claim about lazy-evaluation ML frameworks on unified memory.

### 3. Per-layer split beats fair MLX-FP16 at block level — LLAMA ONLY is replicated
`prefill_scale_benchmark.py`, full decoder blocks, correctness-checked
(max rel err ≤ 3e-3 vs GPU-only forward, FP16 noise level). Two full
independent sessions (run 1 after ~0-50 min of load; run 2 after ~2-3 h):

| Model | seq | split run 1 | split run 2 | dynamic (run 2) | nosplit |
|---|---|---|---|---|---|
| Llama-3-8B | 1024 | **1.178×** | **1.183×** | 1.158× | ~1.00× |
| Llama-3-8B | 2048 | 1.083× | 1.084× | 1.063× | ~1.00× |
| Llama-3-8B | 4096 | 1.062× | **1.130×** | 1.060× | ~1.00× |
| Llama-3-8B | 8192 | 0.990× | 0.972× | 0.919× | ~1.00× |
| GPT-2 XL | 1024 | 1.022× | 0.761× | 0.980× (gate→nosplit) | ~1.00× |
| GPT-2 XL | 2048 | 1.018× (1.084× cold) | 0.859× | 0.944× | ~1.02× |
| GPT-2 XL | 4096 | 1.140× | 1.072× | 1.094× | ~1.06× |
| GPT-2 XL | 8192 | 0.929× | 0.843× | 0.816× | ~1.02× |

**Llama split wins (1.08–1.18× at seq ≤ 4096) replicate across sessions and
thermal states — this is the citable block-level claim.** GPT-2 split results
are UNSTABLE across sessions (0.76–1.14× for the same cell) — the smaller
matmul shapes (D=1600) make the CPU chunk hypersensitive to thermal/scheduler
state. Do NOT cite GPT-2 block-level split numbers until the clean-machine
replication protocol below is done.

**Dynamic gate** (probation + runtime re-probing, `fusion_dynamic` arm):
verification on AC power (adjacent baselines): GPT-2/1024 **0.991×** (refused
the losing split), GPT-2/8192 **1.037×** (took the split and won on the
cooler machine). Llama/8192 exposed a structural hole, since fixed: a
split-vs-compiled gate has no true floor when BOTH modes lose to eager MLX —
`dynamic_convergence_test.py` (n=200, per-call timeline, bracketing
baselines) showed compiled-nosplit itself at ~0.94× vs eager at seq 8192.
Gate is now three-mode (split / compiled / **eager**, where eager ≡ the
baseline code path), making the floor structural. Convergence re-run pending.

**New standalone finding: `mx.compile` LOSES to eager MLX at large sequence
lengths** (llama@8192: ~0.94×, reproduced across hot and cool sessions).
The compiled path FusionML inherited as its default is itself a regression
at scale — worth a paragraph as an independent characterization result.

**New standalone finding: comparative probing is DESTRUCTIVE under memory
pressure (observer effect on unified memory).** Three-mode gate convergence
test at llama/8192 on 8GB (`dynamic_convergence_test.json`, per-call
timeline): probation cold-picked split correctly (2612 ms vs eager 2710);
during the run, probes of inactive modes read 3400–4050 ms — ~1.4× their
true fresh-process latency — because each probe evicts the active mode's
working set, and the first active call after a probe pays ~1.7× reload cost
(4740 ms observed). Consequence: a periodically-probing adaptive gate
(a) cannot observe alternative modes accurately at memory-pressure scale and
(b) pays real disruption cost for trying (segment medians 0.80–0.93× vs
baseline). The gate design is therefore SCOPED: continuous probing where the
working set fits (verified floors: GPT-2/1024 0.991×, GPT-2/8192 1.037×);
probation-then-lock (probe only on ThrottleGuard-style drift evidence, never
on a schedule) where it does not. This observer-effect quantification is a
citable result on its own.

Split contribution is cleanly attributed: nosplit arm (= current FusionML,
FP16+compile) is ~1.00× everywhere, so the entire margin comes from the
per-layer eager-boundary split. Wins fade at seq 8192 (attention share grows,
memory pressure on 8GB, thermal).

### 4. Honest negative results (VALUABLE — keep in paper)
- FusionML-nosplit (FP16+`mx.compile`) has NO advantage over MLX-FP16:
  inference ~0.94–1.01×, training **0.969×** (Llama) / **0.858×** (GPT-2)
  (`mlx_fp16_training_baseline.py`, thermal-matched, loss-finiteness checked).
  The historical 1.10–1.35× claims were FP16-vs-FP32 precision artifacts.
- Cholesky/SYRK: tri-compute never beats CPU BLAS (0.90–0.95×) — benefit is
  workload-shape-dependent.
- ANE: unusable in-process on this stack (three documented crash conditions,
  coremltools 9.0 + MLX); dispatch overhead ~24ms fixed + ~7μs/row.

### 5. Production hardening (SUPPORTING)
- ThrottleGuard: latency-drift throttling detection, verified state machine,
  disclosed limitations.
- smart_matmul: probation-based naive fallback, median-of-3, full-dispatch
  timing including readback.

## What must NOT be claimed
- Any "FusionML beats MLX" number whose baseline is FP32 — training AND
  inference both lose or tie once precision-matched (see §4).
- ANE as a working scheduling target.
- ≥1.30× block-level speedups. Current honest block-level range is
  0.93–1.18×; the 1.3–1.8× range is raw-matmul/batch-scale only.
- The first sustained-run's "tri-compute throttles worse than GPU-only"
  asymmetry (did not replicate).

## Open items before preprint
1. **Clean-machine replication protocol (REQUIRED before citing any block
   numbers).** The benchmark machine is a FANLESS M1 (passive cooling) and
   some 2026-07-10 sessions ran on battery — two uncontrolled variables.
   Protocol: AC power connected (verify `pmset -g batt`), cold boot or
   ≥30 min idle, no other apps, 10-15 min cooldowns after heavy cells
   (fanless needs far longer than the 8 s the suite currently uses),
   randomized arm order within each cell, baseline re-measured adjacent to
   every arm, power source + thermal state logged into the result JSON,
   ≥3 independent sessions. Today's runs shared one long thermal session,
   fixed arm order (split always ran hotter than its baseline), and mixed
   power states. Paper should disclose "passively cooled M1" — it makes the
   thermal-adaptive scheduling story (ThrottleGuard, dynamic gate) stronger,
   since fanless is the worst case the adaptivity exists for.
2. GPT-2 block split instability (0.76–1.14× same cell across sessions) —
   diagnose small-shape CPU-chunk sensitivity before making any GPT-2
   block-level claim. Llama results are stable; scope claims to Llama-class
   shapes (D≥4096) if GPT-2 doesn't stabilize.
3. Re-run `fusion_dynamic` with retightened gate parameters (7-sample
   probation, 10-call probe cadence) under the clean protocol — target claim:
   "never regresses below nosplit, captures split wins where they exist."
4. Llama/8192: split fades to ~0.97-0.99× — scope block-level claim to
   seq ≤ 4096, or investigate memory pressure (8GB machine) on ≥16GB hardware.
5. Multi-hardware replication (M2/M3/M4) of §2 and §3 — generality.
6. Per-layer split currently lives in the benchmark; port into
   `fusionml` package proper (smart_matmul-style API) before claiming it as
   a framework feature.
7. Decide venue: results profile (systems mechanism + scheduling) fits
   MLSys/ASPLOS strongly; ICLR possible but this is not a learning-algorithms
   contribution.
