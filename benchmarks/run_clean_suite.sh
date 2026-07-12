#!/bin/bash
# run_clean_suite.sh — clean-protocol benchmark suite for a fresh machine
# (designed for the M4 Mac mini replication; works on any Apple Silicon Mac).
#
# Protocol: AC power verified, sequential subprocesses only, cooldowns between
# heavy phases, power/thermal state logged into every result JSON.
# Results land in benchmarks/results/<auto-detected-hw-slug>/.
#
# Usage:  ./benchmarks/run_clean_suite.sh
# Runtime: roughly 2-3 hours. Leave the machine alone while it runs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$SCRIPT_DIR/python"
VENV="$PY_DIR/.venv"
COOLDOWN=${COOLDOWN:-120}

cd "$PY_DIR"

if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating venv + installing requirements..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet -r requirements.txt
    "$VENV/bin/pip" install --quiet -e "$SCRIPT_DIR/../python"
fi
PY="$VENV/bin/python"

echo "=== Clean-protocol benchmark suite ==="
"$PY" -c "from bench_hw import get_power_state; import json; print(json.dumps(get_power_state(), indent=2))"

if ! pmset -g batt | grep -q "AC Power"; then
    echo "⚠ NOT on AC power. Connect the power adapter and re-run." >&2
    exit 1
fi

echo ""
echo "── Phase 1/3: prefill-scale benchmark (all 4 arms, both models, seq 1024-8192)"
"$PY" prefill_scale_benchmark.py --cooldown 30

echo ""
echo "── Cooling ${COOLDOWN}s before phase 2..."
sleep "$COOLDOWN"

echo "── Phase 2/3: MLX-FP16 training baseline (fair training comparison)"
"$PY" mlx_fp16_training_baseline.py

echo ""
echo "── Cooling ${COOLDOWN}s before phase 3..."
sleep "$COOLDOWN"

echo "── Phase 3/4: dynamic-gate convergence test (llama @ 8192, n=200 timeline)"
"$PY" dynamic_convergence_test.py

echo ""
echo "── Cooling ${COOLDOWN}s before phase 4..."
sleep "$COOLDOWN"

echo "── Phase 4/4: ANE re-validation (dispatch overhead + MLX-coexistence crash repros)"
# These characterize the ANE/CoreML stack on THIS machine. The two repro
# scripts are EXPECTED to crash (segfault) on stacks with the known
# coremltools/MLX bug — a crash here is a valid result, not a suite failure.
SLUG="$("$PY" -c "from bench_hw import get_system_info; print(get_system_info()['cpu_slug'])")"
ANE_LOG="$SCRIPT_DIR/results/$SLUG/ane_revalidation.txt"
mkdir -p "$(dirname "$ANE_LOG")"
{
    echo "=== ane_spike_test.py (dispatch overhead vs GPU) ==="
    "$PY" ane_spike_test.py 2>&1 || echo "[exit code $? — see above]"
    echo ""
    echo "=== ane_mlx_minimal_repro.py (second-shape segfault repro) ==="
    "$PY" ane_mlx_minimal_repro.py 2>&1 || echo "[exit code $? — segfault(139)=bug still present; 0=FIXED on this stack]"
    echo ""
    echo "=== ane_multi_tier_repro.py (multi-tier segfault repro) ==="
    "$PY" ane_multi_tier_repro.py 2>&1 || echo "[exit code $? — segfault(139)=bug still present; 0=FIXED on this stack]"
} | tee "$ANE_LOG"

echo ""
echo "=== Suite complete ==="
echo "Results directory:"
"$PY" -c "from bench_hw import get_system_info; print('  benchmarks/results/' + get_system_info()['cpu_slug'] + '/')"
echo "Commit and push that directory (plus this script's output above) to share."
