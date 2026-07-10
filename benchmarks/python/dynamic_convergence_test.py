#!/usr/bin/env python3
"""
dynamic_convergence_test.py — does the dynamic gate converge to baseline-or-
better at steady state, with the adaptation transient quantified?

Llama-3-8B block @ seq 8192 (the cell where the 50-run window charged the
gate its full learning cost): adjacent MLX-FP16 baseline (n=50), then the
dynamic arm for n=200 with per-call timeline, then the baseline again (n=50)
to bound thermal drift across the window. Reports median latency per 50-call
segment of the dynamic run vs the bracketing baselines.
"""

import os
import sys
import json
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "prefill_scale_benchmark.py")
MODEL, SEQ = "llama", 8192
COOLDOWN = 90


def run(arm, runs):
    env = os.environ.copy()
    env["PREFILL_RUNS"] = str(runs)
    env["PYTHONPATH"] = os.path.abspath(os.path.join(HERE, "../../python")) + os.pathsep + HERE
    cmd = [sys.executable, SCRIPT, "--sub", "--model", MODEL, "--seq", str(SEQ), "--arm", arm]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if res.returncode != 0:
        print(f"FAILED {arm}: {res.stderr[-500:]}", file=sys.stderr)
        return None
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return None


def main():
    import time
    print(f"Convergence test: {MODEL} seq={SEQ}, baseline(50) → dynamic(200) → baseline(50)")

    b1 = run("mlx_fp16", 50)
    print(f"  baseline pre : {b1['median']:.2f} ms" if b1 else "  baseline pre : FAILED")
    time.sleep(COOLDOWN)

    dyn = run("fusion_dynamic", 200)
    if dyn:
        print(f"  dynamic n=200: {dyn['median']:.2f} ms  final_mode={dyn['final_mode']} switches={dyn['mode_switches']}")
    time.sleep(COOLDOWN)

    b2 = run("mlx_fp16", 50)
    print(f"  baseline post: {b2['median']:.2f} ms" if b2 else "  baseline post: FAILED")

    if not (b1 and dyn and b2):
        sys.exit(1)

    # Timeline includes probation calls first; timed window is the last 200.
    tl = dyn["timeline"]
    timed = tl[-200:]
    print(f"\n  timeline: {len(tl)} calls total ({len(tl)-200} probation/warm), segments of timed 200:")
    for i in range(0, 200, 50):
        seg = timed[i:i+50]
        med = float(np.median([t for _, t in seg]))
        modes = {m: sum(1 for mm, _ in seg if mm == m) for m in sorted({mm for mm, _ in seg})}
        base_interp = b1["median"] + (b2["median"] - b1["median"]) * ((i + 25) / 200.0)
        print(f"    calls {i+1:3d}-{i+50:3d}: median {med:8.2f} ms  vs interp-baseline {base_interp:8.2f} ms "
              f"({base_interp/med:.3f}x)  modes={modes}")

    sys.path.insert(0, HERE)
    from bench_hw import get_system_info, get_power_state
    out = {"environment": get_power_state(), "baseline_pre": b1, "dynamic_200": dyn, "baseline_post": b2}
    out_dir = os.path.join(HERE, "../results", get_system_info()["cpu_slug"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dynamic_convergence_test.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Saved to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
