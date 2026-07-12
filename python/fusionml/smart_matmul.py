"""
smart_matmul — general-purpose, model-agnostic tri-compute matrix
multiplication. Drop-in replacement for `a @ b` on any numpy array shape,
from any caller (your own PyTorch/MLX/numpy model, a numerical solver,
anything) -- not tied to Llama, GPT-2, or any specific model this project
benchmarks.

    import fusionml
    c = fusionml.smart_matmul(a, b)   # a, b: numpy arrays, float32

This is CPU (Accelerate) + GPU (MLX/Metal) concurrent dispatch, real and
proven safe across an extensive test campaign: adaptive, contention-aware
row splitting, a naive-fallback safety net (a shape that isn't actually
faster to split falls back to plain `a @ b` permanently after one measured
attempt, so this can never regress catastrophically vs doing nothing), and
ThrottleGuard protection against sustained-load thermal throttling.

ANE (Apple Neural Engine) support EXISTS but is EXPERIMENTAL and OFF BY
DEFAULT -- do not enable it in any code path where a crash is unacceptable.
During development, real CoreML+MLX interaction segfaults (unrecoverable at
the Python level, exit 139) were reproduced under at least three distinct
conditions: (1) loading a second distinct ANE shape in one process, (2)
compiling an ANE model inline in a process that has touched MLX, (3) loading
even a single, correctly-prewarmed ANE model after enough PRIOR, unrelated
MLX activity had already occurred in that process. Condition 3 was found
late, is not fully characterized, and cannot be reliably guarded against
for an arbitrary caller whose prior MLX usage this module cannot see or
control. Enabling ANE is provided for controlled experimentation only:

    b = my_weight_matrix
    fusionml.prewarm_ane_shape(M, K, N, b)          # isolated subprocess, never touches MLX
    scheduler = fusionml.TriComputeMatmul(enable_ane=True)  # prints a loud runtime warning
    c = scheduler(a, b)                              # may segfault the whole process -- see above

If you need ANE's contribution reliably, the safer pattern is running it in
its own subprocess (matching benchmarks/python/ane_spike_test.py) rather
than in-process with MLX at all.
"""

import os
import sys
import subprocess
import time
import threading
from typing import Dict, Optional, Tuple

import numpy as np

from ._metal.throttle_guard import ThrottleGuard

_HAS_MLX = False
try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    pass

_HAS_ANE = False
try:
    from ._metal.ane_backend import ane_matmul, ane_available, _weight_hash, _cache_dir
    _HAS_ANE = ane_available()
except ImportError:
    pass


# Below this many rows, dispatch/thread overhead exceeds any split benefit --
# just run on GPU (or CPU if GPU unavailable) directly, no splitting at all.
MIN_ROWS_TO_SPLIT = 512

# ANE's fixed CoreML dispatch overhead (~24ms, see ane_spike_test.py) makes it
# not worth considering below this row count even for the one prewarmed shape.
MIN_ROWS_FOR_ANE = 3000

# Default starting split before any shape-specific online refinement exists.
# Deliberately GPU-heavy and ANE-conservative -- _maybe_refine_calibration
# adjusts the GPU/CPU split per shape after a few live measurements.
_DEFAULT_RATIOS = {"gpu": 0.55, "cpu": 0.30, "ane": 0.15}


def _ane_cache_key(M: int, K: int, N: int, b: np.ndarray, compute_units: str = "CPU_AND_NE") -> str:
    b_f16 = np.ascontiguousarray(b, dtype=np.float16)
    wh = _weight_hash([b_f16])
    return f"matmul_{M}_{K}_{N}_{compute_units}_{wh}"


def is_ane_prewarmed(M: int, K: int, N: int, b: np.ndarray) -> bool:
    """Check whether this exact (shape, weight) is already disk-cached -- never
    triggers a compile, just checks a file's existence."""
    if not _HAS_ANE:
        return False
    key = _ane_cache_key(M, K, N, b)
    return os.path.exists(os.path.join(_cache_dir(), f"{key}.mlpackage"))


def prewarm_ane_shape(M: int, K: int, N: int, b: np.ndarray, timeout: float = 60.0) -> bool:
    """
    Compile and disk-cache an ANE model for this exact (M, K, N, b) in an
    ISOLATED SUBPROCESS that never imports MLX -- the only pattern proven
    safe during development. Call this once per distinct shape/weight you
    intend to accelerate with ANE, before constructing a TriComputeMatmul
    with enable_ane=True. Returns True on success.
    """
    if not _HAS_ANE:
        return False
    tmp_path = f"/tmp/fusionml_ane_prewarm_{os.getpid()}_{M}_{K}_{N}.npy"
    np.save(tmp_path, np.ascontiguousarray(b, dtype=np.float32))
    script = (
        "import numpy as np, sys\n"
        "from fusionml._metal.ane_backend import ane_matmul\n"
        f"b = np.load({tmp_path!r})\n"
        f"a = np.zeros(({M}, {K}), dtype=np.float32)\n"
        "ane_matmul(a, b, compute_units='CPU_AND_NE')\n"
        "print('OK')\n"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return res.returncode == 0 and "OK" in res.stdout
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class TriComputeMatmul:
    """
    Stateful scheduler: one instance can serve many different (M, K, N)
    shapes safely across a process's lifetime. ANE is only ever used for a
    shape that `is_ane_prewarmed()` confirms is already disk-cached -- this
    process will LOAD that cached model but never compile one inline (see
    module docstring for why).

    Thread-safe for concurrent calls with DIFFERENT shapes (each shape gets
    its own lock-protected state); concurrent calls with the SAME shape
    share one ThrottleGuard and are serialized through a per-shape lock.
    """

    def __init__(self, enable_ane: bool = False):
        self.enable_ane = enable_ane and _HAS_ANE
        if enable_ane:
            import warnings
            warnings.warn(
                "TriComputeMatmul(enable_ane=True): ANE support is EXPERIMENTAL "
                "and has caused unrecoverable process crashes (segfault) under "
                "multiple conditions during development, including after enough "
                "prior unrelated MLX activity in the same process -- a condition "
                "this module cannot detect or guard against for an arbitrary "
                "caller. Do not use in any path where a crash is unacceptable. "
                "See the module docstring for details.",
                RuntimeWarning, stacklevel=2,
            )
        self._shape_state: Dict[Tuple[int, int, int], dict] = {}
        self._state_lock = threading.Lock()
        self._ane_locked_shape: Optional[Tuple[int, int, int]] = None

    # A single first-call measurement proved too noisy on hardware with this
    # much documented thermal/contention variance (see
    # project_tricompute_ane_findings.md) to make an irrevocable "trust this
    # split forever" decision -- different shapes flip pass/fail across
    # otherwise-identical runs with a one-shot check. PROBATION_CALLS
    # measurements are averaged (median) before committing to a fallback
    # decision, so a single unlucky sample can't lock in the wrong answer.
    PROBATION_CALLS = 3

    def _get_shape_state(self, key: Tuple[int, int, int]) -> dict:
        with self._state_lock:
            if key not in self._shape_state:
                self._shape_state[key] = {
                    "guard": ThrottleGuard(),
                    "ratios": dict(_DEFAULT_RATIOS),
                    "calibration_samples": [],
                    "lock": threading.Lock(),
                    "naive_fallback": False,   # set True if splitting proved not worth it for this shape
                    "probation_naive_ms": [],
                    "probation_split_ms": [],
                }
            return self._shape_state[key]

    def _shape_may_use_ane(self, key: Tuple[int, int, int], total_rows: int, b: np.ndarray) -> bool:
        if not self.enable_ane or total_rows < MIN_ROWS_FOR_ANE:
            return False
        if self._ane_locked_shape is not None and self._ane_locked_shape != key:
            return False  # a different shape already claimed ANE this process
        if not is_ane_prewarmed(*key, b):
            return False  # never compile inline -- see module docstring
        self._ane_locked_shape = key
        return True

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            raise ValueError(f"shape mismatch: {a.shape} @ {b.shape}")
        key = (M, K, N)

        if M < MIN_ROWS_TO_SPLIT or not _HAS_MLX:
            return a @ b  # too small to bother, or no GPU backend available

        state = self._get_shape_state(key)
        with state["lock"]:
            if state["naive_fallback"]:
                return a @ b

            # MIN_ROWS_TO_SPLIT only guards against too few ROWS -- it can't
            # catch a shape with many rows but tiny per-row compute (small
            # K/N), where fixed threading + MLX-dispatch overhead dominates
            # regardless of row count. Measuring a naive baseline during
            # PROBATION_CALLS and comparing catches that case directly instead
            # of guessing at more size heuristics. This was found necessary
            # during development: a (6000,300,900) shape was 65x SLOWER via
            # splitting than plain `a @ b` despite having 6000 rows.
            in_probation = len(state["probation_split_ms"]) < self.PROBATION_CALLS
            naive_result = None
            if in_probation:
                t0 = time.perf_counter()
                naive_result = a @ b
                state["probation_naive_ms"].append((time.perf_counter() - t0) * 1000.0)

            use_ane = self._shape_may_use_ane(key, M, b)
            ratios = dict(state["ratios"])
            if not use_ane:
                ratios["ane"] = 0.0
            rows = state["guard"].recommend_rows(M, ratios)
            wall_ms, result, per_unit_ms, actual_rows = self._dispatch(a, b, rows, use_ane)
            state["guard"].record(wall_ms)
            self._maybe_refine_calibration(state, per_unit_ms, actual_rows, M)

            if in_probation:
                state["probation_split_ms"].append(wall_ms)
                if len(state["probation_split_ms"]) == self.PROBATION_CALLS:
                    # Decide once, from the MEDIAN of several samples rather
                    # than a single noisy measurement -- require the split to
                    # be meaningfully faster (25% margin) before trusting it
                    # long-term.
                    med_naive = sorted(state["probation_naive_ms"])[self.PROBATION_CALLS // 2]
                    med_split = sorted(state["probation_split_ms"])[self.PROBATION_CALLS // 2]
                    if med_split > med_naive * 0.75:
                        state["naive_fallback"] = True
                return naive_result
        return result

    def _dispatch(self, a, b, rows: Dict[str, int], use_ane: bool):
        gpu_rows, cpu_rows, ane_rows = rows["gpu"], rows["cpu"], rows["ane"]
        a_gpu = a[:gpu_rows] if gpu_rows > 0 else None
        a_cpu = a[gpu_rows:gpu_rows + cpu_rows] if cpu_rows > 0 else None
        a_ane = a[gpu_rows + cpu_rows:] if ane_rows > 0 else None

        out = {}
        per_unit_ms = {}

        def timed(key, fn):
            t0 = time.perf_counter()
            fn()
            per_unit_ms[key] = (time.perf_counter() - t0) * 1000.0

        def gpu_task():
            def _run():
                r = mx.array(a_gpu) @ mx.array(b)
                mx.eval(r)
                out["gpu_mx"] = r  # readback deferred to main thread -- see below
            timed("gpu", _run)

        def cpu_task():
            timed("cpu", lambda: out.__setitem__("cpu", a_cpu @ b))

        def ane_task():
            timed("ane", lambda: out.__setitem__("ane", ane_matmul(a_ane, b, compute_units="CPU_AND_NE")))

        threads = []
        if a_gpu is not None:
            threads.append(threading.Thread(target=gpu_task))
        if a_cpu is not None:
            threads.append(threading.Thread(target=cpu_task))
        if a_ane is not None and use_ane:
            threads.append(threading.Thread(target=ane_task))

        # wall_ms spans threads AND the readback/concat below -- it must equal
        # what the caller actually experiences end-to-end, or the naive-fallback
        # safety net silently underestimates cost for shapes with large output
        # arrays (found in testing: an 8192x6400 result's GPU->numpy readback
        # was expensive enough to make the "split" path net slower than naive,
        # but that cost was invisible to the fallback check when this timer
        # stopped at thread-join instead of here).
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Metal readback (mx.array -> numpy) done single-threaded, after all
        # threads join. Doing this from within gpu_task while ANE/CPU threads
        # were concurrently active caused a native SIGSEGV during development
        # -- keep this ordering.
        if "gpu_mx" in out:
            out["gpu"] = np.array(out["gpu_mx"])

        pieces = [out[k] for k in ("gpu", "cpu", "ane") if k in out]
        result = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return wall_ms, result, per_unit_ms, {"gpu": gpu_rows, "cpu": cpu_rows, "ane": ane_rows}

    def _maybe_refine_calibration(self, state: dict, per_unit_ms: dict, rows: dict, total_rows: int):
        """
        Online rebalancing toward equalized per-unit finish time, using the
        same feedback approach validated in tricompute_adaptive_search.py:
        real per-unit rates under concurrent contention differ meaningfully
        from isolated/solo calibration, so this adapts from live measurements
        rather than relying on a one-shot offline calibration.

        Runs every REFINE_EVERY calls per shape so a handful of noisy
        individual measurements don't cause oscillation; ANE's row count is
        deliberately left untouched here -- ThrottleGuard.recommend_rows()
        already owns ANE's on/off decision, and this function only
        rebalances the GPU/CPU split of whatever ANE leaves behind.
        """
        REFINE_EVERY = 10
        samples = state["calibration_samples"]
        samples.append((per_unit_ms, rows))
        if len(samples) > 200:
            del samples[:-200]
        if len(samples) % REFINE_EVERY != 0:
            return

        recent = samples[-REFINE_EVERY:]
        avg_rate = {}  # rows per ms, averaged over recent calls, per unit
        for unit in ("gpu", "cpu"):
            rates = [r[unit] / t[unit] for t, r in recent if unit in t and t[unit] > 0 and r[unit] > 0]
            if rates:
                avg_rate[unit] = sum(rates) / len(rates)
        if len(avg_rate) < 2:
            return

        gpu_cpu_total = state["ratios"]["gpu"] + state["ratios"]["cpu"]
        if gpu_cpu_total <= 0:
            return
        target_t = (gpu_cpu_total * total_rows) / (avg_rate["gpu"] + avg_rate["cpu"])
        new_gpu_share = (avg_rate["gpu"] * target_t / total_rows)
        new_cpu_share = (avg_rate["cpu"] * target_t / total_rows)
        norm = (new_gpu_share + new_cpu_share) / gpu_cpu_total if (new_gpu_share + new_cpu_share) > 0 else 1.0
        state["ratios"]["gpu"] = new_gpu_share / norm
        state["ratios"]["cpu"] = new_cpu_share / norm


# Module-level default instance + convenience function, matching the
# ergonomics of numpy's global RNG: `fusionml.smart_matmul(a, b)` works with
# no setup (CPU+GPU only, safe by default), while `fusionml.TriComputeMatmul()`
# is available for explicit per-caller control (e.g. enable_ane=True after
# prewarming, or separate schedulers for unrelated parts of a larger program).
_default_scheduler = TriComputeMatmul(enable_ane=False)


def smart_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Drop-in replacement for `a @ b` using the shared default tri-compute
    scheduler (CPU+GPU, safe by default -- see module docstring for ANE)."""
    return _default_scheduler(a, b)
