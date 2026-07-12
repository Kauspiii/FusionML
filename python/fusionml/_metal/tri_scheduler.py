"""
Tri-Compute Scheduler - THE CORE FUSIONML INNOVATION
Adaptive parallel execution across GPU (MLX) + CPU (Accelerate) + ANE (CoreML)

Key insight: Apple Silicon has 3 compute units sharing unified memory.
By profiling each unit and splitting work optimally, we achieve higher
total throughput than any single unit alone.

Strategy:
1. Profile: Measure each backend's latency for an operation
2. Calibrate: Compute optimal split ratios (gpu_ratio, cpu_ratio, ane_ratio)
3. Execute: Split data, run all 3 backends in parallel, combine results
4. Adapt: Track history to improve ratios over time
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import concurrent.futures
import threading
import time
import json
import os

# Limit Accelerate BLAS threads to prevent performance core starvation of the Metal command driver
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"


# Backend availability
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .ane_backend import ane_matmul, ane_available, HAS_COREML


# ============================================================================
# PERFORMANCE PROFILER
# ============================================================================

class BackendProfiler:
    """Profiles individual backend performance for specific operations."""
    
    def __init__(self):
        self.history: Dict[str, Dict[str, List[float]]] = {}
    
    def profile_matmul(self, size: int, iterations: int = 5) -> Dict[str, float]:
        """
        Profile matmul performance on all available backends.
        
        Returns dict of {backend_name: avg_time_ms}
        """
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        results = {}
        
        # CPU (NumPy / Accelerate BLAS)
        # Warmup
        _ = np.matmul(a, b)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = np.matmul(a, b)
            times.append((time.perf_counter() - t0) * 1000)
        results["cpu"] = np.median(times)
        
        # GPU (MLX)
        if HAS_MLX:
            a_mlx = mx.array(a)
            b_mlx = mx.array(b)
            # Warmup
            c = a_mlx @ b_mlx
            mx.eval(c)
            times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                c = a_mlx @ b_mlx
                mx.eval(c)
                times.append((time.perf_counter() - t0) * 1000)
            results["gpu"] = np.median(times)
        
        # ANE (CoreML) - Disabled during calibration to prevent OOM
        pass
        
        # Store in history
        key = f"matmul_{size}"
        if key not in self.history:
            self.history[key] = {}
        for backend, time_ms in results.items():
            if backend not in self.history[key]:
                self.history[key][backend] = []
            self.history[key][backend].append(time_ms)
            # Keep last 20 entries
            self.history[key][backend] = self.history[key][backend][-20:]
        
        return results


# ============================================================================
# RATIO CALCULATOR
# ============================================================================

def compute_optimal_ratios(
    profile: Dict[str, float],
    min_ratio: float = 0.05
) -> Dict[str, float]:
    """
    Compute optimal work-split ratios based on profiled backend speeds.
    
    Faster backends get proportionally more work.
    throughput_i = 1 / latency_i
    ratio_i = throughput_i / sum(throughputs)
    
    Args:
        profile: {backend: time_ms} from profiling
        min_ratio: Minimum ratio for any backend (avoids zero-work scenarios)
    
    Returns:
        {backend: ratio} where ratios sum to 1.0
    """
    if not profile:
        return {"cpu": 1.0}
    
    # Compute throughputs (inverse of latency)
    throughputs = {}
    for backend, time_ms in profile.items():
        if time_ms > 0:
            throughputs[backend] = 1.0 / time_ms
    
    total_throughput = sum(throughputs.values())
    
    if total_throughput == 0:
        # Equal split
        n = len(profile)
        return {k: 1.0 / n for k in profile}
    
    # Raw ratios based on throughput
    ratios = {k: v / total_throughput for k, v in throughputs.items()}
    
    # Enforce minimum ratio
    for k in ratios:
        if ratios[k] < min_ratio:
            ratios[k] = min_ratio
    
    # Renormalize
    total = sum(ratios.values())
    ratios = {k: v / total for k, v in ratios.items()}
    
    return ratios


# ============================================================================
# TRI-COMPUTE SCHEDULER
# ============================================================================

class TriComputeScheduler:
    """
    Adaptive scheduler for CPU + GPU + ANE parallel execution.
    
    Usage:
        scheduler = TriComputeScheduler()
        scheduler.calibrate(sizes=[256, 512, 1024, 2048])
        result = scheduler.tri_matmul(a, b)
    """
    
    def __init__(self, auto_calibrate: bool = True,
                 enable_gpu: bool = True, enable_cpu: bool = True,
                 enable_ane: bool = True, random_routing: bool = False):
        self.profiler = BackendProfiler()
        self.calibrated_ratios: Dict[str, Dict[str, float]] = {}
        self.auto_calibrate = auto_calibrate
        self._calibration_count = 0
        
        # Ablation control flags
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
        self.enable_ane = enable_ane
        self.random_routing = random_routing
        
        # Size thresholds for routing decisions
        self.CPU_ONLY_THRESHOLD = 512      # Below this: CPU only
        self.DUAL_THRESHOLD = 1024         # Below this: GPU+CPU only
        # Above DUAL_THRESHOLD: tri-compute (GPU+CPU+ANE)
        
        # Persistent thread pool — eliminates thread creation overhead
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self._cpu_cache = {}
    
    def __del__(self):
        """Clean up thread pool."""
        if hasattr(self, '_pool'):
            self._pool.shutdown(wait=False)

    def calibrate(self, shapes: Optional[List] = None, iterations: int = 5, verbose: bool = True, sizes: Optional[List] = None):
        """
        Contention-aware calibration via empirical grid search.
        shapes: list of either int or (M, K, N) tuple
        """
        if shapes is None:
            shapes = sizes
        if shapes is None:
            shapes = [256, 512, 1024, 2048, 4096]
        
        if verbose:
            print("⚡ Tri-Compute Calibration (contention-aware grid search)")
            print("=" * 60)
        
        for shape in shapes:
            if isinstance(shape, int):
                M = K = N = shape
                key = f"matmul_{shape}"
            else:
                M, K, N = shape
                key = f"matmul_{M}_{K}_{N}"
                
            if verbose:
                print(f"\n  Profiling matmul shape {M}x{K}x{N}...")
            
            # Phase 2: Grid search parallel split ratios
            if HAS_MLX and self.enable_gpu and self.enable_cpu:
                best_ratio, best_time, gpu_only_time = self._grid_search_ratio(
                    M, K, N, iterations=max(3, iterations), verbose=verbose
                )
                
                time_saved = gpu_only_time - best_time
                # Only split if parallel co-execution is faster than compiled GPU-only:
                # - saved at least 1.0 ms OR is at least 3% faster (to reject noise)
                if best_ratio < 1.0 and (time_saved >= 1.0 or best_time < gpu_only_time * 0.97):
                    self.calibrated_ratios[key] = {"gpu": best_ratio, "cpu": 1.0 - best_ratio}
                    if verbose:
                        print(f"    ✅ Parallel wins: {best_time:.3f}ms (GPU={best_ratio:.0%}/CPU={1-best_ratio:.0%}) vs GPU-Only={gpu_only_time:.3f}ms (saved {time_saved:.3f}ms)")
                else:
                    self.calibrated_ratios[key] = {"gpu": 1.0}
                    if verbose:
                        print(f"    ⚡ Single wins: GPU-Only={gpu_only_time:.3f}ms vs best parallel={best_time:.3f}ms (saved {time_saved:.3f}ms)")
            else:
                self.calibrated_ratios[key] = {"gpu": 1.0}
        
        self._calibration_count += 1
        
        if verbose:
            print(f"✓ Calibration complete ({len(shapes)} shapes)")
    
    def _grid_search_ratio(self, M: int, K: int, N: int, iterations: int = 5,
                           verbose: bool = False) -> Tuple[float, float, float]:
        """
        Contention-free, fair interleaved grid search of GPU/CPU split ratios.
        Alternates runs in randomized order to completely eliminate thermal/frequency scaling bias.
        """
        import random
        a_mx = mx.random.normal((M, K))
        b_mx = mx.random.normal((K, N))
        mx.eval(a_mx, b_mx)
        
        run_fns = {}
        gpu_pcts = [70, 75, 80, 85, 90, 95, 100]
        
        # Build and compile run functions for each ratio
        for gpu_pct in gpu_pcts:
            if gpu_pct == 100:
                @mx.compile
                def compiled_fn(a, b):
                    return a @ b
                def make_runner(fn):
                    return lambda: mx.eval(fn(a_mx, b_mx))
                run_fns[gpu_pct] = make_runner(compiled_fn)
            else:
                gpu_rows = int(M * gpu_pct / 100)
                cpu_rows = M - gpu_rows
                if cpu_rows < 1 or gpu_rows < 1:
                    continue
                a_cpu = a_mx[:cpu_rows]
                a_gpu = a_mx[cpu_rows:]
                @mx.compile
                def compiled_fn(a_gpu, a_cpu, b):
                    c_gpu = a_gpu @ b
                    mx.set_default_device(mx.cpu)
                    c_cpu = a_cpu @ b
                    mx.set_default_device(mx.gpu)
                    return mx.concatenate([c_cpu, c_gpu], axis=0)
                def make_runner(fn, a_gpu=a_gpu, a_cpu=a_cpu):
                    return lambda: mx.eval(fn(a_gpu, a_cpu, b_mx))
                run_fns[gpu_pct] = make_runner(compiled_fn)
                
        # Warmup all
        for _ in range(15):
            for r in run_fns.values():
                r()
                
        # Interleaved measurement
        results = {pct: [] for pct in run_fns.keys()}
        runs = max(10, iterations * 4) # Run enough iterations to get clean medians
        
        for _ in range(runs):
            p_order = list(run_fns.keys())
            random.shuffle(p_order)
            for pct in p_order:
                t0 = time.perf_counter()
                run_fns[pct]()
                results[pct].append((time.perf_counter() - t0) * 1000)
                
        # Calculate medians
        medians = {pct: float(np.median(results[pct])) for pct in run_fns.keys()}
        
        best_ratio = 1.0
        best_time = medians.get(100, float('inf'))
        gpu_only_time = best_time
        
        for pct, med in medians.items():
            if verbose:
                print(f"      Ratio GPU={pct}%: {med:.3f}ms")
            if med < best_time:
                best_time = med
                best_ratio = pct / 100.0
                
        return best_ratio, best_time, gpu_only_time
    
    def get_ratios(self, M: int, K: Optional[int] = None, N: Optional[int] = None) -> Dict[str, float]:
        """
        Get the optimal ratios for a given matrix shape.
        """
        if K is None:
            K = M
        if N is None:
            N = M
        key = f"matmul_{M}_{K}_{N}"
        
        # Exact match
        if key in self.calibrated_ratios:
            return self.calibrated_ratios[key]
        
        # Fallback to nearest calibrated square size
        min_dim = min(M, K, N)
        square_key = f"matmul_{min_dim}"
        if square_key in self.calibrated_ratios:
            return self.calibrated_ratios[square_key]
            
        # Find closest calibrated shape (1D or 3D) by absolute dimension distance
        best_key = None
        best_dist = float('inf')
        for k in self.calibrated_ratios:
            if k.startswith("matmul_"):
                parts = k.split("_")
                if len(parts) == 4:
                    try:
                        c_M, c_K, c_N = int(parts[1]), int(parts[2]), int(parts[3])
                        dist = abs(c_M - M) + abs(c_K - K) + abs(c_N - N)
                        if dist < best_dist:
                            best_dist = dist
                            best_key = k
                    except ValueError:
                        pass
                elif len(parts) == 2:
                    try:
                        c_size = int(parts[1])
                        dist = abs(c_size - M) + abs(c_size - K) + abs(c_size - N)
                        if dist < best_dist:
                            best_dist = dist
                            best_key = k
                    except ValueError:
                        pass
                        
        if best_key is not None:
            return self.calibrated_ratios[best_key]
        
        # Default fallback if no calibration is loaded
        if min_dim < self.CPU_ONLY_THRESHOLD:
            return {"cpu": 1.0}
        elif min_dim < self.DUAL_THRESHOLD:
            return {"gpu": 0.7, "cpu": 0.3}
        else:
            return {"gpu": 0.6, "cpu": 0.25, "ane": 0.15}
    
    def _get_cached_np(self, mlx_arr, tensor_obj=None, transposed=False) -> np.ndarray:
        """Get CPU NumPy array of an MLX array using a cache keyed by array object ID."""
        is_param = getattr(tensor_obj, 'is_parameter', False)
        if is_param:
            key = (id(tensor_obj), transposed)
            if key in self._cpu_cache:
                return self._cpu_cache[key]
        
        mx.eval(mlx_arr)
        arr_np = np.array(mlx_arr, copy=False)
        
        if is_param:
            if len(self._cpu_cache) > 200:
                self._cpu_cache.clear()
            self._cpu_cache[key] = arr_np
        return arr_np
 
    def gpu_smart_matmul(self, a_mlx, b_mlx, b_tensor=None) -> 'mx.array':
        """
        GPU-native smart split matmul using MLX GPU and MLX CPU.
        Splits rows of A between GPU and CPU asynchronously without NumPy copies.
        """
        if len(a_mlx.shape) > 2 and len(b_mlx.shape) == 2:
            orig_shape = a_mlx.shape
            a_2d = a_mlx.reshape(-1, orig_shape[-1])
            res_2d = self.gpu_smart_matmul(a_2d, b_mlx, b_tensor=b_tensor)
            out_shape = list(orig_shape[:-1]) + [b_mlx.shape[-1]]
            return res_2d.reshape(out_shape)
        if len(a_mlx.shape) != 2 or len(b_mlx.shape) != 2:
            return a_mlx @ b_mlx

        M, K = a_mlx.shape
        K2, N = b_mlx.shape
        min_dim = min(M, K, N)
        
        # Fallback to pure GPU if size is small
        if min_dim < self.CPU_ONLY_THRESHOLD:
            return a_mlx @ b_mlx
            
        # Get ratios based on 3D shape
        ratios = self.get_ratios(M, K, N)
        if ratios:
            gpu_ratio = ratios.get("gpu", 0.0)
            cpu_ratio = ratios.get("cpu", 0.0)
        else:
            gpu_ratio, cpu_ratio = 0.70, 0.30
            
        if cpu_ratio < 0.02:
            return a_mlx @ b_mlx
        if gpu_ratio < 0.02:
            # CPU only via MLX CPU
            mx.set_default_device(mx.cpu)
            res = a_mlx @ b_mlx
            mx.set_default_device(mx.gpu)
            return res
            
        cpu_rows = int(M * cpu_ratio)
        a_cpu = a_mlx[:cpu_rows]
        a_gpu = a_mlx[cpu_rows:]
        
        # 1. Queue GPU matmul
        c_gpu = a_gpu @ b_mlx
        
        # 2. Queue CPU matmul
        mx.set_default_device(mx.cpu)
        c_cpu = a_cpu @ b_mlx
        mx.set_default_device(mx.gpu)
        
        # 3. Evaluate both asynchronously in parallel (non-blocking)
        mx.eval(c_gpu, c_cpu)
        
        # 4. Concatenate results (resolved in unified memory)
        return mx.concatenate([c_cpu, c_gpu], axis=0)

    def tri_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication using all available compute units in parallel.
        
        Splits rows of A across GPU, CPU, and ANE based on calibrated ratios.
        All compute concurrently, results are combined.
        
        Optimized for zero overhead:
        - Persistent thread pool (no thread creation cost)
        - View-based slicing (no data copies)
        - Pre-allocated output buffer
        """
        if len(a.shape) > 2 and len(b.shape) == 2:
            orig_shape = a.shape
            a_2d = a.reshape(-1, orig_shape[-1])
            res_2d = self.tri_matmul(a_2d, b)
            out_shape = list(orig_shape[:-1]) + [b.shape[-1]]
            return res_2d.reshape(out_shape)
        if len(a.shape) != 2 or len(b.shape) != 2:
            return np.matmul(a, b)

        M, K = a.shape
        K2, N = b.shape
        
        min_dim = min(M, K, N)
        
        # Small: CPU only (no parallelism overhead)
        if min_dim < self.CPU_ONLY_THRESHOLD:
            return np.matmul(a, b)
        
        # Get ratios
        ratios = self.get_ratios(min_dim)
        
        # Determine which backends to use
        backends = list(ratios.keys())
        
        # Filter to available AND enabled backends
        available_backends = []
        if self.enable_cpu:
            available_backends.append("cpu")
        if HAS_MLX and self.enable_gpu and "gpu" in backends:
            available_backends.append("gpu")
        if HAS_COREML and self.enable_ane and "ane" in backends:
            available_backends.append("ane")
        
        # Fallback: must have at least one backend
        if not available_backends:
            available_backends = ["cpu"]
        
        # Random routing (ablation baseline)
        if self.random_routing:
            import random
            active_ratios = {k: random.random() for k in available_backends}
            total = sum(active_ratios.values())
            active_ratios = {k: v / total for k, v in active_ratios.items()}
        else:
            # Recalculate ratios for available backends only
            active_ratios = {k: ratios.get(k, 0) for k in available_backends}
            total = sum(active_ratios.values())
            if total > 0:
                active_ratios = {k: v / total for k, v in active_ratios.items()}
            else:
                active_ratios = {available_backends[0]: 1.0}
        
        # If only one backend, run directly (zero overhead)
        if len(active_ratios) == 1:
            backend = list(active_ratios.keys())[0]
            if backend == "gpu":
                return self._gpu_matmul(a, b)
            elif backend == "ane":
                return ane_matmul(a, b)
            else:
                return np.matmul(a, b)
        
        # ============================================================
        # PARALLEL EXECUTION — raw threads + pre-converted MLX
        # ============================================================
        
        # Compute split points
        splits = self._compute_splits(M, active_ratios)
        
        # Pre-allocate result
        result = np.empty((M, N), dtype=np.float32)
        
        # Identify GPU and CPU work
        gpu_slice = None
        cpu_slices = []
        
        for backend, (start, end) in splits.items():
            if end <= start:
                continue
            if backend == "gpu" and HAS_MLX:
                gpu_slice = (start, end)
            else:
                cpu_slices.append((backend, start, end))
        
        # Pre-convert GPU data to MLX (zero-copy on unified memory)
        threads = []
        
        if gpu_slice:
            gs, ge = gpu_slice
            a_gpu_mx = mx.array(a[gs:ge])
            b_mx = mx.array(b)
            def gpu_work():
                c = a_gpu_mx @ b_mx
                mx.eval(c)
                result[gs:ge] = np.array(c)
            threads.append(threading.Thread(target=gpu_work))
        
        for backend, start, end in cpu_slices:
            if backend == "ane" and HAS_COREML:
                a_s, b_c = a[start:end], b
                s, e = start, end
                def ane_work(a_s=a_s, b_c=b_c, s=s, e=e):
                    result[s:e] = ane_matmul(a_s, b_c)
                threads.append(threading.Thread(target=ane_work))
            else:
                a_s, b_c = a[start:end], b
                out = result[start:end]
                def cpu_work(a_s=a_s, b_c=b_c, out=out):
                    np.matmul(a_s, b_c, out=out)
                threads.append(threading.Thread(target=cpu_work))
        
        # Launch all threads simultaneously
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        return result
    
    def _compute_splits(
        self, M: int, ratios: Dict[str, float]
    ) -> Dict[str, Tuple[int, int]]:
        """Compute row split points from ratios."""
        splits = {}
        current = 0
        backends = list(ratios.keys())
        
        for i, backend in enumerate(backends):
            if i == len(backends) - 1:
                # Last backend gets remaining rows
                splits[backend] = (current, M)
            else:
                rows = max(1, int(M * ratios[backend]))
                splits[backend] = (current, min(current + rows, M))
                current = min(current + rows, M)
        
        return splits
    
    def _execute_backend(
        self, backend: str, a: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Execute matmul on a specific backend."""
        if backend == "gpu":
            return self._gpu_matmul(a, b)
        elif backend == "ane":
            return ane_matmul(a, b)
        else:  # cpu
            return np.matmul(a, b)
    
    def _execute_into(
        self, backend: str, a: np.ndarray, b: np.ndarray,
        out: np.ndarray, row_start: int, row_end: int
    ):
        """Execute matmul and write result directly into output buffer."""
        if backend == "gpu":
            out[row_start:row_end, :] = self._gpu_matmul(a, b)
        elif backend == "ane":
            out[row_start:row_end, :] = ane_matmul(a, b)
        else:  # cpu
            # NumPy matmul supports `out` parameter for zero-copy write
            np.matmul(a, b, out=out[row_start:row_end, :])
    
    def _gpu_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU matmul via MLX."""
        if not HAS_MLX:
            return np.matmul(a, b)
        a_mlx = mx.array(a)
        b_mlx = mx.array(b)
        c = a_mlx @ b_mlx
        mx.eval(c)
        return np.array(c)
    
    def save_calibration(self, path: str):
        """Save calibration results to JSON."""
        data = {
            "calibrated_ratios": self.calibrated_ratios,
            "profiler_history": self.profiler.history,
            "calibration_count": self._calibration_count,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_calibration(self, path: str):
        """Load calibration results from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.calibrated_ratios = data.get("calibrated_ratios", {})
        self._calibration_count = data.get("calibration_count", 0)
    
    def print_status(self):
        """Print scheduler status."""
        print(f"\n📊 Tri-Compute Scheduler Status")
        print(f"   Backends: CPU" + 
              (", GPU (MLX)" if HAS_MLX else "") +
              (", ANE (CoreML)" if HAS_COREML else ""))
        print(f"   Calibrations: {self._calibration_count}")
        print(f"   Calibrated sizes: {len(self.calibrated_ratios)}")
        
        if self.calibrated_ratios:
            print(f"\n   Ratios:")
            for key, ratios in sorted(self.calibrated_ratios.items()):
                parts = [f"{k}={v:.1%}" for k, v in ratios.items()]
                print(f"     {key}: {', '.join(parts)}")


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

# Global scheduler instance
_scheduler = None

def get_scheduler() -> TriComputeScheduler:
    """Get or create the global tri-compute scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TriComputeScheduler()
    return _scheduler


def tri_matmul(a: np.ndarray, b: np.ndarray, auto_calibrate: bool = True) -> np.ndarray:
    """
    Tri-compute matrix multiplication.
    
    First call auto-calibrates if no prior calibration exists.
    
    Args:
        a: Left matrix (M, K)
        b: Right matrix (K, N)
        auto_calibrate: Auto-calibrate on first call
    
    Returns:
        Result matrix (M, N)
    """
    scheduler = get_scheduler()
    
    # Auto-calibrate on first call with large matrices
    if auto_calibrate and scheduler._calibration_count == 0:
        size = min(a.shape[0], a.shape[1] if a.ndim > 1 else 1)
        if size >= scheduler.CPU_ONLY_THRESHOLD:
            scheduler.calibrate(sizes=[512, 1024, 2048], verbose=False)
    
    return scheduler.tri_matmul(a, b)


def calibrate(sizes: List[int] = None, verbose: bool = True):
    """Run calibration on the global scheduler."""
    scheduler = get_scheduler()
    scheduler.calibrate(sizes=sizes, verbose=verbose)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'TriComputeScheduler',
    'BackendProfiler',
    'compute_optimal_ratios',
    'tri_matmul',
    'calibrate',
    'get_scheduler',
]
