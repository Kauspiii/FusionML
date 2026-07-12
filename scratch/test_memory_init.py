import mlx.core as mx
import resource

DIM = 4096
FFN_DIM = 14336

def check_memory(init_mode):
    # Reset peak memory trace if possible (not possible in python, but we can compare raw RSS increase)
    if init_mode == "np_cast":
        import numpy as np
        s = 0.02
        w = {
            'w_q':    mx.array(np.random.randn(DIM, DIM).astype(np.float32) * s).astype(mx.float16),
            'w_k':    mx.array(np.random.randn(DIM, DIM).astype(np.float32) * s).astype(mx.float16),
            'w_v':    mx.array(np.random.randn(DIM, DIM).astype(np.float32) * s).astype(mx.float16),
            'w_o':    mx.array(np.random.randn(DIM, DIM).astype(np.float32) * s).astype(mx.float16),
            'w_gate': mx.array(np.random.randn(DIM, FFN_DIM).astype(np.float32) * s).astype(mx.float16),
            'w_up':   mx.array(np.random.randn(DIM, FFN_DIM).astype(np.float32) * s).astype(mx.float16),
            'w_down': mx.array(np.random.randn(FFN_DIM, DIM).astype(np.float32) * s).astype(mx.float16),
        }
    else:
        # Direct MLX FP16 initialization
        w = {
            'w_q':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
            'w_k':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
            'w_v':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
            'w_o':    mx.random.normal((DIM, DIM), dtype=mx.float16) * 0.02,
            'w_gate': mx.random.normal((DIM, FFN_DIM), dtype=mx.float16) * 0.02,
            'w_up':   mx.random.normal((DIM, FFN_DIM), dtype=mx.float16) * 0.02,
            'w_down': mx.random.normal((FFN_DIM, DIM), dtype=mx.float16) * 0.02,
        }
    mx.eval(*w.values())
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    print(f"Init mode: {init_mode:<10} | Peak memory: {peak_mb:.2f} MB")

import sys
if len(sys.argv) > 1:
    check_memory(sys.argv[1])
