# test_static_linear_ane.py — Compiles a Linear layer with static weights for ANE and benchmarks it.
import os
import time
import numpy as np
import torch
import coremltools as ct

class StaticLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Static weights inside the model
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x):
        return self.linear(x)

def main():
    os.makedirs("models", exist_ok=True)
    
    in_features = 2048
    out_features = 2048
    batch_size = 512
    
    print(f"Generating static weight Linear model: input [{batch_size}, {in_features}] x weights [{out_features}, {in_features}]...")
    model = StaticLinearModel(in_features, out_features).eval()
    
    # Input example
    x = torch.randn(batch_size, in_features)
    
    # Trace
    traced = torch.jit.trace(model, x)
    
    # Convert to CoreML forcing ANE
    cml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=x.shape)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13
    )
    
    out_path = "models/static_linear_2048.mlpackage"
    cml_model.save(out_path)
    print(f"Saved CoreML model to {out_path}")
    
    # Load and run in Python to warm up and verify
    print("Loading model for Python benchmark...")
    loaded_model = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    
    # Prepare input dict
    x_np = np.random.randn(batch_size, in_features).astype(np.float32)
    inputs = {"x": x_np}
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = loaded_model.predict(inputs)
        
    # Measure
    print("Benchmarking...")
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = loaded_model.predict(inputs)
        times.append((time.perf_counter() - t0) * 1000)
        
    lat = np.median(times)
    # 2 * B * I * O
    ops = 2.0 * batch_size * in_features * out_features
    gflops = (ops / lat) / 1_000_000
    tflops = gflops / 1000.0
    
    print("-" * 60)
    print(f"Python CoreML Prediction Latency: {lat:.3f} ms")
    print(f"Throughput: {gflops:.1f} GFLOPS ({tflops:.3f} TFLOPS)")
    print("-" * 60)

if __name__ == "__main__":
    main()
