# generate_and_benchmark_all.py — Generates static linear models for different batch sizes and runs the Swift benchmark.
import os
import subprocess
import torch
import coremltools as ct

class StaticLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x):
        return self.linear(x)

def generate_model(batch_size, in_features=2048, out_features=2048):
    model_name = f"models/static_linear_{batch_size}.mlpackage"
    if os.path.exists(model_name):
        print(f"Model {model_name} already exists. Skipping generation.")
        return model_name
        
    print(f"Generating model for batch_size={batch_size}...")
    model = StaticLinearModel(in_features, out_features).eval()
    x = torch.randn(batch_size, in_features)
    traced = torch.jit.trace(model, x)
    
    cml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=x.shape)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13
    )
    
    cml_model.save(model_name)
    print(f"Saved {model_name}")
    return model_name

def write_swift_benchmark(batch_sizes):
    swift_code = """
import Foundation
import CoreML

func runBenchmark(batchSize: Int) throws {
    let basePath = "/Users/ommohite/Documents/Programming/FusionML/models"
    let modelURL = URL(fileURLWithPath: "\\(basePath)/static_linear_\\(batchSize).mlpackage")
    
    let compiledURL = try MLModel.compileModel(at: modelURL)
    let config = MLModelConfiguration()
    config.computeUnits = .all
    
    let model = try MLModel(contentsOf: compiledURL, configuration: config)
    
    let B = batchSize
    let K = 2048
    let O = 2048
    
    let xArray = try MLMultiArray(shape: [B, K] as [NSNumber], dataType: .float16)
    xArray.withUnsafeMutableBytes { ptr, strides in
        let dest = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
        for i in 0..<(B * K) {
            dest[i] = Float16(1.0)
        }
    }
    
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "x": MLFeatureValue(multiArray: xArray)
    ])
    
    // Warmup
    for _ in 0..<5 {
        _ = try model.prediction(from: input)
    }
    
    var times: [Double] = []
    for _ in 0..<30 {
        let start = CFAbsoluteTimeGetCurrent()
        let output = try model.prediction(from: input)
        let end = CFAbsoluteTimeGetCurrent()
        times.append((end - start) * 1000)
        _ = output.featureValue(for: "var_5")
    }
    
    times.sort()
    let medianLat = times[times.count / 2]
    let ops = 2.0 * Double(B * K * O)
    let gflops = (ops / medianLat) / 1_000_000
    let tflops = gflops / 1000.0
    
    print(String(format: "  %4d | %8.3f ms | %10.1f GFLOPS | %8.3f TFLOPS", B, medianLat, gflops, tflops))
}

print("==============================================================")
print("  Batch|   Latency  |    GFLOPS     |   TFLOPS   ")
print("==============================================================")
"""
    
    for b in batch_sizes:
        swift_code += f"try? runBenchmark(batchSize: {b})\n"
        
    swift_code += "print(\"==============================================================\")\n"
    
    with open("scratch/run_sweep.swift", "w") as f:
        f.write(swift_code)
    print("Written scratch/run_sweep.swift")

def main():
    batch_sizes = [128, 256, 512, 1024]
    for b in batch_sizes:
        generate_model(b)
        
    write_swift_benchmark(batch_sizes)
    
    # Run the Swift sweep
    print("\nRunning Swift performance sweep on ANE...")
    result = subprocess.run(["swift", "scratch/run_sweep.swift"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:\n", result.stderr)

if __name__ == "__main__":
    main()
