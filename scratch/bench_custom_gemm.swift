// Benchmark custom register-blocked GEMM vs MPS
import FusionML
import Foundation

func measure(_ label: String, warmup: Int = 5, iterations: Int = 20, _ op: () throws -> Void) rethrows -> Double {
    for _ in 0..<warmup { try op() }
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations { try op() }
    let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
    return ms
}

print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
print("  FusionML Custom GEMM (Register-Blocked) vs MPS Benchmark")
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))

let M1_GPU_PEAK: Double = 2600.0 // GFLOPS

for size in [512, 1024, 2048] {
    print("\nSize: \(size)x\(size)")
    let a = try! Tensor.random([size, size])
    let b = try! Tensor.random([size, size])
    let flops = 2.0 * Double(size * size * size)
    
    // MPS Path
    let mps_ms = try! measure("MPS") {
        let _ = try GPUEngine.shared.matmulMPS(a, b)
        GPUEngine.shared.sync()
    }
    let mps_gflops = (flops / mps_ms) / 1_000_000
    let mps_util = (mps_gflops / M1_GPU_PEAK) * 100
    print(String(format: "  MPS:         %7.2f ms  %6.0f GFLOPS  %4.0f%% utilization", mps_ms, mps_gflops, mps_util))
    
    // Custom GEMM Path
    let custom_ms = try! measure("Custom GEMM") {
        let _ = try GPUEngine.shared.matmul(a, b)
        GPUEngine.shared.sync()
    }
    let custom_gflops = (flops / custom_ms) / 1_000_000
    let custom_util = (custom_gflops / M1_GPU_PEAK) * 100
    let speedup = mps_ms / custom_ms
    print(String(format: "  Custom GEMM: %7.2f ms  %6.0f GFLOPS  %4.0f%% utilization  → %.2fx speedup", custom_ms, custom_gflops, custom_util, speedup))
}
