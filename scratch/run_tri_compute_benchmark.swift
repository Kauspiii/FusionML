// run_tri_compute_benchmark.swift — Runs genuine co-execution benchmarks on Apple Silicon.
// Measures raw GPU-only, CPU-only, ANE-only, and 3-Way Tri-Compute co-execution GFLOPS.

import Foundation
import FusionML

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public func print(_ items: Any..., separator: String = " ", terminator: String = "\n") {
    let output = items.map { "\($0)" }.joined(separator: separator)
    Swift.print(output, terminator: terminator)
    fflush(stdout)
}

func runBenchmark() throws {
    print("🚀 Initializing FusionML for Tri-Compute Benchmark...")
    Fusion.initialize()
    
    let scheduler = SmartScheduler3.shared
    let sizes = [512, 1024, 2048]
    
    print("\n📐 Phase 1: Calibrating GPU, CPU, and ANE throughput profiles...")
    try scheduler.calibrate(sizes: sizes)
    
    print("\n⚡ Phase 2: Running 3-Way Tri-Compute split matmuls...")
    for size in sizes {
        print("-" .padding(toLength: 60, withPad: "-", startingAt: 0))
        print("  Size: \(size)x\(size)")
        
        let a = try Tensor.random([size, size])
        let b = try Tensor.random([size, size])
        
        // GPU-Only
        let startG = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            _ = try GPUEngine.shared.matmulMPS(a, b)
        }
        GPUEngine.shared.sync()
        let latG = (CFAbsoluteTimeGetCurrent() - startG) * 1000 / 10
        let gflopsG = (2.0 * Double(size * size * size) / latG) / 1_000_000
        
        // CPU-Only
        let startC = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            _ = try Tensor.matmul(a, b)
        }
        let latC = (CFAbsoluteTimeGetCurrent() - startC) * 1000 / 10
        let gflopsC = (2.0 * Double(size * size * size) / latC) / 1_000_000
        
        // 3-Way Split (Tri-Compute)
        let startS = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            _ = try scheduler.smartMatmul(a, b)
        }
        let latS = (CFAbsoluteTimeGetCurrent() - startS) * 1000 / 10
        let gflopsS = (2.0 * Double(size * size * size) / latS) / 1_000_000
        
        let ratioG = (scheduler.optimalSplitRatio(for: size).gpu * 100)
        let ratioC = (scheduler.optimalSplitRatio(for: size).cpu * 100)
        let ratioA = (scheduler.optimalSplitRatio(for: size).ane * 100)
        
        print(String(format: "    GPU-Only:    %.3f ms (%.0f GFLOPS)", latG, gflopsG))
        print(String(format: "    CPU-Only:    %.3f ms (%.0f GFLOPS)", latC, gflopsC))
        print(String(format: "    Tri-Compute: %.3f ms (%.0f GFLOPS)", latS, gflopsS))
        print(String(format: "    Work split:  GPU: %.0f%% | CPU: %.0f%% | ANE: %.0f%%", ratioG, ratioC, ratioA))
        
        let speedup = latG / latS
        if speedup > 1.0 {
            print(String(format: "    ✅ Speedup vs GPU: %.2fx", speedup))
        } else {
            print(String(format: "    ⚡ Split overhead is higher by %.2fx (GPU is faster for this size)", 1/speedup))
        }
    }
    
    print("\n📊 Final scheduler profiles:")
    scheduler.printStats()
}

do {
    try runBenchmark()
} catch {
    print("❌ Fatal error: \(error)")
}
