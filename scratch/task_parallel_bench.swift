import FusionML
import Foundation
import Accelerate
import Metal

func measure(_ label: String, warmup: Int = 10, iterations: Int = 50, _ op: () throws -> Void) rethrows -> [Double] {
    for _ in 0..<warmup { try op() }
    var times: [Double] = []
    for _ in 0..<iterations {
        let start = CFAbsoluteTimeGetCurrent()
        try op()
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        times.append(ms)
    }
    return times
}

func printStats(for times: [Double], label: String) {
    let median = times.sorted()[times.count / 2]
    let mean = times.reduce(0, +) / Double(times.count)
    let sumSq = times.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
    let std = sqrt(sumSq / Double(times.count))
    let ci95 = 1.96 * std / sqrt(Double(times.count))
    print(String(format: "  %-25s | Median: %8.2f ms | Mean: %8.2f ± %5.2f ms | Std: %5.2f ms", label, median, mean, ci95, std))
}

let M = 4096
let N = 4096
let K = 4096

print("=========================================================================")
print("  Heterogeneous CPU-GPU Concurrent Split Matmul Sweep (M=N=K=4096)")
print("=========================================================================")

let a = try! Tensor.random([M, K])
let b = try! Tensor.random([K, N])
let result = try! Tensor(shape: [M, N], dtype: .float32)

let aPtr = a.buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
let bPtr = b.buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)

let aMetal = a.metalBuffer
let bMetal = b.metalBuffer
let rMetal = result.metalBuffer

// GPU Full Baseline (k/M = 0.0)
let gpuTimes = try! measure("GPU-only") {
    _ = try GPUEngine.shared.matmulRawMPS(
        a: aMetal,
        b: bMetal,
        result: rMetal,
        M: M, N: N, K: K,
        aOffset: 0,
        resultOffset: 0,
        waitUntilCompleted: true
    )
}
printStats(for: gpuTimes, label: "GPU-only (Full)")

// CPU Full Baseline (k/M = 1.0)
let cpuTimes = try! measure("CPU-only") {
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        Int32(M), Int32(N), Int32(K),
        1.0,
        aPtr, Int32(K),
        bPtr, Int32(N),
        0.0,
        rPtr, Int32(N)
    )
}
printStats(for: cpuTimes, label: "CPU-only (Full)")

print("-------------------------------------------------------------------------")
print("  Sweeping Split Ratios (k = CPU rows, M - k = GPU rows)")
print("-------------------------------------------------------------------------")

let ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for ratio in ratios {
    let cpuRows = Int(Double(M) * ratio)
    let gpuRows = M - cpuRows
    
    let splitTimes = try! measure("Split \(ratio)") {
        let group = DispatchGroup()
        
        // 1. Dispatch GPU portion asynchronously (wait = true inside the queue)
        group.enter()
        DispatchQueue.global(qos: .userInteractive).async {
            _ = try! GPUEngine.shared.matmulRawMPS(
                a: aMetal,
                b: bMetal,
                result: rMetal,
                M: gpuRows,
                N: N,
                K: K,
                aOffset: cpuRows * K * 4,
                resultOffset: cpuRows * N * 4,
                waitUntilCompleted: true
            )
            group.leave()
        }
        
        // 2. Compute CPU portion synchronously on this thread
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(cpuRows), Int32(N), Int32(K),
            1.0,
            aPtr, Int32(K),
            bPtr, Int32(N),
            0.0,
            rPtr, Int32(N)
        )
        
        // 3. Sync both
        group.wait()
    }
    
    printStats(for: splitTimes, label: String(format: "Split Ratio %.1f", ratio))
}

print("=========================================================================")
