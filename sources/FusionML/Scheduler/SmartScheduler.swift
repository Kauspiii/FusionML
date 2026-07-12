// SmartScheduler - Intelligent Proportional Work Distribution
// Profiles hardware, learns performance, splits work optimally

import Foundation
import Accelerate
import Metal


/// Measured performance for a backend at a specific workload
public struct PerformanceProfile: Codable {
    public var samples: [Double] = []  // Time in ms
    public var totalCount: Int = 0
    
    public var avgTimeMs: Double {
        guard !samples.isEmpty else { return Double.infinity }
        return samples.reduce(0, +) / Double(samples.count)
    }
    
    public var throughput: Double {  // ops per ms
        guard avgTimeMs > 0 else { return 0 }
        return 1.0 / avgTimeMs
    }
    
    mutating func record(_ timeMs: Double) {
        samples.append(timeMs)
        totalCount += 1
        // Keep rolling window of last 20 samples
        if samples.count > 20 {
            samples.removeFirst()
        }
    }
}

/// Smart scheduler that learns and adapts
public final class SmartScheduler: @unchecked Sendable {
    
    public static let shared = SmartScheduler()
    
    // Performance profiles: [operation_size -> [backend -> profile]]
    private var profiles: [String: [HardwareBackend: PerformanceProfile]] = [:]
    // Empirically tuned split ratios: [operation_size -> cpuRatio]
    private var tunedRatios: [String: Double] = [:]
    private let lock = NSLock()
    
    // Calibration status
    private var isCalibrated = false
    
    private init() {
        setenv("VECLIB_MAXIMUM_THREADS", "1", 1)
    }
    
    // MARK: - Calibration
    
    /// Calibrate by measuring actual backend performance and sweeping concurrent split ratios
    public func calibrate(sizes: [Int] = [256, 512, 1024, 2048]) throws {
        print("📐 Calibrating SmartScheduler...")
        
        // Ramping up the DVFS frequencies (warmup)
        print("🔥 Ramping up hardware frequencies (warmup)...")
        for _ in 0..<30 {
            try autoreleasepool {
                let wa = try Tensor.random([1024, 1024])
                let wb = try Tensor.random([1024, 1024])
                _ = try Tensor.matmul(wa, wb)
                _ = try GPUEngine.shared.matmulMPS(wa, wb)
                GPUEngine.shared.sync()
            }
        }
        MemoryManager.shared.clearPool()
        
        for size in sizes {
            try autoreleasepool {
                let a = try Tensor.random([size, size])
                let b = try Tensor.random([size, size])
                
                let key = "matmul_\(size)"
                
                // Measure CPU
                let cpuTime = try measureBackend(.cpu, size: size) {
                    try Tensor.matmul(a, b)
                }
                recordProfile(key: key, backend: .cpu, timeMs: cpuTime)
                
                // Measure GPU (MPS)
                let gpuTime = try measureBackend(.gpu, size: size) {
                    let res = try GPUEngine.shared.matmulMPS(a, b)
                    GPUEngine.shared.sync()
                    return res
                }
                recordProfile(key: key, backend: .gpu, timeMs: gpuTime)
                
                let cpuGFLOPS = (2.0 * Double(size * size * size) / cpuTime) / 1_000_000
                let gpuGFLOPS = (2.0 * Double(size * size * size) / gpuTime) / 1_000_000
                
                // Base split ratio on theoretical throughputs
                let cpuThroughput = 1.0 / cpuTime
                let gpuThroughput = 1.0 / gpuTime
                let total = cpuThroughput + gpuThroughput
                let theoreticalRatio = cpuThroughput / total
                
                // Sweep ratios around theoretical ratio to find actual best
                var candidateRatios = [0.0]
                if theoreticalRatio > 0.02 {
                    let r = theoreticalRatio
                    candidateRatios.append(max(0.02, r - 0.12))
                    candidateRatios.append(max(0.02, r - 0.06))
                    candidateRatios.append(r)
                    candidateRatios.append(min(0.55, r + 0.06))
                    candidateRatios.append(min(0.55, r + 0.12))
                }
                
                var bestRatio = 0.0
                var minTime = gpuTime
                
                for ratio in candidateRatios {
                    let time = try measureSplit(a, b, cpuRatio: ratio)
                    if time < minTime {
                        minTime = time
                        bestRatio = ratio
                    }
                }
                
                lock.lock()
                tunedRatios[key] = bestRatio
                lock.unlock()
                
                let optimalGFLOPS = (2.0 * Double(size * size * size) / minTime) / 1_000_000
                print("  \(size)×\(size): CPU \(String(format: "%.0f", cpuGFLOPS)) GFLOPS, GPU \(String(format: "%.0f", gpuGFLOPS)) GFLOPS, Smart Split (Tuned Ratio: \(String(format: "%.2f", bestRatio))) \(String(format: "%.0f", optimalGFLOPS)) GFLOPS")
                
                // Clean up calibration buffers immediately
                MemoryManager.shared.clearPool()
            }
        }
        
        isCalibrated = true
        print("✅ Calibration complete!")
    }
    
    /// Calibrate shapes by measuring actual performance and sweeping split ratios
    public func calibrateShapes(_ shapes: [(M: Int, N: Int, K: Int)]) throws {
        print("📐 Calibrating SmartScheduler Shapes...")
        
        // Ramping up the DVFS frequencies (warmup)
        print("🔥 Ramping up hardware frequencies (warmup)...")
        for _ in 0..<30 {
            try autoreleasepool {
                let wa = try Tensor.random([1024, 1024])
                let wb = try Tensor.random([1024, 1024])
                _ = try Tensor.matmul(wa, wb)
                _ = try GPUEngine.shared.matmulMPS(wa, wb)
                GPUEngine.shared.sync()
            }
        }
        MemoryManager.shared.clearPool()
        
        for shape in shapes {
            try autoreleasepool {
                let M = shape.M
                let N = shape.N
                let K = shape.K
                let key = "matmul_\(M)_\(N)_\(K)"
                
                let a = try Tensor.random([M, K])
                let b = try Tensor.random([K, N])
                
                // Measure CPU
                let cpuTime = try measureBackend(.cpu, size: M) {
                    try Tensor.matmul(a, b)
                }
                recordProfile(key: key, backend: .cpu, timeMs: cpuTime)
                
                // Measure GPU
                let gpuTime = try measureBackend(.gpu, size: M) {
                    let res = try GPUEngine.shared.matmulMPS(a, b)
                    GPUEngine.shared.sync()
                    return res
                }
                recordProfile(key: key, backend: .gpu, timeMs: gpuTime)
                
                // Base split ratio on theoretical throughputs
                let cpuThroughput = 1.0 / cpuTime
                let gpuThroughput = 1.0 / gpuTime
                let total = cpuThroughput + gpuThroughput
                let theoreticalRatio = cpuThroughput / total
                
                // Sweep ratios around theoretical ratio to find actual best
                var candidateRatios = [0.0]
                if theoreticalRatio > 0.02 {
                    let r = theoreticalRatio
                    candidateRatios.append(max(0.02, r - 0.12))
                    candidateRatios.append(max(0.02, r - 0.06))
                    candidateRatios.append(r)
                    candidateRatios.append(min(0.55, r + 0.06))
                    candidateRatios.append(min(0.55, r + 0.12))
                }
                
                var bestRatio = 0.0
                var minTime = gpuTime
                
                for ratio in candidateRatios {
                    let time = try measureSplit(a, b, cpuRatio: ratio)
                    if time < minTime {
                        minTime = time
                        bestRatio = ratio
                    }
                }
                
                lock.lock()
                tunedRatios[key] = bestRatio
                lock.unlock()
                
                let cpuGFLOPS = (2.0 * Double(M * N * K) / cpuTime) / 1_000_000
                let gpuGFLOPS = (2.0 * Double(M * N * K) / gpuTime) / 1_000_000
                let optimalGFLOPS = (2.0 * Double(M * N * K) / minTime) / 1_000_000
                print("  \(M)×\(N)×\(K): CPU \(String(format: "%.0f", cpuGFLOPS)) GFLOPS, GPU \(String(format: "%.0f", gpuGFLOPS)) GFLOPS, Smart Split (Tuned Ratio: \(String(format: "%.2f", bestRatio))) \(String(format: "%.0f", optimalGFLOPS)) GFLOPS")
                
                MemoryManager.shared.clearPool()
            }
        }
        isCalibrated = true
        print("✅ Shapes calibration complete!")
    }
    
    private func measureBackend<T>(_ backend: HardwareBackend, size: Int, operation: () throws -> T) throws -> Double {
        // Warmup
        for _ in 0..<3 {
            _ = try operation()
        }
        
        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10
        for _ in 0..<iterations {
            _ = try operation()
        }
        return (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
    }
    
    private func measureSplit(_ a: Tensor, _ b: Tensor, cpuRatio: Double) throws -> Double {
        // Warmup
        for _ in 0..<3 {
            try autoreleasepool {
                _ = try smartMatmulWithRatio(a, b, cpuRatio: cpuRatio)
                GPUEngine.shared.sync()
            }
        }
        
        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10
        for _ in 0..<iterations {
            try autoreleasepool {
                _ = try smartMatmulWithRatio(a, b, cpuRatio: cpuRatio)
                GPUEngine.shared.sync()
            }
        }
        return (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
    }
    
    private func recordProfile(key: String, backend: HardwareBackend, timeMs: Double) {
        lock.lock()
        defer { lock.unlock() }
        
        if profiles[key] == nil {
            profiles[key] = [:]
        }
        if profiles[key]![backend] == nil {
            profiles[key]![backend] = PerformanceProfile()
        }
        profiles[key]![backend]!.record(timeMs)
    }
    
    // MARK: - Optimal Split Ratios
    
    /// Calculate optimal work split ratio based on measured throughput
    public func optimalSplitRatio(for M: Int, N: Int, K: Int) -> (cpu: Double, gpu: Double) {
        let key = "matmul_\(M)_\(N)_\(K)"
        
        lock.lock()
        let tuned = tunedRatios[key]
        lock.unlock()
        
        if let cpuRatio = tuned {
            return (cpuRatio, 1.0 - cpuRatio)
        }
        
        // Fallback to square calibration
        lock.lock()
        let squareTuned = tunedRatios["matmul_\(M)"]
        lock.unlock()
        if let cpuRatio = squareTuned {
            return (cpuRatio, 1.0 - cpuRatio)
        }
        
        // Default
        return (0.30, 0.70)
    }
    
    // MARK: - Intelligent Matmul
    
    /// Matrix multiply with intelligent proportional splitting
    public func smartMatmul(_ a: Tensor, _ b: Tensor, transposeLeft: Bool = false, transposeRight: Bool = false) throws -> Tensor {
        guard a.ndim == 2 && b.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        
        let M = transposeLeft ? a.shape[1] : a.shape[0]
        let K = transposeLeft ? a.shape[0] : a.shape[1]
        let N = transposeRight ? b.shape[0] : b.shape[1]
        let expectedK = transposeRight ? b.shape[1] : b.shape[0]
        
        guard K == expectedK else {
            throw MemoryError.invalidShape
        }
        
        // For small matrices, use single best backend
        if M < 512 {
            return try GPUEngine.shared.matmulMPS(a, b, transposeLeft: transposeLeft, transposeRight: transposeRight)
        }
        
        // Get optimal split ratio
        let (cpuRatio, _) = optimalSplitRatio(for: M, N: N, K: K)
        return try smartMatmulWithRatio(a, b, cpuRatio: cpuRatio, transposeLeft: transposeLeft, transposeRight: transposeRight)
    }
    
    private func smartMatmulWithRatio(_ a: Tensor, _ b: Tensor, cpuRatio: Double, transposeLeft: Bool = false, transposeRight: Bool = false) throws -> Tensor {
        let M = transposeLeft ? a.shape[1] : a.shape[0]
        let K = transposeLeft ? a.shape[0] : a.shape[1]
        let N = transposeRight ? b.shape[0] : b.shape[1]
        
        var cpuRows = Int(Double(M) * cpuRatio)
        if M >= 16 {
            cpuRows = ((cpuRows + 8) / 16) * 16
            cpuRows = max(0, min(M, cpuRows))
        }
        let gpuRows = M - cpuRows
        
        // If one backend dominates, just use it
        if cpuRows == M {
            if transposeLeft || transposeRight {
                return try GPUEngine.shared.matmulMPS(a, b, transposeLeft: transposeLeft, transposeRight: transposeRight)
            }
            return try Tensor.matmul(a, b)
        }
        if cpuRows == 0 {
            return try GPUEngine.shared.matmulMPS(a, b, transposeLeft: transposeLeft, transposeRight: transposeRight)
        }
        
        // Ensure pending GPU writes to inputs are complete before CPU reads.
        // On Apple Silicon unified memory, committed GPU writes are coherent — no copies needed.
        // We use commitActiveCommandBuffer() + targeted wait instead of the heavy sync() which would stall all work.
        var prevCommandBuffer: MTLCommandBuffer? = nil
        if a.isDirty || b.isDirty {
            GPUEngine.shared.commitActiveCommandBuffer()
            prevCommandBuffer = GPUEngine.shared.lastCommittedBuffer
            a.isDirty = false
            b.isDirty = false
        }
        
        let result = try Tensor(shape: [M, N], dtype: a.dtype)
        
        // Use _buffer directly to avoid the buffer getter's auto-sync (we already synced above).
        let aPtr = a._buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b._buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
        let rPtr = result._buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        
        let aMetal = a.metalBuffer
        let bMetal = b.metalBuffer
        let rMetal = result.metalBuffer
        
        // GPU portion (enqueue asynchronously on calling thread)
        if gpuRows > 0 {
            try GPUEngine.shared.matmulRawMPS(
                a: aMetal,
                b: bMetal,
                result: rMetal,
                M: gpuRows,
                N: N,
                K: K,
                aOffset: transposeLeft ? cpuRows * 4 : cpuRows * K * 4,
                resultOffset: cpuRows * N * 4,
                transposeLeft: transposeLeft,
                transposeRight: transposeRight,
                waitUntilCompleted: false
            )
            GPUEngine.shared.commitActiveCommandBuffer()
        }
        
        // Wait for the previous command buffer (which wrote to inputs) to complete before CPU reads them.
        if let prevCb = prevCommandBuffer, prevCb.status != .completed {
            prevCb.waitUntilCompleted()
        }
        
        // CPU portion (runs directly on the calling thread in parallel with the GPU)
        if cpuRows > 0 {
            let transA = transposeLeft ? CblasTrans : CblasNoTrans
            let transB = transposeRight ? CblasTrans : CblasNoTrans
            let lda = transposeLeft ? Int32(M) : Int32(K)
            let ldb = transposeRight ? Int32(K) : Int32(N)
            
            cblas_sgemm(
                CblasRowMajor,
                transA,
                transB,
                Int32(cpuRows),
                Int32(N),
                Int32(K),
                1.0,
                aPtr, lda,
                bPtr, ldb,
                0.0,
                rPtr, Int32(N)
            )
        }
        
        result.isDirty = true
        return result
    }
    
    /// Create a view (copy) of tensor rows
    private func tensorView(_ tensor: Tensor, rowStart: Int, rowCount: Int) throws -> Tensor {
        let K = tensor.shape[1]
        let view = try Tensor(shape: [rowCount, K], dtype: tensor.dtype)
        
        let srcPtr = tensor.buffer.pointer.advanced(by: rowStart * K * tensor.dtype.size)
        memcpy(view.buffer.pointer, srcPtr, rowCount * K * tensor.dtype.size)
        
        return view
    }
    
    // MARK: - Statistics
    
    public func printStats() {
        lock.lock()
        defer { lock.unlock() }
        
        print("\n📊 SmartScheduler Performance Profiles:")
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        
        for (key, backends) in profiles.sorted(by: { $0.key < $1.key }) {
            print("\n\(key):")
            
            var throughputs: [(HardwareBackend, Double)] = []
            for (backend, profile) in backends {
                throughputs.append((backend, profile.throughput))
                print("  \(backend.rawValue): \(String(format: "%.2f", profile.avgTimeMs)) ms avg")
            }
            
            // Show optimal split
            let total = throughputs.reduce(0) { $0 + $1.1 }
            if total > 0 {
                for (backend, throughput) in throughputs {
                    let pct = (throughput / total) * 100
                    print("    → \(backend.rawValue) should get \(String(format: "%.0f", pct))% of work")
                }
            }
        }
    }
}
