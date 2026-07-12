// IntelligentRouter - Smart Operation Routing
// Routes operations to CPU, GPU, or ANE based on operation type and measured performance

import Foundation
import CoreML
import Accelerate

/// Operation that can be routed to different hardware
public enum SiliconOperation {
    case matmul(Tensor, Tensor)
    case add(Tensor, Tensor)
    case mul(Tensor, Tensor)
    case relu(Tensor)
    case gelu(Tensor)
    case softmax(Tensor, axis: Int)
    case layerNorm(Tensor, gamma: Tensor, beta: Tensor)
    case conv2d(Tensor, kernel: Tensor, stride: Int, padding: Int)
}

/// Hardware routing decision
public struct RoutingDecision {
    public let backend: HardwareBackend
    public let reason: String
    public let expectedSpeedup: Double
}

/// Intelligent router that learns and adapts
public final class IntelligentRouter: @unchecked Sendable {
    
    public static let shared = IntelligentRouter()
    
    /// Global flag to enable/disable workload splitting (e.g. for profiling baselines)
    public var enableSplitting: Bool = true
    
    /// Globally force all operations to a specific backend (e.g., for profiling baselines)
    public var forcedBackend: HardwareBackend? = nil
    
    /// Global flag to enable 3-Way Splitting (GPU + ANE + CPU) instead of 2-Way Splitting (GPU + CPU)
    public var useThreeWaySplit: Bool = false
    
    // Operation profiles: measured performance for each (operation, size, backend)
    private var profiles: [String: [HardwareBackend: Double]] = [:]  // time in ms
    private let lock = NSLock()
    
    // Hardware characteristics (tuned for Apple M1)
    private let characteristics: [HardwareBackend: HardwareCharacteristics] = [
        .cpu: HardwareCharacteristics(
            name: "CPU (AMX)",
            peakTFLOPS: 1.5,
            bestFor: [.matmul, .softmax, .small],
            overhead: 0.01  // Almost no overhead
        ),
        .gpu: HardwareCharacteristics(
            name: "GPU (MPS)",
            peakTFLOPS: 2.6,
            bestFor: [.matmul, .elementwise, .large, .attention],
            overhead: 0.5  // Command buffer overhead
        ),
        .ane: HardwareCharacteristics(
            name: "ANE",
            peakTFLOPS: 11.0,
            bestFor: [.conv, .batchNorm, .layerNorm, .inference],
            overhead: 2.0  // Core ML overhead
        )
    ]
    
    private init() {}
    
    // MARK: - Forward Pass Batching
    
    /// Begin a batched forward pass. All GPU operations will be accumulated into
    /// a single command buffer instead of committing per-operation.
    /// This eliminates ~100μs of Metal driver overhead per op.
    public func startForwardPass() {
        GPUEngine.shared.startBatch()
    }
    
    /// End the batched forward pass and commit all accumulated GPU work.
    public func endForwardPass() {
        GPUEngine.shared.commitBatch()
    }
    
    /// Execute a block of operations within a batched forward pass.
    /// Automatically handles startBatch/commitBatch lifecycle.
    public func batched<T>(_ block: () throws -> T) rethrows -> T {
        startForwardPass()
        defer { endForwardPass() }
        return try block()
    }
    
    // MARK: - Intelligent Routing
    
    /// Route an operation to the best backend
    public func route(_ op: SiliconOperation) -> RoutingDecision {
        if let forced = forcedBackend {
            return RoutingDecision(
                backend: forced,
                reason: "Forced backend configuration",
                expectedSpeedup: 1.0
            )
        }
        
        switch op {
        case .matmul(let a, _):
            return routeMatmul(size: a.count)
        case .add(let a, _), .mul(let a, _), .relu(let a), .gelu(let a):
            return routeElementwise(size: a.count)
        case .softmax(let a, _):
            return routeSoftmax(size: a.count)
        case .layerNorm(let a, _, _):
            return routeLayerNorm(size: a.count)
        case .conv2d(_, _, _, _):
            return routeConv2d()
        }
    }
    
    private func routeMatmul(size: Int) -> RoutingDecision {
        // Check learned profiles first
        let key = "matmul_\(size)"
        if let best = bestBackendFromProfile(key: key) {
            return RoutingDecision(
                backend: best.0,
                reason: "Learned: \(best.0.rawValue) is \(String(format: "%.0f", best.1))% faster",
                expectedSpeedup: best.1 / 100
            )
        }
        
        // Heuristic: CPU for very small, GPU for others
        if size < 100000 {  // ~300x300
            return RoutingDecision(
                backend: .cpu,
                reason: "Very small matrix: CPU (AMX) has lower overhead",
                expectedSpeedup: 1.0
            )
        } else {
            return RoutingDecision(
                backend: .gpu,
                reason: "GPU optimal / split eligible",
                expectedSpeedup: 1.0
            )
        }
    }
    
    private func routeElementwise(size: Int) -> RoutingDecision {
        if size < 10000 {
            return RoutingDecision(backend: .cpu, reason: "Small tensor: CPU faster", expectedSpeedup: 1.0)
        }
        return RoutingDecision(backend: .gpu, reason: "Large tensor: GPU parallel", expectedSpeedup: 2.0)
    }
    
    private func routeSoftmax(size: Int) -> RoutingDecision {
        // Softmax has data dependencies - CPU is often better
        return RoutingDecision(backend: .cpu, reason: "Softmax: CPU avoids sync overhead", expectedSpeedup: 1.0)
    }
    
    private func routeLayerNorm(size: Int) -> RoutingDecision {
        // Layer norm is good on ANE for inference
        return RoutingDecision(backend: .ane, reason: "LayerNorm: ANE optimized", expectedSpeedup: 1.5)
    }
    
    private func routeConv2d() -> RoutingDecision {
        // Conv2D is ANE's specialty
        return RoutingDecision(backend: .ane, reason: "Conv2D: ANE is 5-10x faster", expectedSpeedup: 5.0)
    }
    
    private func bestBackendFromProfile(key: String) -> (HardwareBackend, Double)? {
        lock.lock()
        defer { lock.unlock() }
        
        guard let backends = profiles[key], backends.count >= 2 else { return nil }
        
        let sorted = backends.sorted { $0.value < $1.value }
        guard sorted.count >= 2 else { return nil }
        
        let best = sorted[0]
        let second = sorted[1]
        let improvement = ((second.value - best.value) / best.value) * 100
        
        return (best.key, improvement)
    }
    
    // MARK: - Execute with Routing
    
    /// Execute matmul with intelligent routing
    /// Execute matmul with intelligent routing
    public func matmul(_ a: Tensor, _ b: Tensor, transposeLeft: Bool = false, transposeRight: Bool = false, isWeightGrad: Bool = false, forceGPU: Bool = false, forceCPU: Bool = false) throws -> Tensor {
        if isWeightGrad {
            return try GPUEngine.shared.matmulMPS(a, b, transposeLeft: transposeLeft, transposeRight: transposeRight)
        }
        
        if a.count >= 4000000 {
            setenv("VECLIB_MAXIMUM_THREADS", "4", 1)
        } else if a.count >= 1500000 {
            setenv("VECLIB_MAXIMUM_THREADS", "2", 1)
        } else {
            setenv("VECLIB_MAXIMUM_THREADS", "1", 1)
        }

        if forceCPU {
            return try executeCPUMatmul(a, b)
        }
        
        // Only bypass if forceGPU is explicitly requested
        if forceGPU {
            return try GPUEngine.shared.matmulMPS(a, b, transposeLeft: transposeLeft, transposeRight: transposeRight)
        }
        
        let decision = route(.matmul(a, b))
        
        switch decision.backend {
        case .cpu:
            if transposeLeft || transposeRight {
                return try GPUEngine.shared.matmulMPS(a, b, transposeLeft: transposeLeft, transposeRight: transposeRight)
            }
            return try executeCPUMatmul(a, b)
        case .gpu:
            // Use smart split for large matrices (>= 1.5M elements) to avoid overhead on small ones.
            // The SmartScheduler handles isDirty sync internally with a lightweight commit+wait
            // instead of bypassing splitting entirely.
            // Allow transposeRight (used in backward activation gradients) since SmartScheduler supports it.
            // Block transposeLeft since that changes row-splitting semantics.
            if enableSplitting && a.count >= 1500000 && !transposeLeft {
                return try SmartScheduler.shared.smartMatmul(a, b, transposeRight: transposeRight)
            }
            return try GPUEngine.shared.matmulMPS(a, b, transposeLeft: transposeLeft, transposeRight: transposeRight)
        case .ane:
            if transposeLeft || transposeRight {
                return try GPUEngine.shared.matmulMPS(a, b, transposeLeft: transposeLeft, transposeRight: transposeRight)
            }
            return try executeCPUMatmul(a, b)
        }
    }
    
    private func executeCPUMatmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        return try Tensor.matmul(a, b)
    }
    
    /// Execute element-wise add with routing
    public func add(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        if a.isDirty || b.isDirty || route(.add(a, b)).backend == .gpu {
            return try GPUEngine.shared.add(a, b)
        }
        return try Tensor.add(a, b)
    }
    
    /// Execute element-wise mul with routing
    public func mul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        if a.isDirty || b.isDirty || route(.mul(a, b)).backend == .gpu {
            return try GPUEngine.shared.mul(a, b)
        }
        return try Tensor.mul(a, b)
    }
    
    /// Execute ReLU with routing
    public func relu(_ x: Tensor) throws -> Tensor {
        if x.isDirty || route(.relu(x)).backend == .gpu {
            return try GPUEngine.shared.relu(x)
        }
        
        // CPU ReLU using Accelerate
        let result = try Tensor(shape: x.shape, dtype: x.dtype)
        let count = Int32(x.count)
        let xPtr = x.buffer.pointer.bindMemory(to: Float.self, capacity: x.count)
        let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        var zero: Float = 0
        vDSP_vthres(xPtr, 1, &zero, rPtr, 1, vDSP_Length(count))
        return result
    }
    
    /// Execute GELU with routing
    public func gelu(_ x: Tensor) throws -> Tensor {
        if x.isDirty || route(.gelu(x)).backend == .gpu {
            return try GPUEngine.shared.gelu(x)
        }
        
        // CPU GELU
        let result = try Tensor(shape: x.shape, dtype: x.dtype)
        let xData = x.toArray()
        let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        
        for i in 0..<x.count {
            let val = xData[i]
            let cdf = 0.5 * (1 + tanh(0.7978845608 * (val + 0.044715 * val * val * val)))
            rPtr[i] = val * Float(cdf)
        }
        return result
    }
    
    // MARK: - Learning
    
    /// Record measured performance
    public func recordPerformance(operation: String, backend: HardwareBackend, timeMs: Double) {
        lock.lock()
        defer { lock.unlock() }
        
        if profiles[operation] == nil {
            profiles[operation] = [:]
        }
        profiles[operation]![backend] = timeMs
    }
    
    /// Calibrate for a specific operation
    public func calibrate<T>(operation: String, iterations: Int = 10, _ block: (HardwareBackend) throws -> T) throws {
        for backend in HardwareBackend.allCases {
            // Warmup
            for _ in 0..<3 {
                _ = try? block(backend)
            }
            
            // Measure
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                _ = try block(backend)
            }
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
            recordPerformance(operation: operation, backend: backend, timeMs: time)
        }
    }
    
    // MARK: - Statistics
    
    public func printRoutingTable() {
        print("\n🧠 Intelligent Router - Learned Profiles:")
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        
        lock.lock()
        defer { lock.unlock() }
        
        for (op, backends) in profiles.sorted(by: { $0.key < $1.key }) {
            print("\n\(op):")
            let sorted = backends.sorted { $0.value < $1.value }
            for (i, (backend, time)) in sorted.enumerated() {
                let marker = i == 0 ? "🏆" : "  "
                print("  \(marker) \(backend.rawValue): \(String(format: "%.2f", time)) ms")
            }
        }
    }
}

/// Hardware characteristics
struct HardwareCharacteristics {
    let name: String
    let peakTFLOPS: Double
    let bestFor: Set<OperationCategory>
    let overhead: Double  // ms
}

/// Operation categories for routing
enum OperationCategory {
    case matmul
    case elementwise
    case softmax
    case conv
    case batchNorm
    case layerNorm
    case attention
    case small      // < 100K elements
    case large      // > 1M elements
    case inference  // Inference vs training
}
