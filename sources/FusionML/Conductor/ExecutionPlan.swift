// ExecutionPlan.swift — Frozen execution schedule for the Conductor
// Compiled once, replayed forever with zero runtime decisions.

import Foundation
import Metal
import CoreML

// MARK: - Execution Step

/// A single step in the frozen execution plan.
/// No decisions at runtime — just dispatch + sync.
public struct ExecutionStep {
    /// Unique ID for this step (index in the plan).
    public let id: Int
    
    /// The operation to perform.
    public let op: ConductorOp
    
    /// Which backend runs this step.
    public let backend: ConductorBackend
    
    /// Indices into the buffer pool for input tensors.
    public let inputSlots: [Int]
    
    /// Index into the buffer pool for the output tensor.
    public let outputSlot: Int
    
    /// IDs of steps that must complete before this step can start.
    /// Empty = no dependencies, can start immediately.
    public let dependencies: [Int]
    
    /// Shape of the output tensor.
    public let outputShape: [Int]
    
    /// Pre-compiled resource for this step (set during plan compilation).
    /// - GPU: cached MPS kernel key
    /// - ANE: pre-compiled MLModel
    /// - CPU: nil (BLAS needs no pre-compilation)
    public var precompiledResource: Any?
    
    /// Optional: extra parameters for the op (e.g., epsilon for layernorm).
    public var params: [String: Any]?
    
    /// Estimated latency from cost model (for debugging/logging).
    public let estimatedLatencyMs: Double
    
    /// Does a subsequent CPU/ANE step or graph output read this step's output?
    public let needsHostSync: Bool
}

// MARK: - Buffer Slot

/// Pre-allocated buffer slot in the execution plan's buffer pool.
public struct BufferSlot {
    public let shape: [Int]
    public var dtype: DType
    public var byteSize: Int
    public var buffer: MTLBuffer?   // Allocated during plan finalization
    public var tensor: Tensor?      // Lazily created wrapper
    
    public init(shape: [Int], dtype: DType = .float32) {
        self.shape = shape
        self.dtype = dtype
        self.byteSize = shape.reduce(1, *) * (dtype == .float16 ? 2 : 4)
    }
}

// MARK: - Execution Plan

/// Frozen, replayable execution plan.
/// Contains all steps, buffer allocations, and pre-compiled resources.
/// Execute with zero allocations and zero decisions at runtime.
public final class ExecutionPlan {
    
    /// All steps in topological order, grouped by parallel wave.
    /// Steps within the same wave have no inter-dependencies and can run concurrently.
    public let waves: [[ExecutionStep]]
    
    /// Flat list of all steps (for debugging).
    public var allSteps: [ExecutionStep] {
        waves.flatMap { $0 }
    }
    
    /// Pre-allocated buffer pool. Indices match step inputSlots/outputSlot.
    public var bufferPool: [BufferSlot]
    
    /// Indices of buffer slots that are plan inputs (provided by user).
    public let inputSlots: [Int]
    
    /// Indices of buffer slots that are plan outputs (returned to user).
    public let outputSlots: [Int]
    
    /// Total estimated latency from cost model.
    public let estimatedTotalLatencyMs: Double
    
    /// Number of parallel waves (fewer = more parallelism exploited).
    public var waveCount: Int { waves.count }
    
    /// Total step count.
    public var stepCount: Int { allSteps.count }
    
    // MARK: - Init
    
    public init(
        waves: [[ExecutionStep]],
        bufferPool: [BufferSlot],
        inputSlots: [Int],
        outputSlots: [Int]
    ) {
        self.waves = waves
        self.bufferPool = bufferPool
        self.inputSlots = inputSlots
        self.outputSlots = outputSlots
        self.estimatedTotalLatencyMs = waves.reduce(0) { total, wave in
            total + (wave.map(\.estimatedLatencyMs).max() ?? 0)
        }
    }
    
    // MARK: - Buffer Allocation
    
    /// Allocate all Metal buffers in the pool.
    /// Called once after plan compilation, before first execution.
    public func allocateBuffers() throws {
        let device = GPUEngine.shared.device
        for i in 0..<bufferPool.count {
            if bufferPool[i].buffer == nil {
                guard let buf = device.makeBuffer(
                    length: max(bufferPool[i].byteSize, 16),  // Metal requires ≥16 bytes
                    options: .storageModeShared  // Unified memory: CPU + GPU + ANE all see it
                ) else {
                    throw ConductorError.bufferAllocationFailed(slot: i, size: bufferPool[i].byteSize)
                }
                bufferPool[i].buffer = buf
            }
        }
    }
    
    /// Create Tensor wrappers for buffer slots (lazy, zero-copy).
    public func tensorForSlot(_ slot: Int) throws -> Tensor {
        guard slot >= 0 && slot < bufferPool.count else {
            throw ConductorError.invalidSlot(slot)
        }
        if let existing = bufferPool[slot].tensor {
            return existing
        }
        guard let metalBuf = bufferPool[slot].buffer else {
            throw ConductorError.bufferNotAllocated(slot: slot)
        }
        let tensor = try Tensor(
            existingBuffer: metalBuf,
            shape: bufferPool[slot].shape,
            dtype: bufferPool[slot].dtype
        )
        bufferPool[slot].tensor = tensor
        return tensor
    }
    
    // MARK: - Debug
    
    /// Print human-readable execution plan.
    public func printPlan() {
        print("╔══════════════════════════════════════════════════════════╗")
        print("║            FusionML Conductor Execution Plan            ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║ Waves: \(waveCount)  |  Steps: \(stepCount)  |  Buffers: \(bufferPool.count)")
        print("║ Estimated total latency: \(String(format: "%.3f", estimatedTotalLatencyMs)) ms")
        print("╠══════════════════════════════════════════════════════════╣")
        
        for (waveIdx, wave) in waves.enumerated() {
            let backends = wave.map { $0.backend.description }.joined(separator: " + ")
            let maxLatency = wave.map(\.estimatedLatencyMs).max() ?? 0
            print("║ Wave \(waveIdx) [\(backends)] ~\(String(format: "%.3f", maxLatency))ms")
            for step in wave {
                let deps = step.dependencies.isEmpty ? "none" : step.dependencies.map(String.init).joined(separator: ",")
                print("║   Step \(step.id): \(step.op) → \(step.backend)  in:\(step.inputSlots) out:\(step.outputSlot)  deps:[\(deps)]")
            }
        }
        
        print("╚══════════════════════════════════════════════════════════╝")
    }
}

// MARK: - Errors

public enum ConductorError: Error, LocalizedError {
    case bufferAllocationFailed(slot: Int, size: Int)
    case bufferNotAllocated(slot: Int)
    case invalidSlot(Int)
    case profilingFailed(op: ConductorOp, backend: ConductorBackend)
    case graphCycleDetected
    case noBackendAvailable(op: ConductorOp)
    case compilationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .bufferAllocationFailed(let slot, let size):
            return "Failed to allocate Metal buffer for slot \(slot) (\(size) bytes)"
        case .bufferNotAllocated(let slot):
            return "Buffer slot \(slot) not yet allocated"
        case .invalidSlot(let slot):
            return "Invalid buffer slot index: \(slot)"
        case .profilingFailed(let op, let backend):
            return "Profiling failed for \(op) on \(backend)"
        case .graphCycleDetected:
            return "Cycle detected in computation graph"
        case .noBackendAvailable(let op):
            return "No backend available for operation: \(op)"
        case .compilationFailed(let msg):
            return "Conductor compilation failed: \(msg)"
        }
    }
}
