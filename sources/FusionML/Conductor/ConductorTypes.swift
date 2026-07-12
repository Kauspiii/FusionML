// ConductorTypes.swift — Core types for the FusionML Conductor
// Defines backends, operation types, and the cost model used for scheduling.

import Foundation
import Metal

// MARK: - Backend

/// Compute backends available on Apple Silicon.
public enum ConductorBackend: Int, CaseIterable, Hashable, CustomStringConvertible {
    case gpu = 0   // Metal / MPS
    case cpu = 1   // Accelerate / AMX
    case ane = 2   // CoreML → Neural Engine
    
    public var description: String {
        switch self {
        case .gpu: return "GPU"
        case .cpu: return "CPU"
        case .ane: return "ANE"
        }
    }
}

// MARK: - Operation Types

/// Operations the Conductor can schedule.
public enum ConductorOp: Int, Hashable, CustomStringConvertible {
    // Linear algebra
    case matmul = 0
    case matmulTranspose = 1
    
    // Element-wise
    case add = 10
    case mul = 11
    case relu = 12
    case gelu = 13
    case silu = 14
    case sigmoid = 15
    
    // Normalization
    case layerNorm = 20
    case batchNorm = 21
    
    // Attention
    case softmax = 30
    case scaledDotProduct = 31
    
    // Convolution
    case conv2d = 40
    
    // Reduction
    case mean = 50
    case sum = 51
    
    // Composite (fused)
    case fusedConvBnRelu = 100
    case fusedQKVProjection = 101
    case fusedMLPGateUp = 102
    case geluMul = 103
    
    public var description: String {
        switch self {
        case .matmul: return "matmul"
        case .matmulTranspose: return "matmul_T"
        case .add: return "add"
        case .mul: return "mul"
        case .relu: return "relu"
        case .gelu: return "gelu"
        case .silu: return "silu"
        case .sigmoid: return "sigmoid"
        case .layerNorm: return "layernorm"
        case .batchNorm: return "batchnorm"
        case .softmax: return "softmax"
        case .scaledDotProduct: return "sdpa"
        case .conv2d: return "conv2d"
        case .mean: return "mean"
        case .sum: return "sum"
        case .fusedConvBnRelu: return "fused_conv_bn_relu"
        case .fusedQKVProjection: return "fused_qkv"
        case .fusedMLPGateUp: return "fused_mlp_gate_up"
        case .geluMul: return "gelu_mul"
        }
    }
    
    /// Whether this op is compute-bound (vs memory-bound).
    /// Compute-bound ops benefit more from multi-backend parallelism.
    public var isComputeBound: Bool {
        switch self {
        case .matmul, .matmulTranspose, .conv2d,
             .fusedConvBnRelu, .fusedQKVProjection, .fusedMLPGateUp:
            return true
        default:
            return false
        }
    }
    
    /// Estimated FLOPs for this op given input shapes.
    /// Returns 0 for element-wise ops (bandwidth-bound).
    public func estimatedFLOPs(shapes: [[Int]]) -> Int {
        switch self {
        case .matmul, .matmulTranspose:
            guard shapes.count >= 2,
                  shapes[0].count == 2,
                  shapes[1].count == 2 else { return 0 }
            let M = shapes[0][0]
            let K = shapes[0][1]
            let N = shapes[1][1]
            return 2 * M * K * N
        case .conv2d:
            // Approximate: 2 * N * C_out * C_in * kH * kW * H_out * W_out
            return 0  // TODO: implement
        default:
            return 0
        }
    }
}

// MARK: - Op Shape Key

/// Compact key for looking up costs: (op_type, shape_hash).
/// Two ops with the same type and shapes will have the same cost.
public struct OpShapeKey: Hashable {
    public let op: ConductorOp
    public let shapeHash: UInt64
    
    public init(op: ConductorOp, shapes: [[Int]]) {
        self.op = op
        // FNV-1a hash of shapes for fast lookup
        var hash: UInt64 = 14695981039346656037
        for shape in shapes {
            for dim in shape {
                hash ^= UInt64(bitPattern: Int64(dim))
                hash &*= 1099511628211
            }
            hash ^= 0xFF  // shape separator
        }
        self.shapeHash = hash
    }
}

// MARK: - Cost Entry

/// Profiled cost for one (op, shape) on one backend.
public struct CostEntry {
    public var latencyMs: Double        // Average execution time
    public var stdDevMs: Double         // Standard deviation
    public var memoryBytes: Int         // Estimated memory traffic
    public var sampleCount: Int         // Number of profiling runs
    
    public init(latencyMs: Double = .infinity, stdDevMs: Double = 0,
                memoryBytes: Int = 0, sampleCount: Int = 0) {
        self.latencyMs = latencyMs
        self.stdDevMs = stdDevMs
        self.memoryBytes = memoryBytes
        self.sampleCount = sampleCount
    }
}

// MARK: - Cost Model

/// Lookup table: (OpType, Shape) → latency per backend.
/// Populated during compile phase, read during scheduling (zero-cost lookup).
public final class CostModel {
    
    /// costs[OpShapeKey][Backend] = CostEntry
    private var costs: [OpShapeKey: [ConductorBackend: CostEntry]] = [:]
    private let lock = NSLock()
    
    /// M1 shared memory bandwidth in bytes/sec
    public static let m1BandwidthBytesPerSec: Double = 68_000_000_000  // 68 GB/s
    
    // MARK: - Query
    
    /// Get the best backend for a given op+shape, or nil if not profiled.
    public func bestBackend(for key: OpShapeKey) -> ConductorBackend? {
        lock.lock()
        defer { lock.unlock() }
        guard let backends = costs[key] else { return nil }
        return backends.min(by: { $0.value.latencyMs < $1.value.latencyMs })?.key
    }
    
    /// Get the estimated latency for an op on a specific backend.
    public func latency(for key: OpShapeKey, on backend: ConductorBackend) -> Double {
        lock.lock()
        defer { lock.unlock() }
        return costs[key]?[backend]?.latencyMs ?? .infinity
    }
    
    /// Get all backend costs for a given op+shape.
    public func allCosts(for key: OpShapeKey) -> [ConductorBackend: CostEntry] {
        lock.lock()
        defer { lock.unlock() }
        return costs[key] ?? [:]
    }
    
    // MARK: - Record
    
    /// Record a profiling result.
    public func record(key: OpShapeKey, backend: ConductorBackend, entry: CostEntry) {
        lock.lock()
        defer { lock.unlock() }
        if costs[key] == nil { costs[key] = [:] }
        costs[key]![backend] = entry
    }
    
    /// Record a single latency sample.
    public func recordSample(key: OpShapeKey, backend: ConductorBackend, latencyMs: Double) {
        lock.lock()
        defer { lock.unlock() }
        if costs[key] == nil { costs[key] = [:] }
        if var existing = costs[key]![backend] {
            // Running average
            let n = Double(existing.sampleCount)
            existing.latencyMs = (existing.latencyMs * n + latencyMs) / (n + 1)
            existing.sampleCount += 1
            costs[key]![backend] = existing
        } else {
            costs[key]![backend] = CostEntry(latencyMs: latencyMs, sampleCount: 1)
        }
    }
    
    // MARK: - Bandwidth Model
    
    /// Estimate memory bandwidth pressure for an op.
    /// Returns fraction of total bandwidth (0.0 - 1.0).
    public func bandwidthPressure(op: ConductorOp, shapes: [[Int]]) -> Double {
        var totalBytes = 0
        for shape in shapes {
            totalBytes += shape.reduce(1, *) * 4  // float32 = 4 bytes
        }
        // Assume the op reads all inputs + writes one output of same size as first input
        totalBytes += (shapes.first?.reduce(1, *) ?? 0) * 4
        
        // At 68 GB/s, how much of the bus does this consume per millisecond?
        // This is a rough model — real bandwidth depends on access patterns.
        let bandwidthPerMs = CostModel.m1BandwidthBytesPerSec / 1000.0
        return Double(totalBytes) / bandwidthPerMs
    }
    
    // MARK: - Persistence
    
    /// Save cost model to disk for instant reload across launches.
    public func save(to url: URL) throws {
        lock.lock()
        let snapshot = costs
        lock.unlock()
        
        var serializable: [String: [String: [String: Double]]] = [:]
        for (key, backends) in snapshot {
            let keyStr = "\(key.op.rawValue)_\(key.shapeHash)"
            serializable[keyStr] = [:]
            for (backend, entry) in backends {
                serializable[keyStr]![backend.description] = [
                    "latencyMs": entry.latencyMs,
                    "stdDevMs": entry.stdDevMs,
                    "samples": Double(entry.sampleCount)
                ]
            }
        }
        let data = try JSONSerialization.data(withJSONObject: serializable, options: .prettyPrinted)
        try data.write(to: url)
    }
    
    /// Load cost model from disk.
    public func load(from url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: [String: [String: Double]]] else {
            return
        }
        
        lock.lock()
        defer { lock.unlock() }
        
        for (keyStr, backends) in dict {
            let parts = keyStr.split(separator: "_")
            guard parts.count == 2,
                  let opRaw = Int(parts[0]),
                  let hash = UInt64(parts[1]),
                  let op = ConductorOp(rawValue: opRaw) else { continue }
            
            let key = OpShapeKey(op: op, shapes: [])  // We'll use stored hash
            var patchedKey = key
            // Direct hash assignment via unsafe — the shapes are lost but hash is preserved
            withUnsafeMutablePointer(to: &patchedKey) { ptr in
                let raw = UnsafeMutableRawPointer(ptr)
                raw.storeBytes(of: hash, toByteOffset: MemoryLayout<ConductorOp>.stride, as: UInt64.self)
            }
            
            for (backendStr, values) in backends {
                let backend: ConductorBackend
                switch backendStr {
                case "GPU": backend = .gpu
                case "CPU": backend = .cpu
                case "ANE": backend = .ane
                default: continue
                }
                let entry = CostEntry(
                    latencyMs: values["latencyMs"] ?? .infinity,
                    stdDevMs: values["stdDevMs"] ?? 0,
                    sampleCount: Int(values["samples"] ?? 0)
                )
                if costs[patchedKey] == nil { costs[patchedKey] = [:] }
                costs[patchedKey]![backend] = entry
            }
        }
    }
}
