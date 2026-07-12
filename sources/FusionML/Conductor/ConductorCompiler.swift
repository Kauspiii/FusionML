// ConductorCompiler.swift — Graph Compiler for the Conductor
// Analyzes computation graphs, profiles ops on all backends,
// and produces frozen ExecutionPlans with optimal scheduling.
// This runs ONCE per graph shape — all runtime cost is amortized to zero.

import Foundation
import Metal
import Accelerate
import CoreML

/// Compiles computation graphs into frozen ExecutionPlans.
/// The "brain" of the Conductor — does all the thinking upfront.
public final class ConductorCompiler {
    
    public static let shared = ConductorCompiler()
    
    /// Profiled cost model — persists across compilations and app launches.
    public let costModel = CostModel()
    
    /// Cache of compiled plans by graph hash — never recompile the same graph.
    private var planCache: [UInt64: ExecutionPlan] = [:]
    private let cacheLock = NSLock()
    
    /// Number of profiling iterations per (op, backend) pair.
    private let profilingIterations = 5
    
    private init() {}
    
    // MARK: - Graph Definition
    
    /// A node in the computation graph (used during compilation only).
    public class GraphNode {
        public let id: Int
        public let op: ConductorOp
        public let inputNodeIds: [Int]        // IDs of nodes that produce this node's inputs
        public let outputShape: [Int]
        public var params: [String: Any]?
        public var staticWeights: [Tensor]?   // For ANE pre-compilation (baked into model)
        
        /// Assigned during scheduling
        var assignedBackend: ConductorBackend?
        var assignedWave: Int?
        
        public init(id: Int, op: ConductorOp, inputs: [Int], outputShape: [Int],
                    params: [String: Any]? = nil, staticWeights: [Tensor]? = nil) {
            self.id = id
            self.op = op
            self.inputNodeIds = inputs
            self.outputShape = outputShape
            self.params = params
            self.staticWeights = staticWeights
        }
    }
    
    /// A computation graph — DAG of nodes with explicit dependencies.
    public class ComputationGraph {
        public var nodes: [GraphNode] = []
        public var inputNodeIds: [Int] = []     // Nodes that are graph inputs
        public var outputNodeIds: [Int] = []    // Nodes that are graph outputs
        
        private var nextId = 0
        
        public init() {}
        
        /// Add an input node (external tensor provided by user).
        @discardableResult
        public func addInput(shape: [Int]) -> Int {
            let id = nextId
            nextId += 1
            // Input nodes have no op — they just hold data
            let node = GraphNode(id: id, op: .add, inputs: [], outputShape: shape)
            nodes.append(node)
            inputNodeIds.append(id)
            return id
        }
        
        /// Add a computation node.
        @discardableResult
        public func addNode(op: ConductorOp, inputs: [Int], outputShape: [Int],
                            params: [String: Any]? = nil, staticWeights: [Tensor]? = nil) -> Int {
            let id = nextId
            nextId += 1
            let node = GraphNode(id: id, op: op, inputs: inputs, outputShape: outputShape,
                                 params: params, staticWeights: staticWeights)
            nodes.append(node)
            return id
        }
        
        /// Mark a node as a graph output.
        public func markOutput(_ nodeId: Int) {
            outputNodeIds.append(nodeId)
        }
        
        /// Compute a hash of the graph structure for caching.
        public func structureHash() -> UInt64 {
            var hash: UInt64 = 14695981039346656037
            for node in nodes {
                hash ^= UInt64(bitPattern: Int64(node.op.rawValue))
                hash &*= 1099511628211
                for dim in node.outputShape {
                    hash ^= UInt64(bitPattern: Int64(dim))
                    hash &*= 1099511628211
                }
                for dep in node.inputNodeIds {
                    hash ^= UInt64(bitPattern: Int64(dep))
                    hash &*= 1099511628211
                }
            }
            return hash
        }
    }
    
    // MARK: - Compile
    
    /// Compile a computation graph into a frozen ExecutionPlan.
    /// This is the expensive operation — runs profiling, scheduling, and resource pre-allocation.
    /// The resulting plan can be replayed thousands of times with zero overhead.
    public func compile(graph: ComputationGraph, dtype: DType = .float32) throws -> ExecutionPlan {
        // Check cache first
        var hasher = Hasher()
        hasher.combine(graph.structureHash())
        hasher.combine(dtype)
        let hash = UInt64(bitPattern: Int64(hasher.finalize()))
        
        cacheLock.lock()
        if let cached = planCache[hash] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()
        
        // Phase 1: Profile each non-input node on all backends
        try profileGraph(graph)
        
        // Phase 2: Assign backends using greedy scheduling
        assignBackends(graph)
        
        // Phase 3: Group into parallel waves
        let waves = buildWaves(graph)
        
        // Phase 4: Allocate buffer pool and build execution steps
        let plan = try buildPlan(graph: graph, waves: waves, dtype: dtype)
        
        // Phase 5: Pre-allocate all Metal buffers
        try plan.allocateBuffers()
        
        // Cache the plan
        cacheLock.lock()
        planCache[hash] = plan
        cacheLock.unlock()
        
        return plan
    }
    
    // MARK: - Phase 1: Profiling
    
    private func profileGraph(_ graph: ComputationGraph) throws {
        for node in graph.nodes {
            // Skip input nodes
            if graph.inputNodeIds.contains(node.id) { continue }
            
            let inputShapes = node.inputNodeIds.map { inputId in
                graph.nodes.first(where: { $0.id == inputId })!.outputShape
            }
            let key = OpShapeKey(op: node.op, shapes: inputShapes + [node.outputShape])
            
            // Skip if already profiled
            if costModel.bestBackend(for: key) != nil { continue }
            
            print("🔍 Profiling node \(node.id) [\(node.op)] with input shapes: \(inputShapes)")
            
            // Profile on each backend
            for backend in ConductorBackend.allCases {
                if backend == .cpu {
                    let totalElements = inputShapes.map { $0.reduce(1, *) }.reduce(0, +)
                    if totalElements > 1_000_000 {
                        print("  Skipping CPU profiling for node \(node.id) (total elements \(totalElements) > 1M)")
                        continue // Skip CPU profiling for large inputs to avoid compile-time hangs
                    }
                }
                print("  Profiling node \(node.id) on \(backend)...")
                if let entry = profileOp(node: node, inputShapes: inputShapes, backend: backend) {
                    print("    Completed: \(String(format: "%.2f", entry.latencyMs)) ms")
                    costModel.record(key: key, backend: backend, entry: entry)
                }
            }
        }
    }
    
    private func profileOp(node: GraphNode, inputShapes: [[Int]], backend: ConductorBackend) -> CostEntry? {
        // Only profile ops that the backend can actually execute
        switch backend {
        case .gpu:
            return profileOnGPU(op: node.op, inputShapes: inputShapes, outputShape: node.outputShape)
        case .cpu:
            return profileOnCPU(op: node.op, inputShapes: inputShapes, outputShape: node.outputShape)
        case .ane:
            return profileOnANE(op: node.op, inputShapes: inputShapes, outputShape: node.outputShape, staticWeights: node.staticWeights)
        }
    }
    
    private func profileOnGPU(op: ConductorOp, inputShapes: [[Int]], outputShape: [Int]) -> CostEntry? {
        guard op == .matmul || op == .matmulTranspose || op == .add || op == .mul ||
              op == .relu || op == .gelu || op == .layerNorm || op == .geluMul else {
            return nil
        }
        
        do {
            // Create random input tensors
            let inputs = try inputShapes.map { try Tensor.random($0) }
            
            // Warmup
            for _ in 0..<3 {
                switch op {
                case .matmul, .matmulTranspose:
                    let _ = try GPUEngine.shared.matmulMPS(inputs[0], inputs[1], transposeRight: op == .matmulTranspose)
                    GPUEngine.shared.sync()
                case .add:
                    let _ = try GPUEngine.shared.add(inputs[0], inputs[1])
                    GPUEngine.shared.sync()
                case .relu:
                    let _ = try GPUEngine.shared.relu(inputs[0])
                    GPUEngine.shared.sync()
                case .layerNorm:
                    if inputs.count >= 3 {
                        let _ = try GPUEngine.shared.layerNorm(inputs[0], gamma: inputs[1], beta: inputs[2])
                        GPUEngine.shared.sync()
                    }
                case .geluMul:
                    if inputs.count >= 2 {
                        let _ = try GPUEngine.shared.geluMul(inputs[0], inputs[1])
                        GPUEngine.shared.sync()
                    }
                default: break
                }
            }
            
            // Measure
            var times: [Double] = []
            for _ in 0..<profilingIterations {
                let start = CFAbsoluteTimeGetCurrent()
                switch op {
                case .matmul, .matmulTranspose:
                    let _ = try GPUEngine.shared.matmulMPS(inputs[0], inputs[1], transposeRight: op == .matmulTranspose)
                    GPUEngine.shared.sync()
                case .add:
                    let _ = try GPUEngine.shared.add(inputs[0], inputs[1])
                    GPUEngine.shared.sync()
                case .relu:
                    let _ = try GPUEngine.shared.relu(inputs[0])
                    GPUEngine.shared.sync()
                case .layerNorm:
                    if inputs.count >= 3 {
                        let _ = try GPUEngine.shared.layerNorm(inputs[0], gamma: inputs[1], beta: inputs[2])
                        GPUEngine.shared.sync()
                    }
                case .geluMul:
                    if inputs.count >= 2 {
                        let _ = try GPUEngine.shared.geluMul(inputs[0], inputs[1])
                        GPUEngine.shared.sync()
                    }
                default: break
                }
                times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
            }
            
            let avg = times.reduce(0, +) / Double(times.count)
            let stdDev = sqrt(times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count))
            return CostEntry(latencyMs: avg, stdDevMs: stdDev, sampleCount: times.count)
        } catch {
            return nil
        }
    }
    
    private func profileOnCPU(op: ConductorOp, inputShapes: [[Int]], outputShape: [Int]) -> CostEntry? {
        guard op == .matmul || op == .matmulTranspose || op == .add || op == .mul ||
              op == .relu || op == .silu || op == .layerNorm || op == .softmax else {
            return nil
        }
        
        do {
            let inputs = try inputShapes.map { try Tensor.random($0) }
            
            // Warmup
            for _ in 0..<3 {
                switch op {
                case .matmul, .matmulTranspose:
                    let _ = try ANEEngine.shared.matmulCPU(inputs[0], inputs[1])
                default: break
                }
            }
            
            var times: [Double] = []
            for _ in 0..<profilingIterations {
                let start = CFAbsoluteTimeGetCurrent()
                switch op {
                case .matmul, .matmulTranspose:
                    let _ = try ANEEngine.shared.matmulCPU(inputs[0], inputs[1])
                case .add:
                    let count = inputs[0].count
                    let aPtr = inputs[0]._buffer.pointer.bindMemory(to: Float.self, capacity: count)
                    let bPtr = inputs[1]._buffer.pointer.bindMemory(to: Float.self, capacity: count)
                    let result = try Tensor(shape: outputShape)
                    let rPtr = result._buffer.pointer.bindMemory(to: Float.self, capacity: count)
                    vDSP_vadd(aPtr, 1, bPtr, 1, rPtr, 1, vDSP_Length(count))
                default: break
                }
                times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
            }
            
            let avg = times.reduce(0, +) / Double(times.count)
            let stdDev = sqrt(times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count))
            return CostEntry(latencyMs: avg, stdDevMs: stdDev, sampleCount: times.count)
        } catch {
            return nil
        }
    }
    
    private func profileOnANE(op: ConductorOp, inputShapes: [[Int]], outputShape: [Int], staticWeights: [Tensor]?) -> CostEntry? {
        // ANE is only profiled for specific op types where CoreML makes sense
        guard op == .matmul || op == .conv2d || op == .layerNorm || op == .batchNorm ||
              op == .fusedConvBnRelu || op == .fusedQKVProjection else {
            return nil
        }
        
        // ANE matmul requires pre-compiled model — skip if no static weights
        if op == .matmul && staticWeights == nil {
            return nil
        }
        
        // For now, estimate ANE cost based on known benchmarks
        // TODO: actually compile and profile CoreML model
        return nil
    }
    
    // MARK: - Phase 2: Backend Assignment (Greedy Scheduling)
    
    private func assignBackends(_ graph: ComputationGraph) {
        // Track when each backend becomes "ready" (finished its last assigned op)
        var backendReadyTime: [ConductorBackend: Double] = [
            .gpu: 0, .cpu: 0, .ane: 0
        ]
        
        // Process nodes in topological order
        let sorted = topologicalSort(graph)
        
        for nodeId in sorted {
            guard let node = graph.nodes.first(where: { $0.id == nodeId }) else { continue }
            
            // Skip input nodes
            if graph.inputNodeIds.contains(node.id) {
                node.assignedBackend = .gpu  // Inputs live in unified memory
                continue
            }
            
            let inputShapes = node.inputNodeIds.map { inputId in
                graph.nodes.first(where: { $0.id == inputId })!.outputShape
            }
            let key = OpShapeKey(op: node.op, shapes: inputShapes + [node.outputShape])
            
            // Find the backend that finishes earliest:
            // finish_time = max(ready_time, dependency_finish_time) + op_latency
            var bestBackend = ConductorBackend.gpu
            var bestFinishTime = Double.infinity
            
            // Earliest time this node's dependencies are all done
            let depFinishTime = node.inputNodeIds.reduce(0.0) { maxTime, depId in
                guard let depNode = graph.nodes.first(where: { $0.id == depId }),
                      let depBackend = depNode.assignedBackend else { return maxTime }
                return max(maxTime, backendReadyTime[depBackend] ?? 0)
            }
            
            for backend in ConductorBackend.allCases {
                let opLatency = costModel.latency(for: key, on: backend)
                guard opLatency < .infinity else { continue }  // Backend can't do this op
                
                let startTime = max(backendReadyTime[backend] ?? 0, depFinishTime)
                let finishTime = startTime + opLatency
                
                if finishTime < bestFinishTime {
                    bestFinishTime = finishTime
                    bestBackend = backend
                }
            }
            
            node.assignedBackend = bestBackend
            backendReadyTime[bestBackend] = bestFinishTime
        }
    }
    
    // MARK: - Phase 3: Wave Building
    
    /// Group nodes into parallel waves. Nodes in the same wave have no
    /// inter-dependencies and execute concurrently on different backends.
    private func buildWaves(_ graph: ComputationGraph) -> [[Int]] {
        let sorted = topologicalSort(graph)
        var nodeWave: [Int: Int] = [:]
        var waves: [[Int]] = []
        
        for nodeId in sorted {
            guard let node = graph.nodes.first(where: { $0.id == nodeId }) else { continue }
            
            // Skip input nodes
            if graph.inputNodeIds.contains(node.id) {
                nodeWave[nodeId] = -1  // Inputs are pre-wave
                continue
            }
            
            // This node's wave = max(dependency waves) + 1
            // BUT: if this node uses the same backend as its wave-mates,
            // it must go to the next wave (can't run 2 ops on same backend simultaneously)
            let depWave = node.inputNodeIds.reduce(-1) { maxWave, depId in
                max(maxWave, nodeWave[depId] ?? -1)
            }
            
            var targetWave = depWave + 1
            
            // Check if target wave already has a step on the same backend
            while targetWave < waves.count {
                let waveNodes = waves[targetWave]
                let sameBackend = waveNodes.contains { waveNodeId in
                    graph.nodes.first(where: { $0.id == waveNodeId })?.assignedBackend == node.assignedBackend
                }
                if sameBackend {
                    targetWave += 1
                } else {
                    break
                }
            }
            
            // Add to wave
            while waves.count <= targetWave {
                waves.append([])
            }
            waves[targetWave].append(nodeId)
            nodeWave[nodeId] = targetWave
            node.assignedWave = targetWave
        }
        
        return waves
    }
    
    // MARK: - Phase 4: Build Final Plan
    
    private func buildPlan(graph: ComputationGraph, waves: [[Int]], dtype: DType) throws -> ExecutionPlan {
        // Create buffer slots: one per node output
        var bufferPool: [BufferSlot] = []
        var nodeToSlot: [Int: Int] = [:]
        
        for node in graph.nodes {
            let slot = bufferPool.count
            bufferPool.append(BufferSlot(shape: node.outputShape, dtype: dtype))
            nodeToSlot[node.id] = slot
        }
        
        // Build execution steps
        var executionWaves: [[ExecutionStep]] = []
        var stepId = 0
        
        for wave in waves {
            var waveSteps: [ExecutionStep] = []
            
            for nodeId in wave {
                guard let node = graph.nodes.first(where: { $0.id == nodeId }) else { continue }
                
                let inputSlots = node.inputNodeIds.compactMap { nodeToSlot[$0] }
                let outputSlot = nodeToSlot[nodeId]!
                
                let inputShapes = node.inputNodeIds.map { inputId in
                    graph.nodes.first(where: { $0.id == inputId })!.outputShape
                }
                let key = OpShapeKey(op: node.op, shapes: inputShapes + [node.outputShape])
                let estimatedLatency = costModel.latency(for: key, on: node.assignedBackend ?? .gpu)
                
                let isConsumedByHost = graph.nodes.contains { consumer in
                    consumer.inputNodeIds.contains(node.id) &&
                    (consumer.assignedBackend == .cpu || consumer.assignedBackend == .ane)
                }
                let isGraphOutput = graph.outputNodeIds.contains(node.id)
                let needsHostSync = isConsumedByHost || isGraphOutput
                
                let step = ExecutionStep(
                    id: stepId,
                    op: node.op,
                    backend: node.assignedBackend ?? .gpu,
                    inputSlots: inputSlots,
                    outputSlot: outputSlot,
                    dependencies: node.inputNodeIds.compactMap { depId in
                        // Find step ID for dependency (if it's in a previous wave)
                        graph.inputNodeIds.contains(depId) ? nil : depId
                    },
                    outputShape: node.outputShape,
                    precompiledResource: nil,  // TODO: pre-compile ANE models here
                    params: node.params,
                    estimatedLatencyMs: estimatedLatency == .infinity ? 1.0 : estimatedLatency,
                    needsHostSync: needsHostSync
                )
                
                waveSteps.append(step)
                stepId += 1
            }
            
            if !waveSteps.isEmpty {
                executionWaves.append(waveSteps)
            }
        }
        
        let inputSlots = graph.inputNodeIds.compactMap { nodeToSlot[$0] }
        let outputSlots = graph.outputNodeIds.compactMap { nodeToSlot[$0] }
        
        return ExecutionPlan(
            waves: executionWaves,
            bufferPool: bufferPool,
            inputSlots: inputSlots,
            outputSlots: outputSlots
        )
    }
    
    // MARK: - Topological Sort
    
    private func topologicalSort(_ graph: ComputationGraph) -> [Int] {
        var visited = Set<Int>()
        var result: [Int] = []
        
        func dfs(_ nodeId: Int) {
            if visited.contains(nodeId) { return }
            visited.insert(nodeId)
            
            if let node = graph.nodes.first(where: { $0.id == nodeId }) {
                for dep in node.inputNodeIds {
                    dfs(dep)
                }
            }
            result.append(nodeId)
        }
        
        for node in graph.nodes {
            dfs(node.id)
        }
        
        return result
    }
}
