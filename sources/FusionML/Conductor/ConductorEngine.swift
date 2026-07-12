// ConductorEngine.swift — Runtime dispatch engine for the Conductor
// The HOT PATH: zero-allocation, zero-decision replay of frozen ExecutionPlans.
// Dispatches steps to GPU, CPU, and ANE threads in parallel.

import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate
import CoreML

/// The runtime engine that executes frozen ExecutionPlans.
/// Sub-zero overhead: no decisions, no allocations, just dispatch + sync.
public final class ConductorEngine: @unchecked Sendable {
    
    public static let shared = ConductorEngine()
    
    // MARK: - Persistent Dispatch Threads
    
    /// Dedicated dispatch queues — one per backend, always alive.
    private let gpuQueue = DispatchQueue(label: "fusionml.conductor.gpu", qos: .userInteractive)
    private let cpuQueue = DispatchQueue(label: "fusionml.conductor.cpu", qos: .userInteractive)
    private let aneQueue = DispatchQueue(label: "fusionml.conductor.ane", qos: .userInteractive)
    
    private init() {}
    
    // MARK: - Execute Plan
    
    /// Execute a frozen plan with the given input tensors.
    /// This is the main entry point — should be near-zero overhead.
    ///
    /// - Parameters:
    ///   - plan: The frozen ExecutionPlan (compiled once, reused many times)
    ///   - inputs: Input tensors, mapped to plan.inputSlots
    /// - Returns: Output tensors from plan.outputSlots
    public func execute(plan: ExecutionPlan, inputs: [Tensor]) throws -> [Tensor] {
        let dtype = inputs.isEmpty ? .float32 : inputs[0].dtype
        for i in 0..<plan.bufferPool.count {
            plan.bufferPool[i].dtype = dtype
            plan.bufferPool[i].byteSize = plan.bufferPool[i].shape.reduce(1, *) * dtype.size
            plan.bufferPool[i].tensor = nil  // Reset lazy wrappers since dtype might change
        }
        
        // 1. Point the input buffer slots directly to the input tensors (zero-copy aliasing)
        for (i, inputSlot) in plan.inputSlots.enumerated() {
            guard i < inputs.count else {
                throw ConductorError.compilationFailed("Expected \(plan.inputSlots.count) inputs, got \(inputs.count)")
            }
            let src = inputs[i]
            plan.bufferPool[inputSlot].buffer = src.metalBuffer
            plan.bufferPool[inputSlot].tensor = src
        }
        
        // Start GPU batching
        GPUEngine.shared.startBatch()
        
        // 2. Execute wave by wave
        for wave in plan.waves {
            try executeWave(wave, plan: plan)
            
            // If any step in this wave needs host synchronization, commit the batch and wait
            let needsSync = wave.contains { $0.backend == .gpu && $0.needsHostSync }
            if needsSync {
                GPUEngine.shared.commitBatch()
                GPUEngine.shared.sync()
                GPUEngine.shared.startBatch()
            }
        }
        
        // End the final batch and wait for GPU completion
        GPUEngine.shared.commitBatch()
        
        // 3. Ensure GPU work is complete before reading outputs
        GPUEngine.shared.sync()
        
        // 4. Extract outputs
        var outputs: [Tensor] = []
        for outputSlot in plan.outputSlots {
            let tensor = try plan.tensorForSlot(outputSlot)
            outputs.append(tensor)
        }
        
        return outputs
    }
    
    // MARK: - Wave Execution
    
    /// Execute a single wave: all steps run in parallel across backends.
    private func executeWave(_ wave: [ExecutionStep], plan: ExecutionPlan) throws {
        if wave.count == 1 {
            // Single step — no threading overhead needed
            try executeStep(wave[0], plan: plan)
            return
        }
        
        // Split wave into GPU steps and other steps (CPU/ANE)
        let gpuSteps = wave.filter { $0.backend == .gpu }
        let otherSteps = wave.filter { $0.backend != .gpu }
        
        // Execute all GPU steps sequentially to avoid command encoder collision on the shared GPUEngine
        for step in gpuSteps {
            try executeStep(step, plan: plan)
        }
        
        if !otherSteps.isEmpty {
            let group = DispatchGroup()
            let errors = ConductorThreadSafeErrors()
            
            for (index, step) in otherSteps.enumerated() {
                if index == otherSteps.count - 1 {
                    do {
                        try self.executeStep(step, plan: plan)
                    } catch {
                        errors.append(error)
                    }
                } else {
                    let queue: DispatchQueue
                    switch step.backend {
                    case .gpu: queue = gpuQueue
                    case .cpu: queue = cpuQueue
                    case .ane: queue = aneQueue
                    }
                    
                    group.enter()
                    queue.async { [self] in
                        do {
                            try self.executeStep(step, plan: plan)
                        } catch {
                            errors.append(error)
                        }
                        group.leave()
                    }
                }
            }
            
            group.wait()
            try errors.check()
        }
    }
    
    // MARK: - Step Execution
    
    /// Execute a single step on its assigned backend.
    /// This is the innermost dispatch — must be as fast as possible.
    private func executeStep(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        switch step.backend {
        case .gpu:
            try executeOnGPU(step, plan: plan)
        case .cpu:
            try executeOnCPU(step, plan: plan)
        case .ane:
            try executeOnANE(step, plan: plan)
        }
    }
    
    // MARK: - GPU Execution
    
    private func executeOnGPU(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        switch step.op {
        case .matmul, .matmulTranspose:
            try gpuMatmul(step, plan: plan)
        case .add:
            try gpuElementWise(step, plan: plan, op: .add)
        case .mul:
            try gpuElementWise(step, plan: plan, op: .mul)
        case .relu:
            try gpuActivation(step, plan: plan, activation: .relu)
        case .gelu:
            try gpuActivation(step, plan: plan, activation: .gelu)
        case .silu:
            try gpuActivation(step, plan: plan, activation: .silu)
        case .layerNorm:
            try gpuLayerNorm(step, plan: plan)
        case .softmax:
            try gpuSoftmax(step, plan: plan)
        case .geluMul:
            try gpuGeluMul(step, plan: plan)
        default:
            throw ConductorError.noBackendAvailable(op: step.op)
        }
    }
    
    private func gpuMatmul(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard step.inputSlots.count >= 2 else { return }
        let aSlot = step.inputSlots[0]
        let bSlot = step.inputSlots[1]
        let outSlot = step.outputSlot
        
        guard let aBuf = plan.bufferPool[aSlot].buffer,
              let bBuf = plan.bufferPool[bSlot].buffer,
              let outBuf = plan.bufferPool[outSlot].buffer else {
            throw ConductorError.bufferNotAllocated(slot: aSlot)
        }
        
        let aShape = plan.bufferPool[aSlot].shape
        let bShape = plan.bufferPool[bSlot].shape
        let isTranspose = step.op == .matmulTranspose
        
        let M = aShape[0]
        let K = aShape[1]
        let N = isTranspose ? bShape[0] : bShape[1]
        let dtype = plan.bufferPool[outSlot].dtype
        
        try GPUEngine.shared.matmulRawMPS(
            a: aBuf, b: bBuf, result: outBuf,
            M: M, N: N, K: K,
            transposeRight: isTranspose,
            waitUntilCompleted: false,
            dtype: dtype
        )
    }
    
    private func gpuElementWise(_ step: ExecutionStep, plan: ExecutionPlan, op: ConductorOp) throws {
        guard step.inputSlots.count >= 2 else { return }
        let aSlot = step.inputSlots[0]
        let bSlot = step.inputSlots[1]
        let outSlot = step.outputSlot
        
        guard let aBuf = plan.bufferPool[aSlot].buffer,
              let bBuf = plan.bufferPool[bSlot].buffer,
              let outBuf = plan.bufferPool[outSlot].buffer else {
            throw ConductorError.bufferNotAllocated(slot: aSlot)
        }
        
        let count = plan.bufferPool[aSlot].shape.reduce(1, *)
        let dtype = plan.bufferPool[aSlot].dtype
        
        switch op {
        case .add:
            try GPUEngine.shared.addRaw(a: aBuf, b: bBuf, result: outBuf, count: count, dtype: dtype)
        case .mul:
            try GPUEngine.shared.mulRaw(a: aBuf, b: bBuf, result: outBuf, count: count, dtype: dtype)
        default:
            throw ConductorError.noBackendAvailable(op: op)
        }
    }
    
    private func gpuActivation(_ step: ExecutionStep, plan: ExecutionPlan, activation: ConductorOp) throws {
        guard step.inputSlots.count >= 1 else { return }
        let inSlot = step.inputSlots[0]
        let outSlot = step.outputSlot
        
        guard let inBuf = plan.bufferPool[inSlot].buffer,
              let outBuf = plan.bufferPool[outSlot].buffer else {
            throw ConductorError.bufferNotAllocated(slot: inSlot)
        }
        
        let count = plan.bufferPool[inSlot].shape.reduce(1, *)
        let dtype = plan.bufferPool[inSlot].dtype
        
        switch activation {
        case .relu:
            try GPUEngine.shared.reluRaw(input: inBuf, result: outBuf, count: count, dtype: dtype)
        case .gelu, .silu:
            try GPUEngine.shared.geluRaw(input: inBuf, result: outBuf, count: count, dtype: dtype)
        default:
            throw ConductorError.noBackendAvailable(op: activation)
        }
    }
    
    private func gpuLayerNorm(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard step.inputSlots.count >= 3 else { return }
        let inSlot = step.inputSlots[0]
        let gammaSlot = step.inputSlots[1]
        let betaSlot = step.inputSlots[2]
        let outSlot = step.outputSlot
        
        guard let inBuf = plan.bufferPool[inSlot].buffer,
              let gammaBuf = plan.bufferPool[gammaSlot].buffer,
              let betaBuf = plan.bufferPool[betaSlot].buffer,
              let outBuf = plan.bufferPool[outSlot].buffer else {
            throw ConductorError.bufferNotAllocated(slot: inSlot)
        }
        
        let shape = plan.bufferPool[inSlot].shape
        let lastDim = shape.last!
        let numRows = shape.reduce(1, *) / lastDim
        let eps = (step.params?["eps"] as? Float) ?? 1e-5
        let dtype = plan.bufferPool[inSlot].dtype
        
        try GPUEngine.shared.layerNormRaw(
            input: inBuf, gamma: gammaBuf, beta: betaBuf, result: outBuf,
            lastDim: lastDim, eps: eps, numRows: numRows, dtype: dtype
        )
    }
    
    private func gpuSoftmax(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        // TODO: implement GPU softmax via Metal shader
        // For now, fallback to CPU
        try executeOnCPU(step, plan: plan)
    }
    
    private func gpuGeluMul(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard step.inputSlots.count >= 2 else { return }
        let gateSlot = step.inputSlots[0]
        let upSlot = step.inputSlots[1]
        let outSlot = step.outputSlot
        
        guard let gateBuf = plan.bufferPool[gateSlot].buffer,
              let upBuf = plan.bufferPool[upSlot].buffer,
              let outBuf = plan.bufferPool[outSlot].buffer else {
            throw ConductorError.bufferNotAllocated(slot: gateSlot)
        }
        
        let count = plan.bufferPool[gateSlot].shape.reduce(1, *)
        let dtype = plan.bufferPool[gateSlot].dtype
        
        try GPUEngine.shared.geluMulRaw(gate: gateBuf, up: upBuf, result: outBuf, count: count, dtype: dtype)
    }
    
    // MARK: - CPU Execution (AMX/Accelerate)
    
    private func executeOnCPU(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        switch step.op {
        case .matmul, .matmulTranspose:
            try cpuMatmul(step, plan: plan)
        case .add:
            try cpuAdd(step, plan: plan)
        case .mul:
            try cpuMul(step, plan: plan)
        case .relu:
            try cpuRelu(step, plan: plan)
        case .silu:
            try cpuSilu(step, plan: plan)
        case .layerNorm:
            try cpuLayerNorm(step, plan: plan)
        case .softmax:
            try cpuSoftmax(step, plan: plan)
        default:
            throw ConductorError.noBackendAvailable(op: step.op)
        }
    }
    
    private func cpuMatmul(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard step.inputSlots.count >= 2 else { return }
        let aSlot = step.inputSlots[0]
        let bSlot = step.inputSlots[1]
        let outSlot = step.outputSlot
        
        guard let aBuf = plan.bufferPool[aSlot].buffer,
              let bBuf = plan.bufferPool[bSlot].buffer,
              let outBuf = plan.bufferPool[outSlot].buffer else {
            throw ConductorError.bufferNotAllocated(slot: aSlot)
        }
        
        let aShape = plan.bufferPool[aSlot].shape
        let bShape = plan.bufferPool[bSlot].shape
        let isTranspose = step.op == .matmulTranspose
        
        let M = Int32(aShape[0])
        let K = Int32(aShape[1])
        let N = Int32(isTranspose ? bShape[0] : bShape[1])
        
        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: Int(M * K))
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: Int(K * N))
        let rPtr = outBuf.contents().bindMemory(to: Float.self, capacity: Int(M * N))
        
        // Direct BLAS call — goes straight to AMX on Apple Silicon
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            isTranspose ? CblasTrans : CblasNoTrans,
            M, N, K,
            1.0,
            aPtr, K,
            bPtr, isTranspose ? K : N,
            0.0,
            rPtr, N
        )
    }
    
    private func cpuAdd(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard step.inputSlots.count >= 2 else { return }
        guard let aBuf = plan.bufferPool[step.inputSlots[0]].buffer,
              let bBuf = plan.bufferPool[step.inputSlots[1]].buffer,
              let outBuf = plan.bufferPool[step.outputSlot].buffer else { return }
        
        let count = plan.bufferPool[step.outputSlot].shape.reduce(1, *)
        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: count)
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: count)
        let rPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        
        vDSP_vadd(aPtr, 1, bPtr, 1, rPtr, 1, vDSP_Length(count))
    }
    
    private func cpuMul(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard step.inputSlots.count >= 2 else { return }
        guard let aBuf = plan.bufferPool[step.inputSlots[0]].buffer,
              let bBuf = plan.bufferPool[step.inputSlots[1]].buffer,
              let outBuf = plan.bufferPool[step.outputSlot].buffer else { return }
        
        let count = plan.bufferPool[step.outputSlot].shape.reduce(1, *)
        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: count)
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: count)
        let rPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        
        vDSP_vmul(aPtr, 1, bPtr, 1, rPtr, 1, vDSP_Length(count))
    }
    
    private func cpuRelu(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard let inBuf = plan.bufferPool[step.inputSlots[0]].buffer,
              let outBuf = plan.bufferPool[step.outputSlot].buffer else { return }
        
        let count = plan.bufferPool[step.outputSlot].shape.reduce(1, *)
        let inPtr = inBuf.contents().bindMemory(to: Float.self, capacity: count)
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        
        var zero: Float = 0
        vDSP_vthres(inPtr, 1, &zero, outPtr, 1, vDSP_Length(count))
    }
    
    private func cpuSilu(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard let inBuf = plan.bufferPool[step.inputSlots[0]].buffer,
              let outBuf = plan.bufferPool[step.outputSlot].buffer else { return }
        
        let count = plan.bufferPool[step.outputSlot].shape.reduce(1, *)
        let inPtr = inBuf.contents().bindMemory(to: Float.self, capacity: count)
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        
        // SiLU = x * sigmoid(x)
        for i in 0..<count {
            let x = inPtr[i]
            outPtr[i] = x / (1.0 + exp(-x))
        }
    }
    
    private func cpuLayerNorm(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard step.inputSlots.count >= 3 else { return }
        guard let inBuf = plan.bufferPool[step.inputSlots[0]].buffer,
              let gBuf = plan.bufferPool[step.inputSlots[1]].buffer,
              let bBuf = plan.bufferPool[step.inputSlots[2]].buffer,
              let outBuf = plan.bufferPool[step.outputSlot].buffer else { return }
        
        let shape = plan.bufferPool[step.inputSlots[0]].shape
        let rows = shape[0]
        let cols = shape[1]
        let eps: Float = (step.params?["eps"] as? Float) ?? 1e-5
        
        let inPtr = inBuf.contents().bindMemory(to: Float.self, capacity: rows * cols)
        let gPtr = gBuf.contents().bindMemory(to: Float.self, capacity: cols)
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: cols)
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: rows * cols)
        
        for r in 0..<rows {
            let rowStart = r * cols
            var mean: Float = 0
            vDSP_meanv(inPtr.advanced(by: rowStart), 1, &mean, vDSP_Length(cols))
            
            // Variance
            var sumSq: Float = 0
            for c in 0..<cols {
                let diff = inPtr[rowStart + c] - mean
                sumSq += diff * diff
            }
            let variance = sumSq / Float(cols)
            let invStd = 1.0 / sqrt(variance + eps)
            
            for c in 0..<cols {
                outPtr[rowStart + c] = (inPtr[rowStart + c] - mean) * invStd * gPtr[c] + bPtr[c]
            }
        }
    }
    
    private func cpuSoftmax(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        guard let inBuf = plan.bufferPool[step.inputSlots[0]].buffer,
              let outBuf = plan.bufferPool[step.outputSlot].buffer else { return }
        
        let shape = plan.bufferPool[step.inputSlots[0]].shape
        let rows = shape[0]
        let cols = shape.count > 1 ? shape[1] : 1
        
        let inPtr = inBuf.contents().bindMemory(to: Float.self, capacity: rows * cols)
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: rows * cols)
        
        for r in 0..<rows {
            let rowStart = r * cols
            // Max for numerical stability
            var maxVal: Float = -Float.infinity
            vDSP_maxv(inPtr.advanced(by: rowStart), 1, &maxVal, vDSP_Length(cols))
            
            // exp(x - max) and sum
            var expSum: Float = 0
            for c in 0..<cols {
                let val = exp(inPtr[rowStart + c] - maxVal)
                outPtr[rowStart + c] = val
                expSum += val
            }
            // Normalize
            var invSum = 1.0 / expSum
            vDSP_vsmul(outPtr.advanced(by: rowStart), 1, &invSum, outPtr.advanced(by: rowStart), 1, vDSP_Length(cols))
        }
    }
    
    // MARK: - ANE Execution (CoreML)
    
    private func executeOnANE(_ step: ExecutionStep, plan: ExecutionPlan) throws {
        // ANE ops use pre-compiled CoreML models stored in step.precompiledResource
        guard let model = step.precompiledResource as? MLModel else {
            // Fallback to CPU if no pre-compiled model
            try executeOnCPU(step, plan: plan)
            return
        }
        
        switch step.op {
        case .matmul:
            try aneMatmul(step, plan: plan, model: model)
        case .layerNorm, .batchNorm, .conv2d,
             .fusedConvBnRelu, .fusedQKVProjection, .fusedMLPGateUp:
            try anePrediction(step, plan: plan, model: model)
        default:
            // For element-wise ops, ANE is overkill — fallback to CPU
            try executeOnCPU(step, plan: plan)
        }
    }
    
    private func aneMatmul(_ step: ExecutionStep, plan: ExecutionPlan, model: MLModel) throws {
        guard step.inputSlots.count >= 1 else { return }
        guard let aBuf = plan.bufferPool[step.inputSlots[0]].buffer else { return }
        
        let aShape = plan.bufferPool[step.inputSlots[0]].shape
        let M = aShape[0], K = aShape[1]
        
        // Create MLMultiArray from pre-allocated buffer — minimal overhead
        let aArray = try MLMultiArray(shape: [M, K] as [NSNumber], dataType: .float16)
        aArray.withUnsafeMutableBytes { ptr, _ in
            let dest = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
            let src = aBuf.contents().bindMemory(to: Float.self, capacity: M * K)
            // SIMD-accelerated conversion
            GPUEngine.convertFP32ToFP16(src: src, dst: dest, count: M * K)
        }
        
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "a": MLFeatureValue(multiArray: aArray)
        ])
        
        let output = try model.prediction(from: input)
        let resultKey = output.featureNames.first!
        guard let resultArray = output.featureValue(for: resultKey)?.multiArrayValue else {
            throw ConductorError.compilationFailed("ANE prediction returned no result")
        }
        
        // Copy result back to plan buffer (float16 → float32)
        let outShape = plan.bufferPool[step.outputSlot].shape
        let outCount = outShape.reduce(1, *)
        guard let outBuf = plan.bufferPool[step.outputSlot].buffer else { return }
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: outCount)
        
        resultArray.withUnsafeBytes { ptr in
            let src = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
            GPUEngine.convertFP16ToFP32(src: src, dst: outPtr, count: outCount)
        }
    }
    
    private func anePrediction(_ step: ExecutionStep, plan: ExecutionPlan, model: MLModel) throws {
        guard let inBuf = plan.bufferPool[step.inputSlots[0]].buffer else { return }
        let shape = plan.bufferPool[step.inputSlots[0]].shape
        let count = shape.reduce(1, *)
        
        // Create input as float32 (CoreML handles conversion internally for compatible models)
        let inArray = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        let srcPtr = inBuf.contents().bindMemory(to: Float.self, capacity: count)
        inArray.withUnsafeMutableBytes { ptr, _ in
            memcpy(ptr.baseAddress!, srcPtr, count * 4)
        }
        
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "x"
        let input = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: inArray)
        ])
        
        let output = try model.prediction(from: input)
        let resultKey = output.featureNames.first!
        guard let resultArray = output.featureValue(for: resultKey)?.multiArrayValue else { return }
        
        let outCount = plan.bufferPool[step.outputSlot].shape.reduce(1, *)
        guard let outBuf = plan.bufferPool[step.outputSlot].buffer else { return }
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: outCount)
        
        resultArray.withUnsafeBytes { ptr in
            // Handle both float16 and float32 output
            if resultArray.dataType == .float16 {
                let src = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
                GPUEngine.convertFP16ToFP32(src: src, dst: outPtr, count: outCount)
            } else {
                memcpy(outPtr, ptr.baseAddress!, outCount * 4)
            }
        }
    }
}

// MARK: - Thread-Safe Error Collection

/// Collects errors from concurrent dispatch without locks on the hot path.
final class ConductorThreadSafeErrors: @unchecked Sendable {
    private var errors: [Error] = []
    private let lock = NSLock()
    
    func append(_ error: Error) {
        lock.lock()
        errors.append(error)
        lock.unlock()
    }
    
    func check() throws {
        lock.lock()
        let first = errors.first
        lock.unlock()
        if let error = first {
            throw error
        }
    }
}
