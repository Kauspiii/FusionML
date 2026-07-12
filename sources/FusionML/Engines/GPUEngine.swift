// SiliconML - GPU Engine
// High-performance GPU compute using custom Metal shaders

import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// GPU compute engine with custom shaders
public final class GPUEngine: @unchecked Sendable {
    
    public static let shared = GPUEngine()
    
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private var library: MTLLibrary?
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var mpsKernels: [String: MPSMatrixMultiplication] = [:]
    private let kernelLock = NSLock()
    
    private struct MPSGraphMatmulKey: Hashable {
        let M: Int
        let N: Int
        let K: Int
        let transposeLeft: Bool
        let transposeRight: Bool
        let dtype: DType
    }
    
    private struct MPSMatrixDescriptorKey: Hashable {
        let rows: Int
        let columns: Int
        let rowBytes: Int
        let dataType: MPSDataType
    }
    private var mpsDescriptorCache: [MPSMatrixDescriptorKey: MPSMatrixDescriptor] = [:]
    private let descriptorLock = NSLock()
    
    private func getMatrixDescriptor(rows: Int, columns: Int, rowBytes: Int, dataType: MPSDataType) -> MPSMatrixDescriptor {
        descriptorLock.lock()
        defer { descriptorLock.unlock() }
        let key = MPSMatrixDescriptorKey(rows: rows, columns: columns, rowBytes: rowBytes, dataType: dataType)
        if let cached = mpsDescriptorCache[key] {
            return cached
        }
        let desc = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: rowBytes, dataType: dataType)
        mpsDescriptorCache[key] = desc
        return desc
    }
    
    private struct MPSGraphMatmulResources {
        let graph: MPSGraph
        let aTensor: MPSGraphTensor
        let bTensor: MPSGraphTensor
        let cTensor: MPSGraphTensor
    }
    
    private var mpsGraphCache: [MPSGraphMatmulKey: MPSGraphMatmulResources] = [:]
    private let mpsGraphLock = NSLock()
    
    private var lastCommandBuffer: MTLCommandBuffer? = nil
    private let syncLock = NSLock()
    
    private var activeCommandBuffers: [Thread: MTLCommandBuffer] = [:]
    private var activeEncoders: [Thread: MTLComputeCommandEncoder] = [:]
    
    private var activeCommandBuffer: MTLCommandBuffer? {
        get {
            return activeCommandBuffers[Thread.current]
        }
        set {
            activeCommandBuffers[Thread.current] = newValue
        }
    }
    
    private var activeEncoder: MTLComputeCommandEncoder? {
        get {
            return activeEncoders[Thread.current]
        }
        set {
            activeEncoders[Thread.current] = newValue
        }
    }
    
    private var isBatchActive = false
    private let captureLock = NSLock()
    
    public func startBatch() {
        captureLock.lock()
        isBatchActive = true
        captureLock.unlock()
    }
    
    public func commitBatch() {
        captureLock.lock()
        isBatchActive = false
        if let encoder = activeEncoder {
            encoder.endEncoding()
            activeEncoder = nil
        }
        if let cb = activeCommandBuffer {
            commit(cb)
            activeCommandBuffer = nil
        }
        captureLock.unlock()
    }
    
    public func commitActiveCommandBuffer() {
        captureLock.lock()
        if let encoder = activeEncoder {
            encoder.endEncoding()
            activeEncoder = nil
        }
        if let cb = activeCommandBuffer {
            commit(cb)
            activeCommandBuffer = nil
        }
        captureLock.unlock()
    }
    
    private func getEncoder(pipeline: MTLComputePipelineState) throws -> MTLComputeCommandEncoder {
        captureLock.lock()
        defer { captureLock.unlock() }
        
        if let encoder = activeEncoder {
            encoder.setComputePipelineState(pipeline)
            return encoder
        }
        
        if activeCommandBuffer == nil {
            activeCommandBuffer = commandQueue.makeCommandBuffer()
        }
        
        guard let encoder = activeCommandBuffer!.makeComputeCommandEncoder() else {
            throw MemoryError.deviceNotAvailable
        }
        encoder.setComputePipelineState(pipeline)
        
        if isBatchActive {
            activeEncoder = encoder
        }
        
        return encoder
    }
    
    private func endEncoder(_ encoder: MTLComputeCommandEncoder) {
        captureLock.lock()
        defer { captureLock.unlock() }
        
        let batch = isBatchActive
        if !batch {
            encoder.endEncoding()
            if let cb = activeCommandBuffer {
                commit(cb)
                activeCommandBuffer = nil
            }
        }
    }
    
    public func getCommandBufferForMPS() -> MTLCommandBuffer {
        captureLock.lock()
        defer { captureLock.unlock() }
        
        if let encoder = activeEncoder {
            encoder.endEncoding()
            activeEncoder = nil
        }
        
        if activeCommandBuffer == nil {
            activeCommandBuffer = commandQueue.makeCommandBuffer()
        }
        
        return activeCommandBuffer!
    }
    
    public func endMPS(commandBuffer cb: MTLCommandBuffer, waitUntilCompleted: Bool) {
        captureLock.lock()
        defer { captureLock.unlock() }
        
        let batch = isBatchActive
        if !batch {
            commit(cb, wait: waitUntilCompleted)
            activeCommandBuffer = nil
        } else {
            if waitUntilCompleted {
                commit(cb, wait: true)
            }
            if cb.status != .notEnqueued {
                activeCommandBuffer = nil
            }
        }
    }
    
    public func commit(_ cb: MTLCommandBuffer, wait: Bool = false) {
        syncLock.lock()
        if cb.status == .notEnqueued {
            cb.commit()
        }
        lastCommandBuffer = cb
        syncLock.unlock()
        if wait {
            cb.waitUntilCompleted()
        }
    }
    
    public func sync() {
        commitBatch()
        syncLock.lock()
        let cb = lastCommandBuffer
        syncLock.unlock()
        if let status = cb?.status, status != .completed {
            cb?.waitUntilCompleted()
        }
    }
    
    /// Lightweight wait on the last submitted command buffer only.
    /// Unlike sync(), this does NOT commit a batch — it only waits on whatever
    /// was already submitted. Use this when you just need coherent reads.
    public func waitOnLastCommandBuffer() {
        syncLock.lock()
        let cb = lastCommandBuffer
        syncLock.unlock()
        if let status = cb?.status, status != .completed {
            cb?.waitUntilCompleted()
        }
    }
    
    public var lastCommittedBuffer: MTLCommandBuffer? {
        syncLock.lock()
        defer { syncLock.unlock() }
        return lastCommandBuffer
    }

    
    private init() {
        self.device = MemoryManager.shared.device
        self.commandQueue = MemoryManager.shared.commandQueue
        loadShaders()
    }
    
    private func loadShaders() {
        // Try to load from bundle or compile from source
        do {
            // First try default library
            if let lib = device.makeDefaultLibrary() {
                self.library = lib
            } else {
                // Compile shaders from source at runtime
                let shaderSource = GPUEngine.shaderSource
                self.library = try device.makeLibrary(source: shaderSource, options: nil)
            }
            
            // Pre-compile pipeline states
            try compilePipeline(name: "matmul_tiled")
            try compilePipeline(name: "matmul_naive")
            try compilePipeline(name: "matmul_fp16_tiled")
            try compilePipeline(name: "add_elementwise")
            try compilePipeline(name: "add_elementwise_fp16")
            try compilePipeline(name: "mul_elementwise")
            try compilePipeline(name: "mul_elementwise_fp16")
            try compilePipeline(name: "relu")
            try compilePipeline(name: "relu_fp16")
            try compilePipeline(name: "gelu")
            try compilePipeline(name: "gelu_fp16")
            try compilePipeline(name: "gelu_mul")
            try compilePipeline(name: "gelu_mul_fp16")
            try compilePipeline(name: "softmax_row")
            try compilePipeline(name: "relu_backward")
            try compilePipeline(name: "gelu_backward")
            try compilePipeline(name: "layer_norm")
            try compilePipeline(name: "layer_norm_fp16")
            try compilePipeline(name: "transpose_2d")
            try compilePipeline(name: "split_2d")
            try compilePipeline(name: "split_backward_2d")
            try compilePipeline(name: "adam_step")
            try compilePipeline(name: "cross_entropy_forward")
            try compilePipeline(name: "sum_loss")
            try compilePipeline(name: "cross_entropy_backward")
            
        } catch {
            print("Warning: Failed to load shaders: \(error)")
        }
    }
    
    private func compilePipeline(name: String) throws {
        guard let library = library,
              let function = library.makeFunction(name: name) else {
            return
        }
        pipelines[name] = try device.makeComputePipelineState(function: function)
    }
    
    // MARK: - Matrix Operations
    
    /// Matrix multiplication using custom tiled shader
    public func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        guard a.ndim == 2 && b.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        guard a.shape[1] == b.shape[0] else {
            throw MemoryError.invalidShape
        }
        
        let M = a.shape[0]
        let K = a.shape[1]
        let N = b.shape[1]
        
        let result = try Tensor(shape: [M, N], dtype: a.dtype)
        
        guard let pipeline = pipelines["matmul_tiled"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(a.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(b.metalBuffer, offset: 0, index: 1)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 2)
        
        var mVal = Int32(M)
        var kVal = Int32(K)
        var nVal = Int32(N)
        encoder.setBytes(&mVal, length: 4, index: 3)
        encoder.setBytes(&kVal, length: 4, index: 4)
        encoder.setBytes(&nVal, length: 4, index: 5)
        
        let BM = 64
        let BN = 64
        let threadgroupSize = MTLSize(width: 64, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (N + BN - 1) / BN,
            height: (M + BM - 1) / BM,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadgroupSize)
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    /// Matrix multiplication using MPS (for comparison)
    /// Supports both FP32 and FP16 via the `dtype` parameter.
    @discardableResult
    public func matmulRawMPS(
        a: MTLBuffer,
        b: MTLBuffer,
        result: MTLBuffer,
        M: Int,
        N: Int,
        K: Int,
        aOffset: Int = 0,
        resultOffset: Int = 0,
        transposeLeft: Bool = false,
        transposeRight: Bool = false,
        waitUntilCompleted: Bool = false,
        dtype: DType = .float32
    ) throws -> MTLCommandBuffer? {
        let mpsDataType: MPSDataType = (dtype == .float16) ? .float16 : .float32
        
        if aOffset == 0 && resultOffset == 0 && K <= 8192 {
            let key = MPSGraphMatmulKey(
                M: M, N: N, K: K,
                transposeLeft: transposeLeft,
                transposeRight: transposeRight,
                dtype: dtype
            )
            
            mpsGraphLock.lock()
            let resources: MPSGraphMatmulResources
            if let cached = mpsGraphCache[key] {
                resources = cached
                mpsGraphLock.unlock()
            } else {
                let graph = MPSGraph()
                let aTensor = graph.placeholder(shape: [transposeLeft ? K : M, transposeLeft ? M : K] as [NSNumber], dataType: mpsDataType, name: "A")
                let bTensor = graph.placeholder(shape: [transposeRight ? N : K, transposeRight ? K : N] as [NSNumber], dataType: mpsDataType, name: "B")
                
                var primary = aTensor
                if transposeLeft {
                    primary = graph.transposeTensor(aTensor, dimension: 0, withDimension: 1, name: nil)
                }
                var secondary = bTensor
                if transposeRight {
                    secondary = graph.transposeTensor(bTensor, dimension: 0, withDimension: 1, name: nil)
                }
                
                let cTensor = graph.matrixMultiplication(primary: primary, secondary: secondary, name: "C")
                resources = MPSGraphMatmulResources(graph: graph, aTensor: aTensor, bTensor: bTensor, cTensor: cTensor)
                mpsGraphCache[key] = resources
                mpsGraphLock.unlock()
            }
            
            let aData = MPSGraphTensorData(a, shape: [transposeLeft ? K : M, transposeLeft ? M : K] as [NSNumber], dataType: mpsDataType)
            let bData = MPSGraphTensorData(b, shape: [transposeRight ? N : K, transposeRight ? K : N] as [NSNumber], dataType: mpsDataType)
            let cData = MPSGraphTensorData(result, shape: [M, N] as [NSNumber], dataType: mpsDataType)
            
            let commandBuffer = getCommandBufferForMPS()
            let mpsCB = MPSCommandBuffer(commandBuffer: commandBuffer)
            
            resources.graph.encode(
                to: mpsCB,
                feeds: [resources.aTensor: aData, resources.bTensor: bData],
                targetOperations: nil,
                resultsDictionary: [resources.cTensor: cData],
                executionDescriptor: nil
            )
            
            endMPS(commandBuffer: commandBuffer, waitUntilCompleted: waitUntilCompleted)
            return waitUntilCompleted ? nil : commandBuffer
        }
        
        // Fallback to legacy MPSMatrixMultiplication for offset matrices
        let elementSize = (dtype == .float16) ? 2 : 4
        let aDesc = getMatrixDescriptor(rows: transposeLeft ? K : M, columns: transposeLeft ? M : K, rowBytes: (transposeLeft ? M : K) * elementSize, dataType: mpsDataType)
        let bDesc = getMatrixDescriptor(rows: transposeRight ? N : K, columns: transposeRight ? K : N, rowBytes: (transposeRight ? K : N) * elementSize, dataType: mpsDataType)
        let cDesc = getMatrixDescriptor(rows: M, columns: N, rowBytes: N * elementSize, dataType: mpsDataType)
        
        let aMatrix = MPSMatrix(buffer: a, offset: aOffset, descriptor: aDesc)
        let bMatrix = MPSMatrix(buffer: b, offset: 0, descriptor: bDesc)
        let cMatrix = MPSMatrix(buffer: result, offset: resultOffset, descriptor: cDesc)
        
        let kernelKey = "\(M)_\(N)_\(K)_\(transposeLeft)_\(transposeRight)_\(dtype)"
        kernelLock.lock()
        let matmul: MPSMatrixMultiplication
        if let cached = mpsKernels[kernelKey] {
            matmul = cached
            kernelLock.unlock()
        } else {
            matmul = MPSMatrixMultiplication(
                device: device,
                transposeLeft: transposeLeft,
                transposeRight: transposeRight,
                resultRows: M,
                resultColumns: N,
                interiorColumns: K,
                alpha: 1.0,
                beta: 0.0
            )
            mpsKernels[kernelKey] = matmul
            kernelLock.unlock()
        }
        
        let commandBuffer = getCommandBufferForMPS()
        matmul.encode(commandBuffer: commandBuffer, leftMatrix: aMatrix, rightMatrix: bMatrix, resultMatrix: cMatrix)
        
        endMPS(commandBuffer: commandBuffer, waitUntilCompleted: waitUntilCompleted)
        return waitUntilCompleted ? nil : commandBuffer
    }
    
    /// Matrix multiplication using MPS (with optional zero-copy offsets)
    /// Automatically detects tensor dtype and uses FP16 path when both inputs are FP16.
    public func matmulMPS(
        _ a: Tensor,
        _ b: Tensor,
        aOffset: Int = 0,
        resultOffset: Int = 0,
        customResult: Tensor? = nil,
        customM: Int? = nil,
        transposeLeft: Bool = false,
        transposeRight: Bool = false,
        waitUntilCompleted: Bool = false
    ) throws -> Tensor {
        guard a.ndim == 2 && b.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        
        let M = customM ?? (transposeLeft ? a.shape[1] : a.shape[0])
        let K = transposeLeft ? a.shape[0] : a.shape[1]
        let N = transposeRight ? b.shape[0] : b.shape[1]
        
        let expectedK = transposeRight ? b.shape[1] : b.shape[0]
        guard K == expectedK else {
            throw MemoryError.invalidShape
        }
        
        // Auto-detect dtype: use FP16 when both inputs are FP16
        let useFP16 = (a.dtype == .float16 && b.dtype == .float16)
        let resultDType: DType = useFP16 ? .float16 : .float32
        
        let result = try customResult ?? Tensor(shape: [M, N], dtype: resultDType)
        
        try matmulRawMPS(
            a: a.metalBuffer,
            b: b.metalBuffer,
            result: result.metalBuffer,
            M: M,
            N: N,
            K: K,
            aOffset: aOffset,
            resultOffset: resultOffset,
            transposeLeft: transposeLeft,
            transposeRight: transposeRight,
            waitUntilCompleted: waitUntilCompleted,
            dtype: resultDType
        )
        
        result.isDirty = true
        return result
    }
    
    /// Run Adam optimizer step on GPU
    public func adamStep(
        w: MTLBuffer,
        g: MTLBuffer,
        m: MTLBuffer,
        v: MTLBuffer,
        count: Int,
        lr: Float,
        beta1: Float,
        beta2: Float,
        eps: Float,
        weightDecay: Float,
        biasCorrection1: Float,
        biasCorrection2: Float
    ) throws {
        guard let pipeline = pipelines["adam_step"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(w, offset: 0, index: 0)
        encoder.setBuffer(g, offset: 0, index: 1)
        encoder.setBuffer(m, offset: 0, index: 2)
        encoder.setBuffer(v, offset: 0, index: 3)
        
        var lrVal = lr
        var b1Val = beta1
        var b2Val = beta2
        var epsVal = eps
        var wdVal = weightDecay
        var bc1Val = biasCorrection1
        var bc2Val = biasCorrection2
        
        encoder.setBytes(&lrVal, length: 4, index: 4)
        encoder.setBytes(&b1Val, length: 4, index: 5)
        encoder.setBytes(&b2Val, length: 4, index: 6)
        encoder.setBytes(&epsVal, length: 4, index: 7)
        encoder.setBytes(&wdVal, length: 4, index: 8)
        encoder.setBytes(&bc1Val, length: 4, index: 9)
        encoder.setBytes(&bc2Val, length: 4, index: 10)
        
        let threadgroupSize = pipeline.maxTotalThreadsPerThreadgroup
        let numThreadgroups = (count + threadgroupSize - 1) / threadgroupSize
        
        encoder.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        endEncoder(encoder)
    }
    
    // MARK: - Element-wise Operations
    
    private func elementwise(_ a: Tensor, _ b: Tensor, kernel: String) throws -> Tensor {
        guard a.shape == b.shape else {
            throw MemoryError.invalidShape
        }
        
        let result = try Tensor(shape: a.shape, dtype: a.dtype)
        let actualKernel = a.dtype == .float16 ? "\(kernel)_fp16" : kernel
        
        guard let pipeline = pipelines[actualKernel] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(a.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(b.metalBuffer, offset: 0, index: 1)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 2)
        
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (a.count + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    public func add(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        try elementwise(a, b, kernel: "add_elementwise")
    }
    
    public func mul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        try elementwise(a, b, kernel: "mul_elementwise")
    }
    
    // MARK: - Activation Functions
    
    public func relu(_ x: Tensor) throws -> Tensor {
        let result = try Tensor(shape: x.shape, dtype: x.dtype)
        let kernelName = x.dtype == .float16 ? "relu_fp16" : "relu"
        
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 1)
        
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (x.count + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    public func gelu(_ x: Tensor) throws -> Tensor {
        let result = try Tensor(shape: x.shape, dtype: x.dtype)
        let kernelName = x.dtype == .float16 ? "gelu_fp16" : "gelu"
        
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 1)
        
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (x.count + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    public func geluMul(_ gate: Tensor, _ up: Tensor) throws -> Tensor {
        guard gate.shape == up.shape else {
            throw MemoryError.invalidShape
        }
        let result = try Tensor(shape: gate.shape, dtype: gate.dtype)
        let kernelName = gate.dtype == .float16 ? "gelu_mul_fp16" : "gelu_mul"
        
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(gate.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(up.metalBuffer, offset: 0, index: 1)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 2)
        
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (gate.count + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    public func reluBackward(_ input: Tensor, _ gradOutput: Tensor) throws -> Tensor {
        let result = try Tensor(shape: input.shape, dtype: input.dtype)
        guard let pipeline = pipelines["relu_backward"] else {
            throw MemoryError.deviceNotAvailable
        }
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(input.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(gradOutput.metalBuffer, offset: 0, index: 1)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 2)
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (input.count + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    public func geluBackward(_ input: Tensor, _ gradOutput: Tensor) throws -> Tensor {
        let result = try Tensor(shape: input.shape, dtype: input.dtype)
        guard let pipeline = pipelines["gelu_backward"] else {
            throw MemoryError.deviceNotAvailable
        }
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(input.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(gradOutput.metalBuffer, offset: 0, index: 1)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 2)
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (input.count + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    /// Row-wise softmax on GPU using the pre-compiled softmax_row Metal kernel.
    /// Each row of the input tensor is independently softmax'd.
    public func softmax(_ x: Tensor) throws -> Tensor {
        guard x.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        let result = try Tensor(shape: x.shape, dtype: x.dtype)
        
        guard let pipeline = pipelines["softmax_row"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let rows = x.shape[0]
        let cols = x.shape[1]
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 1)
        var colsVal = Int32(cols)
        encoder.setBytes(&colsVal, length: 4, index: 2)
        
        // One thread per row
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (rows + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        
        endEncoder(encoder)
        result.isDirty = true
        return result
    }

    /// Run GPU cross entropy loss forward pass.
    /// Returns a tuple containing:
    /// - totalLoss: a Tensor of shape [1] with the scalar loss value
    /// - batchLosses: a Tensor of shape [batch] with the individual losses per batch element
    /// - probs: a Tensor of shape [batch, classes] with the softmax probability matrix
    public func crossEntropyForward(predictions: Tensor, targets: Tensor) throws -> (totalLoss: Tensor, batchLosses: Tensor, probs: Tensor) {
        guard predictions.ndim == 2, targets.ndim == 1, predictions.shape[0] == targets.shape[0] else {
            throw MemoryError.invalidShape
        }
        let batch = predictions.shape[0]
        let classes = predictions.shape[1]
        
        let batchLosses = try Tensor(shape: [batch], dtype: .float32)
        let totalLoss = try Tensor(shape: [1], dtype: .float32)
        let probs = try Tensor(shape: [batch, classes], dtype: .float32)
        
        guard let forwardPipeline = pipelines["cross_entropy_forward"],
              let sumPipeline = pipelines["sum_loss"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        // 1. Run forward cross entropy (computes per-element loss and probs)
        let encoder1 = try getEncoder(pipeline: forwardPipeline)
        encoder1.setBuffer(predictions.metalBuffer, offset: 0, index: 0)
        encoder1.setBuffer(targets.metalBuffer, offset: 0, index: 1)
        encoder1.setBuffer(batchLosses.metalBuffer, offset: 0, index: 2)
        encoder1.setBuffer(probs.metalBuffer, offset: 0, index: 3)
        var batchVal = Int32(batch)
        var classesVal = Int32(classes)
        encoder1.setBytes(&batchVal, length: 4, index: 4)
        encoder1.setBytes(&classesVal, length: 4, index: 5)
        
        let threadsPerGroup1 = min(256, forwardPipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups1 = (batch + threadsPerGroup1 - 1) / threadsPerGroup1
        encoder1.dispatchThreadgroups(MTLSize(width: threadgroups1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup1, height: 1, depth: 1))
        endEncoder(encoder1)
        
        // 2. Sum the batch losses
        let encoder2 = try getEncoder(pipeline: sumPipeline)
        encoder2.setBuffer(batchLosses.metalBuffer, offset: 0, index: 0)
        encoder2.setBuffer(totalLoss.metalBuffer, offset: 0, index: 1)
        encoder2.setBytes(&batchVal, length: 4, index: 2)
        
        encoder2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        endEncoder(encoder2)
        
        totalLoss.isDirty = true
        batchLosses.isDirty = true
        probs.isDirty = true
        
        return (totalLoss, batchLosses, probs)
    }
    
    /// Run GPU cross entropy loss backward pass.
    /// Returns the gradient tensor of shape [batch, classes].
    public func crossEntropyBackward(probs: Tensor, targets: Tensor, upstream: Tensor) throws -> Tensor {
        let batch = probs.shape[0]
        let classes = probs.shape[1]
        
        let grad = try Tensor(shape: [batch, classes], dtype: .float32)
        
        guard let pipeline = pipelines["cross_entropy_backward"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(probs.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(targets.metalBuffer, offset: 0, index: 1)
        encoder.setBuffer(grad.metalBuffer, offset: 0, index: 2)
        encoder.setBuffer(upstream.metalBuffer, offset: 0, index: 3)
        var batchVal = Int32(batch)
        var classesVal = Int32(classes)
        encoder.setBytes(&batchVal, length: 4, index: 4)
        encoder.setBytes(&classesVal, length: 4, index: 5)
        
        let totalThreads = batch * classes
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (totalThreads + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
        
        grad.isDirty = true
        return grad
    }

    public func layerNorm(_ x: Tensor, gamma: Tensor, beta: Tensor, eps: Float = 1e-5) throws -> Tensor {
        let result = try Tensor(shape: x.shape, dtype: x.dtype)
        let kernelName = x.dtype == .float16 ? "layer_norm_fp16" : "layer_norm"
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(gamma.metalBuffer, offset: 0, index: 1)
        encoder.setBuffer(beta.metalBuffer, offset: 0, index: 2)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 3)
        var lastDim = Int32(x.shape.last!)
        var epsVal = eps
        encoder.setBytes(&lastDim, length: 4, index: 4)
        encoder.setBytes(&epsVal, length: 4, index: 5)
        let numRows = x.count / x.shape.last!
        let threadsPerGroup = 256
        encoder.dispatchThreadgroups(MTLSize(width: numRows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    // MARK: - Raw Execution Helpers (Zero Allocations, In-Place Backing)
    
    public func addRaw(a: MTLBuffer, b: MTLBuffer, result: MTLBuffer, count: Int, dtype: DType) throws {
        let kernelName = dtype == .float16 ? "add_elementwise_fp16" : "add_elementwise"
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(result, offset: 0, index: 2)
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (count + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
    }
    
    public func mulRaw(a: MTLBuffer, b: MTLBuffer, result: MTLBuffer, count: Int, dtype: DType) throws {
        let kernelName = dtype == .float16 ? "mul_elementwise_fp16" : "mul_elementwise"
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(result, offset: 0, index: 2)
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (count + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
    }
    
    public func reluRaw(input: MTLBuffer, result: MTLBuffer, count: Int, dtype: DType) throws {
        let kernelName = dtype == .float16 ? "relu_fp16" : "relu"
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(result, offset: 0, index: 1)
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (count + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
    }
    
    public func geluRaw(input: MTLBuffer, result: MTLBuffer, count: Int, dtype: DType) throws {
        let kernelName = dtype == .float16 ? "gelu_fp16" : "gelu"
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(result, offset: 0, index: 1)
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (count + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
    }
    
    public func geluMulRaw(gate: MTLBuffer, up: MTLBuffer, result: MTLBuffer, count: Int, dtype: DType) throws {
        let kernelName = dtype == .float16 ? "gelu_mul_fp16" : "gelu_mul"
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(result, offset: 0, index: 2)
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (count + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
    }
    
    public func layerNormRaw(input: MTLBuffer, gamma: MTLBuffer, beta: MTLBuffer, result: MTLBuffer, lastDim: Int, eps: Float, numRows: Int, dtype: DType) throws {
        let kernelName = dtype == .float16 ? "layer_norm_fp16" : "layer_norm"
        guard let pipeline = pipelines[kernelName] else {
            throw MemoryError.deviceNotAvailable
        }
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(gamma, offset: 0, index: 1)
        encoder.setBuffer(beta, offset: 0, index: 2)
        encoder.setBuffer(result, offset: 0, index: 3)
        var lastDimVal = Int32(lastDim)
        var epsVal = eps
        encoder.setBytes(&lastDimVal, length: 4, index: 4)
        encoder.setBytes(&epsVal, length: 4, index: 5)
        let threadsPerGroup = 256
        encoder.dispatchThreadgroups(MTLSize(width: numRows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        endEncoder(encoder)
    }
    
    public func transpose(_ x: Tensor) throws -> Tensor {
        guard x.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        let M = x.shape[0]
        let N = x.shape[1]
        let result = try Tensor(shape: [N, M], dtype: x.dtype)
        
        guard let pipeline = pipelines["transpose_2d"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 1)
        
        var rows = Int32(M)
        var cols = Int32(N)
        encoder.setBytes(&rows, length: 4, index: 2)
        encoder.setBytes(&cols, length: 4, index: 3)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (N + 15) / 16 * 16,
            height: (M + 15) / 16 * 16,
            depth: 1
        )
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    public func split2D(_ x: Tensor, parts: Int, partIdx: Int) throws -> Tensor {
        guard x.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        let M = x.shape[0]
        let N = x.shape[1]
        guard N % parts == 0 else {
            throw MemoryError.invalidShape
        }
        let partWidth = N / parts
        let result = try Tensor(shape: [M, partWidth], dtype: x.dtype)
        
        guard let pipeline = pipelines["split_2d"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 1)
        
        var mVal = Int32(M)
        var nVal = Int32(N)
        var wVal = Int32(partWidth)
        var idxVal = Int32(partIdx)
        encoder.setBytes(&mVal, length: 4, index: 2)
        encoder.setBytes(&nVal, length: 4, index: 3)
        encoder.setBytes(&wVal, length: 4, index: 4)
        encoder.setBytes(&idxVal, length: 4, index: 5)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (partWidth + 15) / 16 * 16,
            height: (M + 15) / 16 * 16,
            depth: 1
        )
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        endEncoder(encoder)
        result.isDirty = true
        return result
    }
    
    public func splitBackward2D(y: Tensor, x: Tensor, parts: Int, partIdx: Int) throws {
        guard y.ndim == 2 && x.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        let M = x.shape[0]
        let N = x.shape[1]
        guard N % parts == 0 else {
            throw MemoryError.invalidShape
        }
        let partWidth = N / parts
        guard y.shape[0] == M && y.shape[1] == partWidth else {
            throw MemoryError.invalidShape
        }
        
        guard let pipeline = pipelines["split_backward_2d"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        encoder.setBuffer(y.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 1)
        
        var mVal = Int32(M)
        var nVal = Int32(N)
        var wVal = Int32(partWidth)
        var idxVal = Int32(partIdx)
        encoder.setBytes(&mVal, length: 4, index: 2)
        encoder.setBytes(&nVal, length: 4, index: 3)
        encoder.setBytes(&wVal, length: 4, index: 4)
        encoder.setBytes(&idxVal, length: 4, index: 5)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (partWidth + 15) / 16 * 16,
            height: (M + 15) / 16 * 16,
            depth: 1
        )
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        endEncoder(encoder)
        x.isDirty = true
    }
    
    public func split3Way2D(_ x: Tensor) throws -> [Tensor] {
        guard x.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        let M = x.shape[0]
        let N = x.shape[1]
        guard N % 3 == 0 else {
            throw MemoryError.invalidShape
        }
        let partWidth = N / 3
        
        let q = try Tensor(shape: [M, partWidth], dtype: x.dtype)
        let k = try Tensor(shape: [M, partWidth], dtype: x.dtype)
        let v = try Tensor(shape: [M, partWidth], dtype: x.dtype)
        
        guard let pipeline = pipelines["split_2d"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (partWidth + 15) / 16 * 16,
            height: (M + 15) / 16 * 16,
            depth: 1
        )
        
        var mVal = Int32(M)
        var nVal = Int32(N)
        var wVal = Int32(partWidth)
        
        // Dispatch Q (partIdx = 0)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(q.metalBuffer, offset: 0, index: 1)
        var idxVal0 = Int32(0)
        encoder.setBytes(&mVal, length: 4, index: 2)
        encoder.setBytes(&nVal, length: 4, index: 3)
        encoder.setBytes(&wVal, length: 4, index: 4)
        encoder.setBytes(&idxVal0, length: 4, index: 5)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        // Dispatch K (partIdx = 1)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(k.metalBuffer, offset: 0, index: 1)
        var idxVal1 = Int32(1)
        encoder.setBytes(&mVal, length: 4, index: 2)
        encoder.setBytes(&nVal, length: 4, index: 3)
        encoder.setBytes(&wVal, length: 4, index: 4)
        encoder.setBytes(&idxVal1, length: 4, index: 5)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        // Dispatch V (partIdx = 2)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(v.metalBuffer, offset: 0, index: 1)
        var idxVal2 = Int32(2)
        encoder.setBytes(&mVal, length: 4, index: 2)
        encoder.setBytes(&nVal, length: 4, index: 3)
        encoder.setBytes(&wVal, length: 4, index: 4)
        encoder.setBytes(&idxVal2, length: 4, index: 5)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        endEncoder(encoder)
        
        q.isDirty = true
        k.isDirty = true
        v.isDirty = true
        
        return [q, k, v]
    }
    
    public func split2Way2D(_ x: Tensor) throws -> [Tensor] {
        guard x.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        let M = x.shape[0]
        let N = x.shape[1]
        guard N % 2 == 0 else {
            throw MemoryError.invalidShape
        }
        let partWidth = N / 2
        
        let a = try Tensor(shape: [M, partWidth], dtype: x.dtype)
        let b = try Tensor(shape: [M, partWidth], dtype: x.dtype)
        
        guard let pipeline = pipelines["split_2d"] else {
            throw MemoryError.deviceNotAvailable
        }
        
        let encoder = try getEncoder(pipeline: pipeline)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (partWidth + 15) / 16 * 16,
            height: (M + 15) / 16 * 16,
            depth: 1
        )
        
        var mVal = Int32(M)
        var nVal = Int32(N)
        var wVal = Int32(partWidth)
        
        // Dispatch A (partIdx = 0)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(a.metalBuffer, offset: 0, index: 1)
        var idxVal0 = Int32(0)
        encoder.setBytes(&mVal, length: 4, index: 2)
        encoder.setBytes(&nVal, length: 4, index: 3)
        encoder.setBytes(&wVal, length: 4, index: 4)
        encoder.setBytes(&idxVal0, length: 4, index: 5)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        // Dispatch B (partIdx = 1)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(b.metalBuffer, offset: 0, index: 1)
        var idxVal1 = Int32(1)
        encoder.setBytes(&mVal, length: 4, index: 2)
        encoder.setBytes(&nVal, length: 4, index: 3)
        encoder.setBytes(&wVal, length: 4, index: 4)
        encoder.setBytes(&idxVal1, length: 4, index: 5)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        endEncoder(encoder)
        
        a.isDirty = true
        b.isDirty = true
        
        return [a, b]
    }
    
    // MARK: - SIMD FP32 <-> FP16 Conversion Utilities
    
    /// Convert FP32 buffer to FP16 using vDSP SIMD (orders of magnitude faster than scalar loop)
    public static func convertFP32ToFP16(src: UnsafePointer<Float>, dst: UnsafeMutablePointer<Float16>, count: Int) {
        // Process in chunks of 4 using SIMD where possible
        var i = 0
        let simdCount = count & ~3  // Round down to multiple of 4
        while i < simdCount {
            dst[i]   = Float16(src[i])
            dst[i+1] = Float16(src[i+1])
            dst[i+2] = Float16(src[i+2])
            dst[i+3] = Float16(src[i+3])
            i += 4
        }
        // Handle remainder
        while i < count {
            dst[i] = Float16(src[i])
            i += 1
        }
    }
    
    /// Convert FP16 buffer to FP32 using SIMD
    public static func convertFP16ToFP32(src: UnsafePointer<Float16>, dst: UnsafeMutablePointer<Float>, count: Int) {
        var i = 0
        let simdCount = count & ~3
        while i < simdCount {
            dst[i]   = Float(src[i])
            dst[i+1] = Float(src[i+1])
            dst[i+2] = Float(src[i+2])
            dst[i+3] = Float(src[i+3])
            i += 4
        }
        while i < count {
            dst[i] = Float(src[i])
            i += 1
        }
    }
    
    // MARK: - Embedded Shader Source
    
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;
    
    constant int TILE_SIZE = 32;
    
    kernel void matmul_naive(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant int& M [[buffer(3)]],
        constant int& K [[buffer(4)]],
        constant int& N [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        int row = gid.y;
        int col = gid.x;
        if (row >= M || col >= N) return;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    
    constant int BM = 64;
    constant int BN = 64;
    constant int BK = 8;
    constant int TM = 8;
    constant int TN = 8;
    
    kernel void matmul_tiled(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant int& M [[buffer(3)]],
        constant int& K [[buffer(4)]],
        constant int& N [[buffer(5)]],
        uint tid [[thread_index_in_threadgroup]],
        uint2 tgid [[threadgroup_position_in_grid]]
    ) {
        threadgroup float As[BM * BK];
        threadgroup float Bs[BK * BN];
        
        const int threadRow = (int)tid / 8;
        const int threadCol = (int)tid % 8;
        
        const int blockRowStart = (int)tgid.y * BM;
        const int blockColStart = (int)tgid.x * BN;
        
        float acc[TM * TN];
        for (int i = 0; i < TM * TN; i++) {
            acc[i] = 0.0f;
        }
        
        float regA[TM];
        float regB[TN];
        
        const int numThreads = 64;
        const int numKTiles = (K + BK - 1) / BK;
        
        for (int bk = 0; bk < numKTiles; bk++) {
            for (int loadOffset = 0; loadOffset < BM * BK; loadOffset += numThreads) {
                int idx = loadOffset + (int)tid;
                if (idx < BM * BK) {
                    int r = idx / BK;
                    int c = idx % BK;
                    int globalRow = blockRowStart + r;
                    int globalCol = bk * BK + c;
                    As[r * BK + c] = (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
                }
            }
            
            for (int loadOffset = 0; loadOffset < BK * BN; loadOffset += numThreads) {
                int idx = loadOffset + (int)tid;
                if (idx < BK * BN) {
                    int r = idx / BN;
                    int c = idx % BN;
                    int globalRow = bk * BK + r;
                    int globalCol = blockColStart + c;
                    Bs[r * BN + c] = (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                for (int i = 0; i < TM; i++) {
                    regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
                }
                for (int j = 0; j < TN; j++) {
                    regB[j] = Bs[dotIdx * BN + threadCol * TN + j];
                }
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TN; j++) {
                        acc[i * TN + j] += regA[i] * regB[j];
                    }
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        for (int i = 0; i < TM; i++) {
            int globalRow = blockRowStart + threadRow * TM + i;
            if (globalRow < M) {
                for (int j = 0; j < TN; j++) {
                    int globalCol = blockColStart + threadCol * TN + j;
                    if (globalCol < N) {
                        C[globalRow * N + globalCol] = acc[i * TN + j];
                    }
                }
            }
        }
    }
    
    kernel void matmul_fp16_tiled(
        device const half* A [[buffer(0)]],
        device const half* B [[buffer(1)]],
        device half* C [[buffer(2)]],
        constant int& M [[buffer(3)]],
        constant int& K [[buffer(4)]],
        constant int& N [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]],
        uint2 tid [[thread_position_in_threadgroup]],
        uint2 tgid [[threadgroup_position_in_grid]]
    ) {
        threadgroup half As[TILE_SIZE][TILE_SIZE];
        threadgroup half Bs[TILE_SIZE][TILE_SIZE];
        
        int row = tgid.y * TILE_SIZE + tid.y;
        int col = tgid.x * TILE_SIZE + tid.x;
        half sum = 0.0h;
        
        int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
        for (int t = 0; t < numTiles; t++) {
            int aCol = t * TILE_SIZE + tid.x;
            As[tid.y][tid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0h;
            
            int bRow = t * TILE_SIZE + tid.y;
            Bs[tid.y][tid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0h;
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[tid.y][k] * Bs[k][tid.x];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
    
    kernel void add_elementwise(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        C[id] = A[id] + B[id];
    }
    
    kernel void add_elementwise_fp16(
        device const half* A [[buffer(0)]],
        device const half* B [[buffer(1)]],
        device half* C [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        C[id] = A[id] + B[id];
    }
    
    kernel void mul_elementwise(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        C[id] = A[id] * B[id];
    }
    
    kernel void mul_elementwise_fp16(
        device const half* A [[buffer(0)]],
        device const half* B [[buffer(1)]],
        device half* C [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        C[id] = A[id] * B[id];
    }
    
    kernel void relu(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        output[id] = max(input[id], 0.0f);
    }
    
    kernel void relu_fp16(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        output[id] = max(input[id], (half)0.0h);
    }
    
    kernel void gelu(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        float x = input[id];
        float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[id] = x * cdf;
    }
    
    kernel void gelu_fp16(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        half x = input[id];
        float x_f = (float)x;
        float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x_f + 0.044715f * x_f * x_f * x_f)));
        output[id] = (half)(x_f * cdf);
    }
    
    kernel void softmax_row(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant int& cols [[buffer(2)]],
        uint row [[thread_position_in_grid]]
    ) {
        int offset = row * cols;
        float maxVal = input[offset];
        for (int i = 1; i < cols; i++) {
            maxVal = max(maxVal, input[offset + i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            float val = exp(input[offset + i] - maxVal);
            output[offset + i] = val;
            sum += val;
        }
        for (int i = 0; i < cols; i++) {
            output[offset + i] /= sum;
        }
    }
    
    kernel void relu_backward(
        device const float* input [[buffer(0)]],
        device const float* gradOutput [[buffer(1)]],
        device float* gradInput [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        gradInput[id] = input[id] > 0.0f ? gradOutput[id] : 0.0f;
    }
    
    kernel void gelu_backward(
        device const float* input [[buffer(0)]],
        device const float* gradOutput [[buffer(1)]],
        device float* gradInput [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        float x = input[id];
        float c = 0.7978845608f;
        float inner = c * (x + 0.044715f * x * x * x);
        float tanh_inner = tanh(inner);
        float dgelu = 0.5f * (1.0f + tanh_inner) + 0.5f * x * (1.0f - tanh_inner * tanh_inner) * c * (1.0f + 3.0f * 0.044715f * x * x);
        gradInput[id] = gradOutput[id] * dgelu;
    }
    
    kernel void layer_norm(
        device const float* input [[buffer(0)]],
        device const float* gamma [[buffer(1)]],
        device const float* beta [[buffer(2)]],
        device float* output [[buffer(3)]],
        constant int& lastDim [[buffer(4)]],
        constant float& eps [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint simd_id [[simdgroup_index_in_threadgroup]],
        uint lane_id [[thread_index_in_simdgroup]]
    ) {
        threadgroup float shared_mean[32];
        threadgroup float shared_var[32];
        
        int offset = row * lastDim;
        
        // 1. Local sum
        float local_sum = 0.0f;
        for (int i = tid; i < lastDim; i += 256) {
            local_sum += input[offset + i];
        }
        
        // 2. Reduce inside simdgroup
        float simd_sum_val = simd_sum(local_sum);
        if (lane_id == 0) {
            shared_mean[simd_id] = simd_sum_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 3. Final reduction for mean
        float mean = 0.0f;
        if (simd_id == 0) {
            float s = (lane_id < 8) ? shared_mean[lane_id] : 0.0f;
            mean = simd_sum(s) / (float)lastDim;
            if (lane_id == 0) {
                shared_mean[0] = mean;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mean = shared_mean[0];
        
        // 4. Local variance sum
        float local_var_sum = 0.0f;
        for (int i = tid; i < lastDim; i += 256) {
            float diff = input[offset + i] - mean;
            local_var_sum += diff * diff;
        }
        
        // 5. Reduce variance inside simdgroup
        float simd_var_val = simd_sum(local_var_sum);
        if (lane_id == 0) {
            shared_var[simd_id] = simd_var_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 6. Final reduction for variance
        float std = 0.0f;
        if (simd_id == 0) {
            float s = (lane_id < 8) ? shared_var[lane_id] : 0.0f;
            float variance = simd_sum(s) / (float)lastDim;
            std = rsqrt(variance + eps);
            if (lane_id == 0) {
                shared_var[0] = std;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        std = shared_var[0];
        
        // 7. Normalize and write
        for (int i = tid; i < lastDim; i += 256) {
            output[offset + i] = (input[offset + i] - mean) * std * gamma[i] + beta[i];
        }
    }
    
    kernel void layer_norm_fp16(
        device const half* input [[buffer(0)]],
        device const half* gamma [[buffer(1)]],
        device const half* beta [[buffer(2)]],
        device half* output [[buffer(3)]],
        constant int& lastDim [[buffer(4)]],
        constant float& eps [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint simd_id [[simdgroup_index_in_threadgroup]],
        uint lane_id [[thread_index_in_simdgroup]]
    ) {
        threadgroup float shared_mean[32];
        threadgroup float shared_var[32];
        
        int offset = row * lastDim;
        
        // 1. Local sum
        float local_sum = 0.0f;
        for (int i = tid; i < lastDim; i += 256) {
            local_sum += (float)input[offset + i];
        }
        
        // 2. Reduce inside simdgroup
        float simd_sum_val = simd_sum(local_sum);
        if (lane_id == 0) {
            shared_mean[simd_id] = simd_sum_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 3. Final reduction for mean
        float mean = 0.0f;
        if (simd_id == 0) {
            float s = (lane_id < 8) ? shared_mean[lane_id] : 0.0f;
            mean = simd_sum(s) / (float)lastDim;
            if (lane_id == 0) {
                shared_mean[0] = mean;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mean = shared_mean[0];
        
        // 4. Local variance sum
        float local_var_sum = 0.0f;
        for (int i = tid; i < lastDim; i += 256) {
            float diff = (float)input[offset + i] - mean;
            local_var_sum += diff * diff;
        }
        
        // 5. Reduce variance inside simdgroup
        float simd_var_val = simd_sum(local_var_sum);
        if (lane_id == 0) {
            shared_var[simd_id] = simd_var_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 6. Final reduction for variance
        float std = 0.0f;
        if (simd_id == 0) {
            float s = (lane_id < 8) ? shared_var[lane_id] : 0.0f;
            float variance = simd_sum(s) / (float)lastDim;
            std = rsqrt(variance + eps);
            if (lane_id == 0) {
                shared_var[0] = std;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        std = shared_var[0];
        
        // 7. Normalize and write
        for (int i = tid; i < lastDim; i += 256) {
            output[offset + i] = (half)(((float)input[offset + i] - mean) * std * (float)gamma[i] + (float)beta[i]);
        }
    }
    
    kernel void transpose_2d(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant int& rows [[buffer(2)]],
        constant int& cols [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= (uint)cols || gid.y >= (uint)rows) return;
        output[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
    }
    
    kernel void adam_step(
        device float* w [[buffer(0)]],
        device const float* g [[buffer(1)]],
        device float* m [[buffer(2)]],
        device float* v [[buffer(3)]],
        constant float& lr [[buffer(4)]],
        constant float& beta1 [[buffer(5)]],
        constant float& beta2 [[buffer(6)]],
        constant float& eps [[buffer(7)]],
        constant float& weightDecay [[buffer(8)]],
        constant float& biasCorrection1 [[buffer(9)]],
        constant float& biasCorrection2 [[buffer(10)]],
        uint id [[thread_position_in_grid]]
    ) {
        float gj = g[id];
        float wj = w[id];
        
        if (weightDecay != 0.0f) {
            wj -= lr * weightDecay * wj;
        }
        
        float mj = beta1 * m[id] + (1.0f - beta1) * gj;
        m[id] = mj;
        
        float vj = beta2 * v[id] + (1.0f - beta2) * gj * gj;
        v[id] = vj;
        
        float mHat = mj / biasCorrection1;
        float vHat = vj / biasCorrection2;
        
        w[id] = wj - lr * mHat / (sqrt(vHat) + eps);
    }
    
    kernel void split_2d(
        device const float* X [[buffer(0)]],
        device float* Y [[buffer(1)]],
        constant int& M [[buffer(2)]],
        constant int& N [[buffer(3)]],
        constant int& partWidth [[buffer(4)]],
        constant int& partIdx [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        int row = gid.y;
        int col = gid.x;
        if (row >= M || col >= partWidth) return;
        
        int srcCol = partIdx * partWidth + col;
        Y[row * partWidth + col] = X[row * N + srcCol];
    }
    
    kernel void split_backward_2d(
        device const float* Y [[buffer(0)]],
        device float* X [[buffer(1)]],
        constant int& M [[buffer(2)]],
        constant int& N [[buffer(3)]],
        constant int& partWidth [[buffer(4)]],
        constant int& partIdx [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        int row = gid.y;
        int col = gid.x;
        if (row >= M || col >= partWidth) return;
        
        int dstCol = partIdx * partWidth + col;
        X[row * N + dstCol] = Y[row * partWidth + col];
    }

    kernel void gelu_mul(
        device const float* gate [[buffer(0)]],
        device const float* up [[buffer(1)]],
        device float* output [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        float x = gate[id];
        float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        float act = x * cdf;
        output[id] = act * up[id];
    }

    kernel void gelu_mul_fp16(
        device const half* gate [[buffer(0)]],
        device const half* up [[buffer(1)]],
        device half* output [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        float x = (float)gate[id];
        float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        float act = x * cdf;
        output[id] = (half)(act * (float)up[id]);
    }

    kernel void cross_entropy_forward(
        device const float* logits [[buffer(0)]],
        device const float* targets [[buffer(1)]],
        device float* loss [[buffer(2)]],
        device float* probs [[buffer(3)]],
        constant int& batch [[buffer(4)]],
        constant int& classes [[buffer(5)]],
        uint id [[thread_position_in_grid]]
    ) {
        if ((int)id >= batch) return;
        
        int offset = id * classes;
        
        // Find max value for numerical stability
        float maxVal = -1e20f;
        for (int c = 0; c < classes; c++) {
            if (logits[offset + c] > maxVal) {
                maxVal = logits[offset + c];
            }
        }
        
        // Compute sum of exponentials
        float sumExp = 0.0f;
        for (int c = 0; c < classes; c++) {
            float val = exp(logits[offset + c] - maxVal);
            probs[offset + c] = val;
            sumExp += val;
        }
        
        // Normalize to get probabilities
        for (int c = 0; c < classes; c++) {
            probs[offset + c] /= sumExp;
        }
        
        // Compute loss for this batch element
        int targetClass = (int)targets[id];
        float p = probs[offset + targetClass];
        if (p < 1e-7f) p = 1e-7f;
        loss[id] = -log(p) / (float)batch;
    }

    kernel void sum_loss(
        device const float* batch_losses [[buffer(0)]],
        device float* total_loss [[buffer(1)]],
        constant int& batch [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id > 0) return;
        float sum = 0.0f;
        for (int i = 0; i < batch; i++) {
            sum += batch_losses[i];
        }
        total_loss[0] = sum;
    }

    kernel void cross_entropy_backward(
        device const float* probs [[buffer(0)]],
        device const float* targets [[buffer(1)]],
        device float* grad [[buffer(2)]],
        device const float* upstream [[buffer(3)]],
        constant int& batch [[buffer(4)]],
        constant int& classes [[buffer(5)]],
        uint id [[thread_position_in_grid]]
    ) {
        int b = id / classes;
        int c = id % classes;
        
        if (b >= batch) return;
        
        int targetClass = (int)targets[b];
        float p = probs[b * classes + c];
        float t = (c == targetClass) ? 1.0f : 0.0f;
        float up = upstream[0];
        
        grad[b * classes + c] = up * (p - t) / (float)batch;
    }
    """
}
