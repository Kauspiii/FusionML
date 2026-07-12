// SiliconML - Tensor
// High-performance tensor backed by unified memory

import Metal
import Accelerate
import Foundation

/// A tensor with unified memory backing
/// Zero-copy access from CPU, GPU, and ANE
public final class Tensor: @unchecked Sendable, CustomStringConvertible {
    
    // MARK: - Properties
    
    public let shape: [Int]
    public let dtype: DType
    
    public var buffer: UnifiedBuffer {
        if isDirty {
            GPUEngine.shared.sync()
            isDirty = false
        }
        return _buffer
    }
    public let _buffer: UnifiedBuffer
    public var isParameter: Bool = false
    public var isDirty: Bool = false
    
    /// Total number of elements
    public var count: Int {
        shape.reduce(1, *)
    }
    
    /// Number of dimensions
    public var ndim: Int {
        shape.count
    }
    
    /// Size in bytes
    public var byteSize: Int {
        count * dtype.size
    }
    
    /// Access underlying Metal buffer for GPU operations
    public var metalBuffer: MTLBuffer {
        _buffer.buffer
    }
    
    public var description: String {
        "Tensor(shape: \(shape), dtype: \(dtype))"
    }
    
    // MARK: - Initialization
    
    /// Create tensor with given shape (uninitialized)
    public init(shape: [Int], dtype: DType = .float32) throws {
        self.shape = shape
        self.dtype = dtype
        let size = shape.reduce(1, *) * dtype.size
        self._buffer = try MemoryManager.shared.allocate(size: size)
    }
    
    /// Create tensor from Float array
    public init(_ data: [Float], shape: [Int]? = nil) throws {
        let tensorShape = shape ?? [data.count]
        guard tensorShape.reduce(1, *) == data.count else {
            throw MemoryError.invalidShape
        }
        self.shape = tensorShape
        self.dtype = .float32
        self._buffer = try MemoryManager.shared.allocate(from: data)
    }
    
    /// Create tensor from Float16 array  
    public init(_ data: [Float16], shape: [Int]? = nil) throws {
        let tensorShape = shape ?? [data.count]
        guard tensorShape.reduce(1, *) == data.count else {
            throw MemoryError.invalidShape
        }
        self.shape = tensorShape
        self.dtype = .float16
        self._buffer = try MemoryManager.shared.allocate(from: data)
    }
    
    /// Internal initializer for sharing buffer (zero-copy)
    internal init(shape: [Int], dtype: DType, buffer: UnifiedBuffer) {
        self.shape = shape
        self.dtype = dtype
        self._buffer = buffer
    }
    
    /// Public initializer from an existing MTLBuffer (zero-copy)
    public convenience init(existingBuffer buffer: MTLBuffer, shape: [Int], dtype: DType) throws {
        let size = shape.reduce(1, *) * dtype.size
        let unifiedBuffer = UnifiedBuffer(device: GPUEngine.shared.device, buffer: buffer, size: size)
        self.init(shape: shape, dtype: dtype, buffer: unifiedBuffer)
    }
    
    deinit {
        MemoryManager.shared.release(_buffer)
    }
    
    // MARK: - Factory Methods
    
    /// Create tensor filled with zeros
    public static func zeros(_ shape: [Int], dtype: DType = .float32) throws -> Tensor {
        let tensor = try Tensor(shape: shape, dtype: dtype)
        memset(tensor.buffer.pointer, 0, tensor.byteSize)
        return tensor
    }
    
    /// Create tensor filled with ones
    public static func ones(_ shape: [Int], dtype: DType = .float32) throws -> Tensor {
        let tensor = try Tensor(shape: shape, dtype: dtype)
        let ptr = tensor.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
        var val: Float = 1.0
        vDSP_vfill(&val, ptr, 1, vDSP_Length(tensor.count))
        return tensor
    }
    
    /// Create tensor with random values
    public static func random(_ shape: [Int], dtype: DType = .float32) throws -> Tensor {
        let tensor = try Tensor(shape: shape, dtype: dtype)
        let ptr = tensor.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
        for i in 0..<tensor.count {
            ptr[i] = Float.random(in: -1...1)
        }
        return tensor
    }
    
    /// Create 2D identity matrix
    public static func eye(_ size: Int, dtype: DType = .float32) throws -> Tensor {
        let tensor = try zeros([size, size], dtype: dtype)
        let ptr = tensor.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
        for i in 0..<size {
            ptr[i * size + i] = 1.0
        }
        return tensor
    }
    
    // MARK: - Data Access
    
    /// Get data as Float array (CPU read)
    public func toArray() -> [Float] {
        return buffer.read(as: Float.self, count: count)
    }
    
    /// Get data as Float16 array
    public func toFloat16Array() -> [Float16] {
        return buffer.read(as: Float16.self, count: count)
    }
    
    /// Get single element
    public subscript(_ indices: Int...) -> Float {
        get {
            let flatIndex = flattenIndex(indices)
            let ptr = buffer.pointer.bindMemory(to: Float.self, capacity: count)
            return ptr[flatIndex]
        }
        set {
            let flatIndex = flattenIndex(indices)
            let ptr = buffer.pointer.bindMemory(to: Float.self, capacity: count)
            ptr[flatIndex] = newValue
        }
    }
    
    private func flattenIndex(_ indices: [Int]) -> Int {
        var index = 0
        var stride = 1
        for i in (0..<indices.count).reversed() {
            index += indices[i] * stride
            stride *= shape[i]
        }
        return index
    }
    
    // MARK: - Reshaping
    
    /// Reshape tensor (shares underlying buffer)
    public func reshape(_ newShape: [Int]) throws -> Tensor {
        let newCount = newShape.reduce(1, *)
        guard newCount == count else {
            throw MemoryError.invalidShape
        }
        // Create new tensor sharing the same buffer (true zero-copy)
        let tensor = Tensor(shape: newShape, dtype: dtype, buffer: _buffer)
        tensor.isParameter = isParameter
        tensor.isDirty = isDirty
        return tensor
    }
    
    /// Transpose 2D tensor
    public func transpose() throws -> Tensor {
        guard ndim == 2 else {
            throw MemoryError.invalidShape
        }
        if IntelligentRouter.shared.forcedBackend == .cpu {
            if isDirty {
                GPUEngine.shared.sync()
                isDirty = false
            }
            let result = try Tensor(shape: [shape[1], shape[0]], dtype: dtype)
            result.isParameter = isParameter
            let srcPtr = buffer.pointer.bindMemory(to: Float.self, capacity: count)
            let dstPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: count)
            vDSP_mtrans(srcPtr, 1, dstPtr, 1, vDSP_Length(shape[1]), vDSP_Length(shape[0]))
            return result
        } else {
            let result = try GPUEngine.shared.transpose(self)
            result.isParameter = isParameter
            return result
        }
    }
    
    /// Transpose a 2D tensor (static version)
    public static func transpose(_ a: Tensor) throws -> Tensor {
        return try a.transpose()
    }
}

// MARK: - CPU Operations (using Accelerate)

extension Tensor {
    
    /// Element-wise addition (CPU path)
    public static func add(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        guard a.shape == b.shape else {
            throw MemoryError.invalidShape
        }
        if a.isDirty || b.isDirty {
            GPUEngine.shared.sync()
            a.isDirty = false
            b.isDirty = false
        }
        let result = try Tensor(shape: a.shape, dtype: a.dtype)
        
        var count = Int32(a.count)
        let aPtr = a.buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b.buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
        let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        
        vDSP_vadd(aPtr, 1, bPtr, 1, rPtr, 1, vDSP_Length(a.count))
        
        return result
    }
    
    /// Element-wise multiplication (CPU path)
    public static func mul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        guard a.shape == b.shape else {
            throw MemoryError.invalidShape
        }
        if a.isDirty || b.isDirty {
            GPUEngine.shared.sync()
            a.isDirty = false
            b.isDirty = false
        }
        let result = try Tensor(shape: a.shape, dtype: a.dtype)
        
        let aPtr = a.buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b.buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
        let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        
        vDSP_vmul(aPtr, 1, bPtr, 1, rPtr, 1, vDSP_Length(a.count))
        
        return result
    }
    
    /// Matrix multiplication using Accelerate (CPU with AMX)
    public static func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        guard a.ndim == 2 && b.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        guard a.shape[1] == b.shape[0] else {
            throw MemoryError.invalidShape
        }
        if a.isDirty || b.isDirty {
            GPUEngine.shared.sync()
            a.isDirty = false
            b.isDirty = false
        }
        let M = a.shape[0]
        let K = a.shape[1]
        let N = b.shape[1]
        
        let result = try Tensor(shape: [M, N], dtype: a.dtype)
        
        let aPtr = a.buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b.buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
        let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        
        // Use BLAS for optimal AMX acceleration
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            Int32(M),
            Int32(N),
            Int32(K),
            1.0,
            aPtr, Int32(K),
            bPtr, Int32(N),
            0.0,
            rPtr, Int32(N)
        )
        
        return result
    }
    
    /// Split a 2D tensor along the second dimension into equal parts
    public func split(parts: Int) throws -> [Tensor] {
        guard ndim == 2 else { throw MemoryError.invalidShape }
        let N = shape[1]
        guard N % parts == 0 else { throw MemoryError.invalidShape }
        let partWidth = N / parts
        let M = shape[0]
        
        if IntelligentRouter.shared.forcedBackend == .cpu {
            if isDirty {
                GPUEngine.shared.sync()
                isDirty = false
            }
            
            var results: [Tensor] = []
            let srcPtr = buffer.pointer.bindMemory(to: Float.self, capacity: count)
            
            for p in 0..<parts {
                let result = try Tensor(shape: [M, partWidth], dtype: dtype)
                let dstPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
                for i in 0..<M {
                    let srcOffset = i * N + p * partWidth
                    let dstOffset = i * partWidth
                    memcpy(dstPtr.advanced(by: dstOffset), srcPtr.advanced(by: srcOffset), partWidth * 4)
                }
                results.append(result)
            }
            return results
        } else {
            if parts == 3 {
                return try GPUEngine.shared.split3Way2D(self)
            } else if parts == 2 {
                return try GPUEngine.shared.split2Way2D(self)
            }
            var results: [Tensor] = []
            for p in 0..<parts {
                let result = try GPUEngine.shared.split2D(self, parts: parts, partIdx: p)
                results.append(result)
            }
            return results
        }
    }
}
