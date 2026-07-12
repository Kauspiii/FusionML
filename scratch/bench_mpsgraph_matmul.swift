import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

let M = 4096
let N = 4096
let K = 14336

// Allocate buffers
let byteSize = M * K * 4 // Float32 size (safe for both)
let aBuffer = device.makeBuffer(length: byteSize, options: .storageModeShared)!
let bBuffer = device.makeBuffer(length: byteSize, options: .storageModeShared)!
let cBuffer = device.makeBuffer(length: byteSize, options: .storageModeShared)!

for dtype in [MPSDataType.float32, MPSDataType.float16] {
    let mpsDataType = dtype
    let graphDataType = dtype
    let elementSize = dtype == .float16 ? 2 : 4
    let dtypeLabel = dtype == .float16 ? "FP16" : "FP32"
    
    print("\n------------------------------")
    print("Testing \(dtypeLabel)")
    print("------------------------------")
    
    // 1. MPS Matrix Multiplication
    func runMPSMatrixMultiplication() {
        let aDesc = MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K * elementSize, dataType: mpsDataType)
        let bDesc = MPSMatrixDescriptor(rows: K, columns: N, rowBytes: N * elementSize, dataType: mpsDataType)
        let cDesc = MPSMatrixDescriptor(rows: M, columns: N, rowBytes: N * elementSize, dataType: mpsDataType)
        
        let aMatrix = MPSMatrix(buffer: aBuffer, offset: 0, descriptor: aDesc)
        let bMatrix = MPSMatrix(buffer: bBuffer, offset: 0, descriptor: bDesc)
        let cMatrix = MPSMatrix(buffer: cBuffer, offset: 0, descriptor: cDesc)
        
        let matmul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: M,
            resultColumns: N,
            interiorColumns: K,
            alpha: 1.0,
            beta: 0.0
        )
        
        // Warmup
        for _ in 0..<3 {
            let cb = commandQueue.makeCommandBuffer()!
            matmul.encode(commandBuffer: cb, leftMatrix: aMatrix, rightMatrix: bMatrix, resultMatrix: cMatrix)
            cb.commit()
            cb.waitUntilCompleted()
        }
        
        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10
        for _ in 0..<iterations {
            let cb = commandQueue.makeCommandBuffer()!
            matmul.encode(commandBuffer: cb, leftMatrix: aMatrix, rightMatrix: bMatrix, resultMatrix: cMatrix)
            cb.commit()
            cb.waitUntilCompleted()
        }
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
        print(String(format: "MPSMatrixMultiplication (\(dtypeLabel)): %.2f ms", ms))
    }

    // 2. MPSGraph Matrix Multiplication
    func runMPSGraphMultiplication() {
        let graph = MPSGraph()
        
        let aTensor = graph.placeholder(shape: [M, K] as [NSNumber], dataType: graphDataType, name: "A")
        let bTensor = graph.placeholder(shape: [K, N] as [NSNumber], dataType: graphDataType, name: "B")
        let cTensor = graph.matrixMultiplication(primary: aTensor, secondary: bTensor, name: "C")
        
        let aData = MPSGraphTensorData(aBuffer, shape: [M, K] as [NSNumber], dataType: graphDataType)
        let bData = MPSGraphTensorData(bBuffer, shape: [K, N] as [NSNumber], dataType: graphDataType)
        let cData = MPSGraphTensorData(cBuffer, shape: [M, N] as [NSNumber], dataType: graphDataType)
        
        // Warmup
        for _ in 0..<3 {
            let _ = graph.run(
                with: commandQueue,
                feeds: [aTensor: aData, bTensor: bData],
                targetTensors: [cTensor],
                targetOperations: nil
            )
        }
        
        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10
        for _ in 0..<iterations {
            let _ = graph.run(
                with: commandQueue,
                feeds: [aTensor: aData, bTensor: bData],
                targetTensors: [cTensor],
                targetOperations: nil
            )
        }
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
        print(String(format: "MPSGraph MatrixMultiplication (\(dtypeLabel)): %.2f ms", ms))
    }

    runMPSMatrixMultiplication()
    runMPSGraphMultiplication()
}
exit(0)

