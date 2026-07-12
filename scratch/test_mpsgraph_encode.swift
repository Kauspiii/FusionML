import MetalPerformanceShadersGraph
import Metal

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let graph = MPSGraph()

let aTensor = graph.placeholder(shape: [2, 2], dataType: .float32, name: nil)
let bTensor = graph.placeholder(shape: [2, 2], dataType: .float32, name: nil)
let cTensor = graph.matrixMultiplication(primary: aTensor, secondary: bTensor, name: nil)

let cb = queue.makeCommandBuffer()!

let aBuffer = device.makeBuffer(length: 16, options: [])!
let bBuffer = device.makeBuffer(length: 16, options: [])!
let cBuffer = device.makeBuffer(length: 16, options: [])!

let aData = MPSGraphTensorData(mtlBuffer: aBuffer, shape: [2, 2], dataType: .float32, offset: 0, rowBytes: 8)
let bData = MPSGraphTensorData(mtlBuffer: bBuffer, shape: [2, 2], dataType: .float32, offset: 0, rowBytes: 8)
let cData = MPSGraphTensorData(mtlBuffer: cBuffer, shape: [2, 2], dataType: .float32, offset: 0, rowBytes: 8)

// Test execution options
let feeds = [aTensor: aData, bTensor: bData]

// Try to compile or find encode method signature
let executionDescriptor = MPSGraphExecutionDescriptor()
let mpsCB = MPSCommandBuffer(commandBuffer: cb)

// Let's run this method:
let _ = graph.encode(
    to: mpsCB,
    feeds: feeds,
    targetOperations: nil,
    resultsDictionary: [cTensor: cData],
    executionDescriptor: executionDescriptor
)
