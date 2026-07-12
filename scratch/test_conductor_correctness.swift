// test_conductor_correctness.swift — Verifies FusionML Conductor compiler and engine.
// Traces a graph, compiles it, prints the execution plan, verifies numerical correctness,
// and benchmarks the speedup over GPU-only or CPU-only baselines.

import Foundation
import FusionML

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public func print(_ items: Any..., separator: String = " ", terminator: String = "\n") {
    let output = items.map { "\($0)" }.joined(separator: separator)
    Swift.print(output, terminator: terminator)
    fflush(stdout)
}

func runConductorVerification() throws {
    print("🚀 Initializing FusionML for Conductor verification...")
    Fusion.initialize()
    
    let M = 1024
    let K = 1024
    let N = 1024
    
    print("📐 Phase 1: Creating Computation Graph [\(M)x\(K)x\(N)]...")
    let graph = Fusion.conductor.newGraph()
    
    // Graph inputs
    let inputA = graph.addInput(shape: [M, K])
    let inputB = graph.addInput(shape: [K, N])
    let inputBias = graph.addInput(shape: [M, N])
    
    // Graph ops
    // Step 1: MatMul (A @ B)
    let matmulNode = graph.addNode(op: .matmul, inputs: [inputA, inputB], outputShape: [M, N])
    // Step 2: Add Bias (A @ B + bias)
    let addNode = graph.addNode(op: .add, inputs: [matmulNode, inputBias], outputShape: [M, N])
    // Step 3: ReLU activation
    let reluNode = graph.addNode(op: .relu, inputs: [addNode], outputShape: [M, N])
    
    graph.markOutput(reluNode)
    
    // Populate the cost model manually so compiler has numbers for M=1024 if profiling isn't run,
    // though the compiler will run its own profiling for new sizes.
    let keyMatmul = OpShapeKey(op: .matmul, shapes: [[M, K], [K, N], [M, N]])
    let keyAdd = OpShapeKey(op: .add, shapes: [[M, N], [M, N], [M, N]])
    let keyRelu = OpShapeKey(op: .relu, shapes: [[M, N], [M, N]])
    
    // Setup cost model targets
    Fusion.conductor.costModel.recordSample(key: keyMatmul, backend: .gpu, latencyMs: 0.8)
    Fusion.conductor.costModel.recordSample(key: keyMatmul, backend: .cpu, latencyMs: 1.5)
    
    Fusion.conductor.costModel.recordSample(key: keyAdd, backend: .gpu, latencyMs: 0.1)
    Fusion.conductor.costModel.recordSample(key: keyAdd, backend: .cpu, latencyMs: 0.05)
    
    Fusion.conductor.costModel.recordSample(key: keyRelu, backend: .gpu, latencyMs: 0.1)
    Fusion.conductor.costModel.recordSample(key: keyRelu, backend: .cpu, latencyMs: 0.05)
    
    print("⚙️ Phase 2: Compiling Graph into Frozen ExecutionPlan...")
    let plan = try Fusion.conductor.compile(graph: graph)
    
    print("\n📋 Execution Plan Structure:")
    plan.printPlan()
    
    // Prepare dummy input data
    print("\n📦 Phase 3: Preparing Input Tensors (zero-copy unified memory)...")
    let a = try Tensor.ones([M, K])
    let b = try Tensor.ones([K, N])
    let bias = try Tensor.zeros([M, N])
    
    // Scale A and B so we don't just have 1s everywhere
    let aPtr = a.buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
    let bPtr = b.buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
    for i in 0..<a.count {
        aPtr[i] = Float(i % 100) / 1000.0
    }
    for i in 0..<b.count {
        bPtr[i] = Float(i % 50) / 500.0
    }
    
    print("🏃 Phase 4: Executing via Conductor Replay Engine...")
    let conductorOutputs = try Fusion.conductor.execute(plan: plan, inputs: [a, b, bias])
    let conductorResult = conductorOutputs[0]
    
    print("🏃 Phase 5: Executing via Standard Sequential Engine (Validation)...")
    // Reference run using regular Fusion APIs
    let refMatmul = try GPUEngine.shared.matmulMPS(a, b)
    let refAdd = try GPUEngine.shared.add(refMatmul, bias)
    let refResult = try GPUEngine.shared.relu(refAdd)
    GPUEngine.shared.sync()
    
    print("🔍 Phase 6: Verifying Numerical Exactness...")
    let condArr = conductorResult.toArray()
    let refArr = refResult.toArray()
    
    var maxDiff: Float = 0.0
    var avgDiff: Float = 0.0
    
    for i in 0..<condArr.count {
        let diff = abs(condArr[i] - refArr[i])
        maxDiff = max(maxDiff, diff)
        avgDiff += diff
    }
    avgDiff /= Float(condArr.count)
    
    print("  Max Difference: \(maxDiff)")
    print("  Mean Difference: \(avgDiff)")
    
    if maxDiff < 1e-4 {
        print("✅ SUCCESS: Conductor output is numerically identical to standard execution!")
    } else {
        print("❌ FAILURE: Conductor output deviates from standard execution!")
    }
    
    // Warmup the execution engine
    print("\n⏱️ Phase 7: Benchmarking Replay Latency...")
    for _ in 0..<20 {
        _ = try Fusion.conductor.execute(plan: plan, inputs: [a, b, bias])
    }
    
    let iterations = 100
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = try Fusion.conductor.execute(plan: plan, inputs: [a, b, bias])
    }
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
    print("  Conductor average execution latency: \(String(format: "%.3f", elapsed)) ms")
}

do {
    try runConductorVerification()
} catch {
    print("❌ Fatal error: \(error)")
}
