// Benchmark LLaMA-3 Decoder Block using the zero-allocation ConductorEngine
import FusionML
import Foundation

func printFlush(_ items: Any...) {
    let output = items.map { "\($0)" }.joined(separator: " ")
    print(output)
    fflush(stdout)
}

func measure(_ label: String, warmup: Int = 1, iterations: Int = 5, _ op: () throws -> Void) rethrows -> Double {
    for _ in 0..<warmup { try op() }
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations { try op() }
    let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
    return ms
}

let BS = 4096
let H = 4096
let FFN = 14336

printFlush("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
printFlush("  FusionML Conductor: LLaMA-3 Decoder Block (Zero Allocation)")
printFlush("=" .padding(toLength: 75, withPad: "=", startingAt: 0))

// 1. Build the Conductor Computation Graph
let graph = Fusion.conductor.newGraph()

let x_in = graph.addInput(shape: [BS, H])
let ln1_gamma = graph.addInput(shape: [H])
let ln1_beta = graph.addInput(shape: [H])

let wq = graph.addInput(shape: [H, H])
let wk = graph.addInput(shape: [H, H])
let wv = graph.addInput(shape: [H, H])
let wo = graph.addInput(shape: [H, H])

let ln2_gamma = graph.addInput(shape: [H])
let ln2_beta = graph.addInput(shape: [H])

let wg = graph.addInput(shape: [H, FFN])
let wu = graph.addInput(shape: [H, FFN])
let wd = graph.addInput(shape: [FFN, H])

// Define graph DAG
let ln1 = graph.addNode(op: .layerNorm, inputs: [x_in, ln1_gamma, ln1_beta], outputShape: [BS, H])
let q = graph.addNode(op: .matmul, inputs: [ln1, wq], outputShape: [BS, H])
let k = graph.addNode(op: .matmul, inputs: [ln1, wk], outputShape: [BS, H])
let v = graph.addNode(op: .matmul, inputs: [ln1, wv], outputShape: [BS, H])

let attn_out = graph.addNode(op: .matmul, inputs: [q, wo], outputShape: [BS, H])
let attn_add = graph.addNode(op: .add, inputs: [x_in, attn_out], outputShape: [BS, H])

let ln2 = graph.addNode(op: .layerNorm, inputs: [attn_add, ln2_gamma, ln2_beta], outputShape: [BS, H])
let gate = graph.addNode(op: .matmul, inputs: [ln2, wg], outputShape: [BS, FFN])
let up = graph.addNode(op: .matmul, inputs: [ln2, wu], outputShape: [BS, FFN])

// Fused geluMul node
let gated = graph.addNode(op: .geluMul, inputs: [gate, up], outputShape: [BS, FFN])
let down = graph.addNode(op: .matmul, inputs: [gated, wd], outputShape: [BS, H])
let out = graph.addNode(op: .add, inputs: [attn_add, down], outputShape: [BS, H])

graph.markOutput(out)

printFlush("⚙️ Compiling Conductor FP32 Execution Plan...")
let plan32 = try! Fusion.conductor.compile(graph: graph, dtype: .float32)
printFlush("✅ FP32 Plan compiled! Wave Count: \(plan32.waves.count), Buffers: \(plan32.bufferPool.count)")

// 2. Initialize inputs/weights in FP32
let x_32 = try! Tensor.random([BS, H], dtype: .float32)
let ln1_gamma_32 = try! Tensor.ones([H], dtype: .float32)
let ln1_beta_32 = try! Tensor.zeros([H], dtype: .float32)
let wq_32 = try! Tensor.random([H, H], dtype: .float32)
let wk_32 = try! Tensor.random([H, H], dtype: .float32)
let wv_32 = try! Tensor.random([H, H], dtype: .float32)
let wo_32 = try! Tensor.random([H, H], dtype: .float32)
let ln2_gamma_32 = try! Tensor.ones([H], dtype: .float32)
let ln2_beta_32 = try! Tensor.zeros([H], dtype: .float32)
let wg_32 = try! Tensor.random([H, FFN], dtype: .float32)
let wu_32 = try! Tensor.random([H, FFN], dtype: .float32)
let wd_32 = try! Tensor.random([FFN, H], dtype: .float32)

let inputs_32 = [
    x_32, ln1_gamma_32, ln1_beta_32,
    wq_32, wk_32, wv_32, wo_32,
    ln2_gamma_32, ln2_beta_32,
    wg_32, wu_32, wd_32
]

printFlush("\n🔵 Running Conductor FP32 Decoder Block...")
let fp32_ms = try! measure("Conductor FP32") {
    let _ = try! Fusion.conductor.execute(plan: plan32, inputs: inputs_32)
}
let total_flops = Double(BS) * Double(H) * Double(H) * 2.0 * 4 + Double(BS) * Double(H) * Double(FFN) * 2.0 * 3
let fp32_gflops = (total_flops / fp32_ms) / 1_000_000
let M1_GPU_PEAK: Double = 2600.0
printFlush(String(format: "  Conductor FP32 Block Time: %.2f ms  %.0f GFLOPS (%.0f%% GPU util)", fp32_ms, fp32_gflops, fp32_gflops/M1_GPU_PEAK*100))

// 3. Convert weights/inputs to FP16
let x_16 = try! Tensor(shape: [BS, H], dtype: .float16)
let ln1_gamma_16 = try! Tensor(shape: [H], dtype: .float16)
let ln1_beta_16 = try! Tensor(shape: [H], dtype: .float16)
let wq_16 = try! Tensor(shape: [H, H], dtype: .float16)
let wk_16 = try! Tensor(shape: [H, H], dtype: .float16)
let wv_16 = try! Tensor(shape: [H, H], dtype: .float16)
let wo_16 = try! Tensor(shape: [H, H], dtype: .float16)
let ln2_gamma_16 = try! Tensor(shape: [H], dtype: .float16)
let ln2_beta_16 = try! Tensor(shape: [H], dtype: .float16)
let wg_16 = try! Tensor(shape: [H, FFN], dtype: .float16)
let wu_16 = try! Tensor(shape: [H, FFN], dtype: .float16)
let wd_16 = try! Tensor(shape: [FFN, H], dtype: .float16)

GPUEngine.convertFP32ToFP16(src: x_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: x_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: BS*H)
GPUEngine.convertFP32ToFP16(src: ln1_gamma_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: ln1_gamma_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H)
GPUEngine.convertFP32ToFP16(src: ln1_beta_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: ln1_beta_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H)
GPUEngine.convertFP32ToFP16(src: wq_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wq_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*H)
GPUEngine.convertFP32ToFP16(src: wk_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wk_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*H)
GPUEngine.convertFP32ToFP16(src: wv_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wv_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*H)
GPUEngine.convertFP32ToFP16(src: wo_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wo_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*H)
GPUEngine.convertFP32ToFP16(src: ln2_gamma_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: ln2_gamma_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H)
GPUEngine.convertFP32ToFP16(src: ln2_beta_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: ln2_beta_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H)
GPUEngine.convertFP32ToFP16(src: wg_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wg_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*FFN)
GPUEngine.convertFP32ToFP16(src: wu_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wu_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*FFN)
GPUEngine.convertFP32ToFP16(src: wd_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wd_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: FFN*H)

let inputs_16 = [
    x_16, ln1_gamma_16, ln1_beta_16,
    wq_16, wk_16, wv_16, wo_16,
    ln2_gamma_16, ln2_beta_16,
    wg_16, wu_16, wd_16
]

printFlush("\n⚙️ Compiling Conductor FP16 Execution Plan...")
let plan16 = try! Fusion.conductor.compile(graph: graph, dtype: .float16)
printFlush("✅ FP16 Plan compiled! Wave Count: \(plan16.waves.count), Buffers: \(plan16.bufferPool.count)")

printFlush("\n🟢 Running Conductor FP16 Decoder Block...")
let fp16_ms = try! measure("Conductor FP16") {
    let _ = try! Fusion.conductor.execute(plan: plan16, inputs: inputs_16)
}
let fp16_gflops = (total_flops / fp16_ms) / 1_000_000
let speedup = fp32_ms / fp16_ms
printFlush(String(format: "  Conductor FP16 Block Time: %.2f ms  %.0f GFLOPS (%.0f%% GPU util)  → %.2fx speedup", fp16_ms, fp16_gflops, fp16_gflops/M1_GPU_PEAK*100, speedup))

printFlush("\n" + "=" .padding(toLength: 75, withPad: "=", startingAt: 0))
printFlush("  MLX Reference (Best Measured): 1135.82 ms")
printFlush("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
