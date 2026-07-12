// Benchmark LLaMA-3 Decoder Block in FP32 vs FP16
import FusionML
import Foundation

func measure(_ label: String, warmup: Int = 3, iterations: Int = 10, _ op: () throws -> Void) rethrows -> Double {
    for _ in 0..<warmup { try op() }
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations { try op() }
    let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
    return ms
}

print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
print("  FusionML LLaMA-3 Decoder Block: FP32 vs FP16")
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))

let BS = 4096
let H = 4096
let FFN = 14336

let total_flops = Double(BS) * Double(H) * Double(H) * 2.0 * 4 + Double(BS) * Double(H) * Double(FFN) * 2.0 * 3
let M1_GPU_PEAK: Double = 2600.0

// ─── FP32 DECODER BLOCK ──────────────────────────────────────────
print("\n🔵 Running FP32 Decoder Block...")
let wq_32 = try! Tensor.random([H, H], dtype: .float32)
let wk_32 = try! Tensor.random([H, H], dtype: .float32)
let wv_32 = try! Tensor.random([H, H], dtype: .float32)
let wo_32 = try! Tensor.random([H, H], dtype: .float32)
let wg_32 = try! Tensor.random([H, FFN], dtype: .float32)
let wu_32 = try! Tensor.random([H, FFN], dtype: .float32)
let wd_32 = try! Tensor.random([FFN, H], dtype: .float32)
let ln_gamma_32 = try! Tensor.ones([H], dtype: .float32)
let ln_beta_32 = try! Tensor.zeros([H], dtype: .float32)
let x_32 = try! Tensor.random([BS, H], dtype: .float32)

let fp32_ms = try! measure("FP32") {
    GPUEngine.shared.startBatch()
    let ln1 = try GPUEngine.shared.layerNorm(x_32, gamma: ln_gamma_32, beta: ln_beta_32)
    let q = try GPUEngine.shared.matmulMPS(ln1, wq_32)
    let k = try GPUEngine.shared.matmulMPS(ln1, wk_32)
    let v = try GPUEngine.shared.matmulMPS(ln1, wv_32)
    let attn_out = try GPUEngine.shared.matmulMPS(q, wo_32)
    let attn_add = try GPUEngine.shared.add(x_32, attn_out)
    let ln2 = try GPUEngine.shared.layerNorm(attn_add, gamma: ln_gamma_32, beta: ln_beta_32)
    let gate = try GPUEngine.shared.matmulMPS(ln2, wg_32)
    let up = try GPUEngine.shared.matmulMPS(ln2, wu_32)
    let gated = try GPUEngine.shared.geluMul(gate, up)
    let down = try GPUEngine.shared.matmulMPS(gated, wd_32)
    let _ = try GPUEngine.shared.add(attn_add, down)
    GPUEngine.shared.commitBatch()
    GPUEngine.shared.sync()
}
let fp32_gflops = (total_flops / fp32_ms) / 1_000_000
print(String(format: "  FP32 Block Time: %.2f ms  %.0f GFLOPS (%.0f%% GPU util)", fp32_ms, fp32_gflops, fp32_gflops/M1_GPU_PEAK*100))

// ─── FP16 DECODER BLOCK ──────────────────────────────────────────
print("\n🟢 Running FP16 Decoder Block...")
let wq_16 = try! Tensor(shape: [H, H], dtype: .float16)
let wk_16 = try! Tensor(shape: [H, H], dtype: .float16)
let wv_16 = try! Tensor(shape: [H, H], dtype: .float16)
let wo_16 = try! Tensor(shape: [H, H], dtype: .float16)
let wg_16 = try! Tensor(shape: [H, FFN], dtype: .float16)
let wu_16 = try! Tensor(shape: [H, FFN], dtype: .float16)
let wd_16 = try! Tensor(shape: [FFN, H], dtype: .float16)
let ln_gamma_16 = try! Tensor(shape: [H], dtype: .float16)
let ln_beta_16 = try! Tensor(shape: [H], dtype: .float16)
let x_16 = try! Tensor(shape: [BS, H], dtype: .float16)

// Initialize data using SIMD helper
GPUEngine.convertFP32ToFP16(src: wq_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wq_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*H)
GPUEngine.convertFP32ToFP16(src: wk_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wk_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*H)
GPUEngine.convertFP32ToFP16(src: wv_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wv_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*H)
GPUEngine.convertFP32ToFP16(src: wo_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wo_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*H)
GPUEngine.convertFP32ToFP16(src: wg_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wg_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*FFN)
GPUEngine.convertFP32ToFP16(src: wu_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wu_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H*FFN)
GPUEngine.convertFP32ToFP16(src: wd_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: wd_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: FFN*H)
GPUEngine.convertFP32ToFP16(src: ln_gamma_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: ln_gamma_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H)
GPUEngine.convertFP32ToFP16(src: ln_beta_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: ln_beta_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: H)
GPUEngine.convertFP32ToFP16(src: x_32.buffer.pointer.assumingMemoryBound(to: Float.self), dst: x_16._buffer.pointer.assumingMemoryBound(to: Float16.self), count: BS*H)

let fp16_ms = try! measure("FP16") {
    GPUEngine.shared.startBatch()
    let ln1 = try GPUEngine.shared.layerNorm(x_16, gamma: ln_gamma_16, beta: ln_beta_16)
    let q = try GPUEngine.shared.matmulMPS(ln1, wq_16)
    let k = try GPUEngine.shared.matmulMPS(ln1, wk_16)
    let v = try GPUEngine.shared.matmulMPS(ln1, wv_16)
    let attn_out = try GPUEngine.shared.matmulMPS(q, wo_16)
    let attn_add = try GPUEngine.shared.add(x_16, attn_out)
    let ln2 = try GPUEngine.shared.layerNorm(attn_add, gamma: ln_gamma_16, beta: ln_beta_16)
    let gate = try GPUEngine.shared.matmulMPS(ln2, wg_16)
    let up = try GPUEngine.shared.matmulMPS(ln2, wu_16)
    let gated = try GPUEngine.shared.geluMul(gate, up)
    let down = try GPUEngine.shared.matmulMPS(gated, wd_16)
    let _ = try GPUEngine.shared.add(attn_add, down)
    GPUEngine.shared.commitBatch()
    GPUEngine.shared.sync()
}
let fp16_gflops = (total_flops / fp16_ms) / 1_000_000
let speedup = fp32_ms / fp16_ms
print(String(format: "  FP16 Block Time: %.2f ms  %.0f GFLOPS (%.0f%% GPU util)  → %.2fx speedup", fp16_ms, fp16_gflops, fp16_gflops/M1_GPU_PEAK*100, speedup))

print("\n" + "=" .padding(toLength: 75, withPad: "=", startingAt: 0))
print("  MLX Reference: 1180.84 ms")
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
