// Overhead Isolator — Find where FusionML's 10% GPU efficiency gap vs MLX comes from
//
// We compare:
//   A. Raw MPS calls with pre-allocated buffers (minimal overhead)
//   B. Our Tensor API path (matmulMPS through full stack)
//   C. If A ≈ B, the gap is MPS itself
//   D. If A >> B, the gap is our allocation/dispatch overhead

import FusionML
import Foundation
import Metal
import MetalPerformanceShaders

func measure(_ label: String, warmup: Int = 5, iterations: Int = 15, _ op: () throws -> Void) rethrows -> Double {
    for _ in 0..<warmup { try op() }
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations { try op() }
    let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
    return ms
}

print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
print("  Overhead Isolator — Where's the 10% GPU Efficiency Gap?")
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))

let device = MemoryManager.shared.device
let queue = MemoryManager.shared.commandQueue
let M1_GPU_PEAK: Double = 2600

// ─── TEST: 7 matmuls = LLaMA decoder block matmul portion ────────
// 4× [4096,4096]×[4096,4096] (QKV+O)
// 2× [4096,4096]×[4096,14336] (gate+up)
// 1× [4096,14336]×[14336,4096] (down)

let BS = 4096
let H = 4096
let FFN = 14336

// Total FLOPs in the 7 matmuls
let total_flops = 4.0 * 2 * Double(BS*H*H)  // QKV+O
    + 2.0 * 2 * Double(BS*H*FFN)             // gate+up
    + 1.0 * 2 * Double(BS*FFN*H)             // down

print(String(format: "\n  Total matmul FLOPs: %.1f TFLOP", total_flops / 1e12))
print(String(format: "  Theory min at 100%%: %.1f ms", total_flops / (M1_GPU_PEAK * 1e9) * 1000))

// ─── PATH A: Raw MPS with pre-allocated buffers ──────────────────
print("\n🔴 Path A: Raw MPS — Pre-allocated buffers, zero overhead")
print("-" .padding(toLength: 75, withPad: "-", startingAt: 0))

// Pre-allocate ALL buffers once
let buf_x   = device.makeBuffer(length: BS * H * 4, options: .storageModeShared)!
let buf_wq  = device.makeBuffer(length: H * H * 4, options: .storageModeShared)!
let buf_wk  = device.makeBuffer(length: H * H * 4, options: .storageModeShared)!
let buf_wv  = device.makeBuffer(length: H * H * 4, options: .storageModeShared)!
let buf_wo  = device.makeBuffer(length: H * H * 4, options: .storageModeShared)!
let buf_wg  = device.makeBuffer(length: H * FFN * 4, options: .storageModeShared)!
let buf_wu  = device.makeBuffer(length: H * FFN * 4, options: .storageModeShared)!
let buf_wd  = device.makeBuffer(length: FFN * H * 4, options: .storageModeShared)!
let buf_q   = device.makeBuffer(length: BS * H * 4, options: .storageModeShared)!
let buf_k   = device.makeBuffer(length: BS * H * 4, options: .storageModeShared)!
let buf_v   = device.makeBuffer(length: BS * H * 4, options: .storageModeShared)!
let buf_o   = device.makeBuffer(length: BS * H * 4, options: .storageModeShared)!
let buf_gate = device.makeBuffer(length: BS * FFN * 4, options: .storageModeShared)!
let buf_up  = device.makeBuffer(length: BS * FFN * 4, options: .storageModeShared)!
let buf_down = device.makeBuffer(length: BS * H * 4, options: .storageModeShared)!

// Pre-create MPS kernels (cached, reused across iterations)
let mps_hh = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                      resultRows: BS, resultColumns: H, interiorColumns: H, alpha: 1.0, beta: 0.0)
let mps_hf = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                      resultRows: BS, resultColumns: FFN, interiorColumns: H, alpha: 1.0, beta: 0.0)
let mps_fh = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                      resultRows: BS, resultColumns: H, interiorColumns: FFN, alpha: 1.0, beta: 0.0)

// Pre-create descriptors
let desc_x = MPSMatrixDescriptor(rows: BS, columns: H, rowBytes: H * 4, dataType: .float32)
let desc_wh = MPSMatrixDescriptor(rows: H, columns: H, rowBytes: H * 4, dataType: .float32)
let desc_wf = MPSMatrixDescriptor(rows: H, columns: FFN, rowBytes: FFN * 4, dataType: .float32)
let desc_wd = MPSMatrixDescriptor(rows: FFN, columns: H, rowBytes: H * 4, dataType: .float32)
let desc_rh = MPSMatrixDescriptor(rows: BS, columns: H, rowBytes: H * 4, dataType: .float32)
let desc_rf = MPSMatrixDescriptor(rows: BS, columns: FFN, rowBytes: FFN * 4, dataType: .float32)

// Pre-create matrix wrappers
let mat_x  = MPSMatrix(buffer: buf_x, descriptor: desc_x)
let mat_wq = MPSMatrix(buffer: buf_wq, descriptor: desc_wh)
let mat_wk = MPSMatrix(buffer: buf_wk, descriptor: desc_wh)
let mat_wv = MPSMatrix(buffer: buf_wv, descriptor: desc_wh)
let mat_wo = MPSMatrix(buffer: buf_wo, descriptor: desc_wh)
let mat_wg = MPSMatrix(buffer: buf_wg, descriptor: desc_wf)
let mat_wu = MPSMatrix(buffer: buf_wu, descriptor: desc_wf)
let mat_wd = MPSMatrix(buffer: buf_wd, descriptor: desc_wd)
let mat_q  = MPSMatrix(buffer: buf_q, descriptor: desc_rh)
let mat_k  = MPSMatrix(buffer: buf_k, descriptor: desc_rh)
let mat_v  = MPSMatrix(buffer: buf_v, descriptor: desc_rh)
let mat_o  = MPSMatrix(buffer: buf_o, descriptor: desc_rh)
let mat_gate = MPSMatrix(buffer: buf_gate, descriptor: desc_rf)
let mat_up = MPSMatrix(buffer: buf_up, descriptor: desc_rf)
let mat_down = MPSMatrix(buffer: buf_down, descriptor: desc_rh)

let rawA_ms = try! measure("Raw MPS") {
    let cb = queue.makeCommandBuffer()!
    
    // 4× H×H matmuls: Q, K, V, O
    mps_hh.encode(commandBuffer: cb, leftMatrix: mat_x, rightMatrix: mat_wq, resultMatrix: mat_q)
    mps_hh.encode(commandBuffer: cb, leftMatrix: mat_x, rightMatrix: mat_wk, resultMatrix: mat_k)
    mps_hh.encode(commandBuffer: cb, leftMatrix: mat_x, rightMatrix: mat_wv, resultMatrix: mat_v)
    mps_hh.encode(commandBuffer: cb, leftMatrix: mat_q, rightMatrix: mat_wo, resultMatrix: mat_o)
    
    // 2× H×FFN matmuls: gate, up
    mps_hf.encode(commandBuffer: cb, leftMatrix: mat_x, rightMatrix: mat_wg, resultMatrix: mat_gate)
    mps_hf.encode(commandBuffer: cb, leftMatrix: mat_x, rightMatrix: mat_wu, resultMatrix: mat_up)
    
    // 1× FFN×H matmul: down
    mps_fh.encode(commandBuffer: cb, leftMatrix: mat_gate, rightMatrix: mat_wd, resultMatrix: mat_down)
    
    cb.commit()
    cb.waitUntilCompleted()
}

let rawA_gflops = (total_flops / rawA_ms) / 1_000_000
print(String(format: "  Raw MPS (7 matmuls, 1 CB): %.2f ms  %.0f GFLOPS  (%.0f%% util)", rawA_ms, rawA_gflops, rawA_gflops/M1_GPU_PEAK*100))

// ─── PATH B: FusionML Tensor API (batched) ───────────────────────
print("\n🔵 Path B: FusionML Tensor API — Full stack, batched")
print("-" .padding(toLength: 75, withPad: "-", startingAt: 0))

let t_x  = try! Tensor.random([BS, H])
let t_wq = try! Tensor.random([H, H])
let t_wk = try! Tensor.random([H, H])
let t_wv = try! Tensor.random([H, H])
let t_wo = try! Tensor.random([H, H])
let t_wg = try! Tensor.random([H, FFN])
let t_wu = try! Tensor.random([H, FFN])
let t_wd = try! Tensor.random([FFN, H])

let tensorB_ms = try! measure("Tensor API batched") {
    GPUEngine.shared.startBatch()
    let q = try GPUEngine.shared.matmulMPS(t_x, t_wq)
    let k = try GPUEngine.shared.matmulMPS(t_x, t_wk)
    let v = try GPUEngine.shared.matmulMPS(t_x, t_wv)
    let o = try GPUEngine.shared.matmulMPS(q, t_wo)
    let gate = try GPUEngine.shared.matmulMPS(t_x, t_wg)
    let up = try GPUEngine.shared.matmulMPS(t_x, t_wu)
    let down = try GPUEngine.shared.matmulMPS(gate, t_wd)
    GPUEngine.shared.commitBatch()
    GPUEngine.shared.sync()
}

let tensorB_gflops = (total_flops / tensorB_ms) / 1_000_000
print(String(format: "  Tensor API (7 matmuls, batched): %.2f ms  %.0f GFLOPS  (%.0f%% util)", tensorB_ms, tensorB_gflops, tensorB_gflops/M1_GPU_PEAK*100))

// ─── PATH C: FusionML Tensor API (unbatched — per-op sync) ──────
print("\n⚪ Path C: FusionML Tensor API — Full stack, per-op sync")
print("-" .padding(toLength: 75, withPad: "-", startingAt: 0))

let tensorC_ms = try! measure("Tensor API unbatched") {
    let q = try GPUEngine.shared.matmulMPS(t_x, t_wq)
    GPUEngine.shared.sync()
    let k = try GPUEngine.shared.matmulMPS(t_x, t_wk)
    GPUEngine.shared.sync()
    let v = try GPUEngine.shared.matmulMPS(t_x, t_wv)
    GPUEngine.shared.sync()
    let o = try GPUEngine.shared.matmulMPS(q, t_wo)
    GPUEngine.shared.sync()
    let gate = try GPUEngine.shared.matmulMPS(t_x, t_wg)
    GPUEngine.shared.sync()
    let up = try GPUEngine.shared.matmulMPS(t_x, t_wu)
    GPUEngine.shared.sync()
    let down = try GPUEngine.shared.matmulMPS(gate, t_wd)
    GPUEngine.shared.sync()
}

let tensorC_gflops = (total_flops / tensorC_ms) / 1_000_000
print(String(format: "  Tensor API (7 matmuls, 7 syncs): %.2f ms  %.0f GFLOPS  (%.0f%% util)", tensorC_ms, tensorC_gflops, tensorC_gflops/M1_GPU_PEAK*100))

// ─── PATH D: Raw MPS with pre-allocated, with result reuse ──────
print("\n🟢 Path D: Raw MPS — Pre-alloc + reuse result buffers")
print("-" .padding(toLength: 75, withPad: "-", startingAt: 0))

// Same as A but reuse Q buffer for O result (since Q isn't needed after O)
let rawD_ms = try! measure("Raw MPS reuse") {
    let cb = queue.makeCommandBuffer()!
    
    mps_hh.encode(commandBuffer: cb, leftMatrix: mat_x, rightMatrix: mat_wq, resultMatrix: mat_q)
    mps_hh.encode(commandBuffer: cb, leftMatrix: mat_x, rightMatrix: mat_wk, resultMatrix: mat_k)
    mps_hh.encode(commandBuffer: cb, leftMatrix: mat_x, rightMatrix: mat_wv, resultMatrix: mat_v)
    mps_hh.encode(commandBuffer: cb, leftMatrix: mat_q, rightMatrix: mat_wo, resultMatrix: mat_o)
    mps_hf.encode(commandBuffer: cb, leftMatrix: mat_o, rightMatrix: mat_wg, resultMatrix: mat_gate)  // reuse o as input
    mps_hf.encode(commandBuffer: cb, leftMatrix: mat_o, rightMatrix: mat_wu, resultMatrix: mat_up)
    mps_fh.encode(commandBuffer: cb, leftMatrix: mat_gate, rightMatrix: mat_wd, resultMatrix: mat_down)
    
    cb.commit()
    cb.waitUntilCompleted()
}

let rawD_gflops = (total_flops / rawD_ms) / 1_000_000
print(String(format: "  Raw MPS reuse (7 matmuls, 1 CB): %.2f ms  %.0f GFLOPS  (%.0f%% util)", rawD_ms, rawD_gflops, rawD_gflops/M1_GPU_PEAK*100))

// ─── ANALYSIS ────────────────────────────────────────────────────
print("\n" + "=" .padding(toLength: 75, withPad: "=", startingAt: 0))
print("  OVERHEAD ANALYSIS")
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))

let tensor_overhead = tensorB_ms - rawA_ms
let sync_overhead = tensorC_ms - tensorB_ms
let alloc_overhead = tensorB_ms - rawD_ms

print(String(format: "  Raw MPS (baseline):          %8.2f ms  %6.0f GFLOPS", rawA_ms, rawA_gflops))
print(String(format: "  Tensor API overhead:         %+8.2f ms  (alloc + descriptor + lock)", tensor_overhead))
print(String(format: "  Per-op sync overhead:        %+8.2f ms  (7 extra syncs)", sync_overhead))
print(String(format: "  Total Tensor API (batched):  %8.2f ms  %6.0f GFLOPS", tensorB_ms, tensorB_gflops))
print(String(format: "  Total Tensor API (synced):   %8.2f ms  %6.0f GFLOPS", tensorC_ms, tensorC_gflops))
print("")
print(String(format: "  MLX reference:               %8.2f ms  %6.0f GFLOPS", 1180.84, 1688.0))
print("")
if rawA_ms < 1181 {
    print("  ✅ Raw MPS BEATS MLX → Our overhead is in the Tensor API, not MPS!")
    print(String(format: "     Fix needed: reduce %.0fms of Tensor API overhead", tensor_overhead))
} else {
    print("  ❌ Raw MPS is SLOWER than MLX → MPS itself is the bottleneck!")
    print("     Fix needed: write custom Metal GEMM kernel or optimize MPS usage")
}
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
