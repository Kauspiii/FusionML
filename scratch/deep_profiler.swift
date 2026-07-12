// Deep Profiler — Find EXACTLY Where Time Goes
// Tests real LLaMA-3 shapes with per-operation timing

import FusionML
import Foundation
import Accelerate

func measure(_ label: String, warmup: Int = 5, iterations: Int = 20, _ op: () throws -> Void) rethrows -> Double {
    for _ in 0..<warmup { try op() }
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations { try op() }
    let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
    return ms
}

print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
print("  FusionML Deep Profiler — Where Does Every Millisecond Go?")
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))

// ─── GPU UTILIZATION CHECK ───────────────────────────────────────────
print("\n🔬 GPU Utilization vs Theoretical Peak")
print("-" .padding(toLength: 75, withPad: "-", startingAt: 0))

let M1_FP32_PEAK: Double = 2600  // GFLOPS
let M1_CPU_PEAK: Double  = 2000  // GFLOPS (AMX)
let M1_TOTAL_PEAK: Double = 4600 // GFLOPS (GPU + CPU)

for size in [512, 1024, 2048, 4096] {
    let a = try! Tensor.random([size, size])
    let b = try! Tensor.random([size, size])
    let flops = 2.0 * Double(size) * Double(size) * Double(size)
    
    // GPU only (MPS)
    let gpu_ms = try! measure("GPU \(size)") {
        let _ = try GPUEngine.shared.matmulMPS(a, b)
        GPUEngine.shared.sync()
    }
    let gpu_gflops = (flops / gpu_ms) / 1_000_000
    let gpu_util = (gpu_gflops / M1_FP32_PEAK) * 100
    
    // CPU only (AMX via BLAS)
    let cpu_ms = try! measure("CPU \(size)") {
        let _ = try Tensor.matmul(a, b)
    }
    let cpu_gflops = (flops / cpu_ms) / 1_000_000
    let cpu_util = (cpu_gflops / M1_CPU_PEAK) * 100
    
    // GPU+CPU split (batched properly)
    let split_ms: Double
    if size >= 512 {
        // Manual split: GPU does 60% rows, CPU does 40% rows concurrently
        let gpuRows = Int(Double(size) * 0.57)
        let cpuRows = size - gpuRows
        
        let aPtr = a.buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b.buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
        
        split_ms = try! measure("Split \(size)") {
            let result = try Tensor(shape: [size, size], dtype: .float32)
            let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
            
            // GPU portion (async) — uses buffer offsets for zero-copy
            try GPUEngine.shared.matmulRawMPS(
                a: a.metalBuffer, b: b.metalBuffer, result: result.metalBuffer,
                M: gpuRows, N: size, K: size,
                aOffset: cpuRows * size * 4,
                resultOffset: cpuRows * size * 4,
                waitUntilCompleted: false
            )
            
            // CPU portion (sync) — runs concurrently with GPU
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       Int32(cpuRows), Int32(size), Int32(size), 1.0,
                       aPtr, Int32(size), bPtr, Int32(size), 0.0, rPtr, Int32(size))
            
            // Wait for GPU
            GPUEngine.shared.sync()
        }
    } else {
        split_ms = gpu_ms
    }
    let split_gflops = (flops / split_ms) / 1_000_000
    let split_util = (split_gflops / M1_TOTAL_PEAK) * 100
    let speedup_vs_gpu = gpu_ms / split_ms
    
    // Theoretical minimum
    let theory_gpu_ms = flops / (M1_FP32_PEAK * 1_000_000_000) * 1000
    let theory_split_ms = flops / (M1_TOTAL_PEAK * 1_000_000_000) * 1000
    
    print(String(format: "  %4d×%4d:", size, size))
    print(String(format: "    GPU-only:  %7.2f ms  %6.0f GFLOPS  %4.0f%% utilization  (theory min: %.2f ms)", gpu_ms, gpu_gflops, gpu_util, theory_gpu_ms))
    print(String(format: "    CPU-only:  %7.2f ms  %6.0f GFLOPS  %4.0f%% utilization", cpu_ms, cpu_gflops, cpu_util))
    print(String(format: "    GPU+CPU:   %7.2f ms  %6.0f GFLOPS  %4.0f%% utilization  (theory min: %.2f ms)  → %.2fx vs GPU", split_ms, split_gflops, split_util, theory_split_ms, speedup_vs_gpu))
}

// ─── COMMAND BUFFER OVERHEAD MEASUREMENT ─────────────────────────────
print("\n🔬 Command Buffer Overhead Measurement")
print("-" .padding(toLength: 75, withPad: "-", startingAt: 0))

let cbTest = try! Tensor.random([1024, 1024])
let cbTest2 = try! Tensor.random([1024, 1024])

// Single matmul, sync immediately
let single_sync_ms = try! measure("1 matmul + sync") {
    let _ = try GPUEngine.shared.matmulMPS(cbTest, cbTest2)
    GPUEngine.shared.sync()
}

// 10 matmuls, sync after each
let ten_sync_ms = try! measure("10 matmul + 10 syncs", iterations: 10) {
    for _ in 0..<10 {
        let _ = try GPUEngine.shared.matmulMPS(cbTest, cbTest2)
        GPUEngine.shared.sync()
    }
}

// 10 matmuls batched, 1 sync
let ten_batch_ms = try! measure("10 matmul + 1 sync (batched)", iterations: 10) {
    GPUEngine.shared.startBatch()
    for _ in 0..<10 {
        let _ = try GPUEngine.shared.matmulMPS(cbTest, cbTest2)
    }
    GPUEngine.shared.commitBatch()
    GPUEngine.shared.sync()
}

let overhead_per_sync = (ten_sync_ms - ten_batch_ms) / 10
print(String(format: "  Single matmul + sync:     %.2f ms", single_sync_ms))
print(String(format: "  10 matmuls + 10 syncs:    %.2f ms (%.2f ms/op)", ten_sync_ms, ten_sync_ms/10))
print(String(format: "  10 matmuls + 1 sync:      %.2f ms (%.2f ms/op)", ten_batch_ms, ten_batch_ms/10))
print(String(format: "  Sync overhead per call:   %.3f ms", overhead_per_sync))

// ─── LLAMA-3 SHAPES PROFILING ────────────────────────────────────────
print("\n🔬 LLaMA-3 8B Real Shapes (B=4, S=1024, H=4096, FFN=14336)")
print("-" .padding(toLength: 75, withPad: "-", startingAt: 0))

let B = 4
let S = 1024
let H = 4096
let FFN = 14336
let BS = B * S  // 4096

// QKV projection: [4096, 4096] × [4096, 4096]
let qkv_x = try! Tensor.random([BS, H])
let qkv_w = try! Tensor.random([H, H])

let qkv_ms = try! measure("QKV Projection") {
    let _ = try GPUEngine.shared.matmulMPS(qkv_x, qkv_w)
    GPUEngine.shared.sync()
}
let qkv_flops = 2.0 * Double(BS) * Double(H) * Double(H)
let qkv_gflops = (qkv_flops / qkv_ms) / 1_000_000
print(String(format: "  QKV [%d,%d]×[%d,%d]:     %7.2f ms  %.0f GFLOPS  (%.0f%% peak)", BS, H, H, H, qkv_ms, qkv_gflops, qkv_gflops/M1_FP32_PEAK*100))

// MLP gate_up: [4096, 4096] × [4096, 14336]
let mlp_w1 = try! Tensor.random([H, FFN])
let mlp_up_ms = try! measure("MLP gate_up") {
    let _ = try GPUEngine.shared.matmulMPS(qkv_x, mlp_w1)
    GPUEngine.shared.sync()
}
let mlp_up_flops = 2.0 * Double(BS) * Double(H) * Double(FFN)
let mlp_up_gflops = (mlp_up_flops / mlp_up_ms) / 1_000_000
print(String(format: "  MLP up [%d,%d]×[%d,%d]: %7.2f ms  %.0f GFLOPS  (%.0f%% peak)", BS, H, H, FFN, mlp_up_ms, mlp_up_gflops, mlp_up_gflops/M1_FP32_PEAK*100))

// MLP down: [4096, 14336] × [14336, 4096]
let mlp_h = try! Tensor.random([BS, FFN])
let mlp_w2 = try! Tensor.random([FFN, H])
let mlp_down_ms = try! measure("MLP down") {
    let _ = try GPUEngine.shared.matmulMPS(mlp_h, mlp_w2)
    GPUEngine.shared.sync()
}
let mlp_down_flops = 2.0 * Double(BS) * Double(FFN) * Double(H)
let mlp_down_gflops = (mlp_down_flops / mlp_down_ms) / 1_000_000
print(String(format: "  MLP dn [%d,%d]×[%d,%d]: %7.2f ms  %.0f GFLOPS  (%.0f%% peak)", BS, FFN, FFN, H, mlp_down_ms, mlp_down_gflops, mlp_down_gflops/M1_FP32_PEAK*100))

// ─── GPU+CPU SPLIT ON REAL SHAPES ────────────────────────────────────
print("\n🔬 GPU+CPU Co-Execution on LLaMA Shapes")
print("-" .padding(toLength: 75, withPad: "-", startingAt: 0))

// QKV split
let qkv_split_ms: Double = try! {
    let gpuRatio = 0.57  // Based on typical GPU/(GPU+CPU) throughput ratio
    let gpuRows = Int(Double(BS) * gpuRatio)
    let cpuRows = BS - gpuRows
    
    let aPtr = qkv_x.buffer.pointer.bindMemory(to: Float.self, capacity: qkv_x.count)
    let bPtr = qkv_w.buffer.pointer.bindMemory(to: Float.self, capacity: qkv_w.count)
    
    return try measure("QKV Split") {
        let result = try Tensor(shape: [BS, H], dtype: .float32)
        let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        
        try GPUEngine.shared.matmulRawMPS(
            a: qkv_x.metalBuffer, b: qkv_w.metalBuffer, result: result.metalBuffer,
            M: gpuRows, N: H, K: H,
            aOffset: cpuRows * H * 4,
            resultOffset: cpuRows * H * 4,
            waitUntilCompleted: false
        )
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   Int32(cpuRows), Int32(H), Int32(H), 1.0,
                   aPtr, Int32(H), bPtr, Int32(H), 0.0, rPtr, Int32(H))
        
        GPUEngine.shared.sync()
    }
}()

let qkv_split_gflops = (qkv_flops / qkv_split_ms) / 1_000_000
let qkv_speedup = qkv_ms / qkv_split_ms
print(String(format: "  QKV GPU-only: %7.2f ms  %.0f GFLOPS", qkv_ms, qkv_gflops))
print(String(format: "  QKV GPU+CPU:  %7.2f ms  %.0f GFLOPS  → %.2fx speedup", qkv_split_ms, qkv_split_gflops, qkv_speedup))

// MLP up split
let mlp_up_split_ms: Double = try! {
    let gpuRows = Int(Double(BS) * 0.57)
    let cpuRows = BS - gpuRows
    
    let aPtr = qkv_x.buffer.pointer.bindMemory(to: Float.self, capacity: qkv_x.count)
    let bPtr = mlp_w1.buffer.pointer.bindMemory(to: Float.self, capacity: mlp_w1.count)
    
    return try measure("MLP up Split") {
        let result = try Tensor(shape: [BS, FFN], dtype: .float32)
        let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        
        try GPUEngine.shared.matmulRawMPS(
            a: qkv_x.metalBuffer, b: mlp_w1.metalBuffer, result: result.metalBuffer,
            M: gpuRows, N: FFN, K: H,
            aOffset: cpuRows * H * 4,
            resultOffset: cpuRows * FFN * 4,
            waitUntilCompleted: false
        )
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   Int32(cpuRows), Int32(FFN), Int32(H), 1.0,
                   aPtr, Int32(H), bPtr, Int32(FFN), 0.0, rPtr, Int32(FFN))
        
        GPUEngine.shared.sync()
    }
}()

let mlp_up_split_gflops = (mlp_up_flops / mlp_up_split_ms) / 1_000_000
let mlp_up_speedup = mlp_up_ms / mlp_up_split_ms
print(String(format: "  MLP up GPU-only: %7.2f ms  %.0f GFLOPS", mlp_up_ms, mlp_up_gflops))
print(String(format: "  MLP up GPU+CPU:  %7.2f ms  %.0f GFLOPS  → %.2fx speedup", mlp_up_split_ms, mlp_up_split_gflops, mlp_up_speedup))

// ─── FULL DECODER BLOCK COMPARISON ───────────────────────────────────
print("\n🔬 Full Decoder Block: GPU-only vs GPU+CPU (Batched)")
print("-" .padding(toLength: 75, withPad: "-", startingAt: 0))

// Attention weights
let wq = try! Tensor.random([H, H])
let wk = try! Tensor.random([H, H])
let wv = try! Tensor.random([H, H])
let wo = try! Tensor.random([H, H])
let wg = try! Tensor.random([H, FFN])
let wu = try! Tensor.random([H, FFN])
let wd = try! Tensor.random([FFN, H])
let ln_gamma = try! Tensor.ones([H])
let ln_beta = try! Tensor.zeros([H])
let x = try! Tensor.random([BS, H])

// GPU-only decoder block (with batching)
let gpu_block_ms = try! measure("GPU-only block", warmup: 3, iterations: 10) {
    GPUEngine.shared.startBatch()
    
    // LayerNorm 1
    let ln1 = try GPUEngine.shared.layerNorm(x, gamma: ln_gamma, beta: ln_beta)
    
    // QKV projections
    let q = try GPUEngine.shared.matmulMPS(ln1, wq)
    let k = try GPUEngine.shared.matmulMPS(ln1, wk)
    let v = try GPUEngine.shared.matmulMPS(ln1, wv)
    
    // Output projection
    let attn_out = try GPUEngine.shared.matmulMPS(q, wo) // simplified
    let attn_add = try GPUEngine.shared.add(x, attn_out) // residual
    
    // LayerNorm 2
    let ln2 = try GPUEngine.shared.layerNorm(attn_add, gamma: ln_gamma, beta: ln_beta)
    
    // MLP
    let gate = try GPUEngine.shared.matmulMPS(ln2, wg)
    let up = try GPUEngine.shared.matmulMPS(ln2, wu)
    let act = try GPUEngine.shared.gelu(gate)
    let gated = try GPUEngine.shared.mul(act, up)
    let down = try GPUEngine.shared.matmulMPS(gated, wd)
    let out = try GPUEngine.shared.add(attn_add, down) // residual
    
    GPUEngine.shared.commitBatch()
    GPUEngine.shared.sync()
}

// GPU+CPU co-execution decoder block
let split_block_ms = try! measure("GPU+CPU block", warmup: 3, iterations: 10) {
    GPUEngine.shared.startBatch()
    
    // LayerNorm 1
    let ln1 = try GPUEngine.shared.layerNorm(x, gamma: ln_gamma, beta: ln_beta)
    
    GPUEngine.shared.commitBatch()
    GPUEngine.shared.sync()  // Need ln1 on CPU for split
    
    // QKV projections — GPU+CPU split
    let gpuRows = Int(Double(BS) * 0.57)
    let cpuRows = BS - gpuRows
    
    let ln1Ptr = ln1.buffer.pointer.bindMemory(to: Float.self, capacity: ln1.count)
    let wqPtr = wq.buffer.pointer.bindMemory(to: Float.self, capacity: wq.count)
    
    let q_result = try Tensor(shape: [BS, H], dtype: .float32)
    let qPtr = q_result.buffer.pointer.bindMemory(to: Float.self, capacity: q_result.count)
    
    // GPU does top gpuRows, CPU does bottom cpuRows concurrently
    try GPUEngine.shared.matmulRawMPS(
        a: ln1.metalBuffer, b: wq.metalBuffer, result: q_result.metalBuffer,
        M: gpuRows, N: H, K: H,
        aOffset: cpuRows * H * 4, resultOffset: cpuRows * H * 4,
        waitUntilCompleted: false
    )
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               Int32(cpuRows), Int32(H), Int32(H), 1.0,
               ln1Ptr, Int32(H), wqPtr, Int32(H), 0.0, qPtr, Int32(H))
    
    // K,V similarly on GPU only for simplicity
    let k = try GPUEngine.shared.matmulMPS(ln1, wk)
    let v = try GPUEngine.shared.matmulMPS(ln1, wv)
    
    // Output projection
    let attn_out = try GPUEngine.shared.matmulMPS(q_result, wo)
    let attn_add = try GPUEngine.shared.add(x, attn_out)
    
    // LayerNorm 2
    let ln2 = try GPUEngine.shared.layerNorm(attn_add, gamma: ln_gamma, beta: ln_beta)
    
    GPUEngine.shared.commitBatch()
    GPUEngine.shared.sync()
    
    // MLP — GPU+CPU split on the big matmuls
    let ln2Ptr = ln2.buffer.pointer.bindMemory(to: Float.self, capacity: ln2.count)
    let wgPtr = wg.buffer.pointer.bindMemory(to: Float.self, capacity: wg.count)
    
    let gate_result = try Tensor(shape: [BS, FFN], dtype: .float32)
    let gatePtr = gate_result.buffer.pointer.bindMemory(to: Float.self, capacity: gate_result.count)
    
    try GPUEngine.shared.matmulRawMPS(
        a: ln2.metalBuffer, b: wg.metalBuffer, result: gate_result.metalBuffer,
        M: gpuRows, N: FFN, K: H,
        aOffset: cpuRows * H * 4, resultOffset: cpuRows * FFN * 4,
        waitUntilCompleted: false
    )
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               Int32(cpuRows), Int32(FFN), Int32(H), 1.0,
               ln2Ptr, Int32(H), wgPtr, Int32(FFN), 0.0, gatePtr, Int32(FFN))
    GPUEngine.shared.sync()
    
    let up = try GPUEngine.shared.matmulMPS(ln2, wu)
    GPUEngine.shared.sync()
    
    let act = try GPUEngine.shared.gelu(gate_result)
    let gated = try GPUEngine.shared.mul(act, up)
    let down = try GPUEngine.shared.matmulMPS(gated, wd)
    let out = try GPUEngine.shared.add(attn_add, down)
    GPUEngine.shared.sync()
}

let block_speedup = gpu_block_ms / split_block_ms

// Total FLOPs in decoder block (approximate)
let total_block_flops = Double(BS) * Double(H) * Double(H) * 2.0 * 4  // 4 attention matmuls
    + Double(BS) * Double(H) * Double(FFN) * 2.0 * 3  // 3 MLP matmuls
let gpu_block_gflops = (total_block_flops / gpu_block_ms) / 1_000_000
let split_block_gflops = (total_block_flops / split_block_ms) / 1_000_000
let theory_gpu_block = total_block_flops / (M1_FP32_PEAK * 1_000_000_000) * 1000
let theory_split_block = total_block_flops / (M1_TOTAL_PEAK * 1_000_000_000) * 1000

print(String(format: "  GPU-only:   %7.2f ms  %6.0f GFLOPS (%.0f%% util)", gpu_block_ms, gpu_block_gflops, gpu_block_gflops/M1_FP32_PEAK*100))
print(String(format: "  GPU+CPU:    %7.2f ms  %6.0f GFLOPS (%.0f%% util)  → %.2fx", split_block_ms, split_block_gflops, split_block_gflops/M1_TOTAL_PEAK*100, block_speedup))
print(String(format: "  Theory GPU: %7.2f ms (at 100%% util)", theory_gpu_block))
print(String(format: "  Theory Tri: %7.2f ms (at 100%% util)  → %.2fx vs GPU", theory_split_block, theory_gpu_block/theory_split_block))

print("\n" + "=" .padding(toLength: 75, withPad: "=", startingAt: 0))
print("  VERDICT")
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
print("  If GPU+CPU speedup < 1.3x, bottleneck is sync overhead between splits")
print("  If utilization < 50%, bottleneck is MPS kernel efficiency or memory")
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
