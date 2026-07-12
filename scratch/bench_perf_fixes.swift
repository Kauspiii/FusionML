// PerformanceFix Benchmark
// Validates the performance improvements from:
//   1. FP16 MPS matmul support
//   2. Forward pass batching (command buffer reuse)
//   3. GPU-only backward ops
//   4. SIMD FP32<->FP16 conversions

import FusionML
import Foundation

func benchmarkMatmul(label: String, iterations: Int = 20, warmup: Int = 5, _ op: () throws -> Void) rethrows -> Double {
    for _ in 0..<warmup { try op() }
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations { try op() }
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
    return elapsed
}

print("=" .padding(toLength: 70, withPad: "=", startingAt: 0))
print("  FusionML Performance Fix Benchmark")
print("=" .padding(toLength: 70, withPad: "=", startingAt: 0))

// ─────────────────────────────────────────────────────────────
// TEST 1: FP32 vs FP16 Matmul via MPS
// ─────────────────────────────────────────────────────────────
print("\n📊 TEST 1: FP32 vs FP16 Matmul (MPS)")
print("-" .padding(toLength: 70, withPad: "-", startingAt: 0))

for size in [512, 1024, 2048] {
    let fp32_a = try! Tensor.random([size, size])
    let fp32_b = try! Tensor.random([size, size])
    
    // FP32 path
    let fp32_ms = try! benchmarkMatmul(label: "FP32 \(size)x\(size)") {
        let _ = try GPUEngine.shared.matmulMPS(fp32_a, fp32_b)
        GPUEngine.shared.sync()
    }
    
    // FP16 path - create FP16 tensors
    let fp16_a = try! Tensor(shape: [size, size], dtype: .float16)
    let fp16_b = try! Tensor(shape: [size, size], dtype: .float16)
    
    // Convert to FP16
    let srcA = fp32_a.buffer.pointer.bindMemory(to: Float.self, capacity: size * size)
    let dstA = fp16_a._buffer.pointer.bindMemory(to: Float16.self, capacity: size * size)
    GPUEngine.convertFP32ToFP16(src: srcA, dst: dstA, count: size * size)
    
    let srcB = fp32_b.buffer.pointer.bindMemory(to: Float.self, capacity: size * size)
    let dstB = fp16_b._buffer.pointer.bindMemory(to: Float16.self, capacity: size * size)
    GPUEngine.convertFP32ToFP16(src: srcB, dst: dstB, count: size * size)
    
    let fp16_ms = try! benchmarkMatmul(label: "FP16 \(size)x\(size)") {
        let _ = try GPUEngine.shared.matmulMPS(fp16_a, fp16_b)
        GPUEngine.shared.sync()
    }
    
    let flops = 2.0 * Double(size * size * size)
    let fp32_gflops = (flops / fp32_ms) / 1_000_000
    let fp16_gflops = (flops / fp16_ms) / 1_000_000
    let speedup = fp32_ms / fp16_ms
    
    print("  \(size)×\(size): FP32 \(String(format: "%.2f", fp32_ms))ms (\(String(format: "%.0f", fp32_gflops)) GFLOPS) | FP16 \(String(format: "%.2f", fp16_ms))ms (\(String(format: "%.0f", fp16_gflops)) GFLOPS) | Speedup: \(String(format: "%.2f", speedup))x")
}

// ─────────────────────────────────────────────────────────────
// TEST 2: Batched vs Un-batched Forward Pass
// ─────────────────────────────────────────────────────────────
print("\n📊 TEST 2: Batched vs Un-batched GPU Commands")
print("-" .padding(toLength: 70, withPad: "-", startingAt: 0))

let size = 1024
let a = try! Tensor.random([size, size])
let b = try! Tensor.random([size, size])
let c = try! Tensor.random([size, size])

// Un-batched: 3 separate matmuls, each commits its own command buffer
let unbatched_ms = try! benchmarkMatmul(label: "Unbatched") {
    let r1 = try GPUEngine.shared.matmulMPS(a, b)
    GPUEngine.shared.sync()
    let r2 = try GPUEngine.shared.matmulMPS(r1, c)
    GPUEngine.shared.sync()
    let r3 = try GPUEngine.shared.matmulMPS(r2, a)
    GPUEngine.shared.sync()
}

// Batched: all 3 matmuls in one command buffer, sync only at end
let batched_ms = try! benchmarkMatmul(label: "Batched") {
    GPUEngine.shared.startBatch()
    let r1 = try GPUEngine.shared.matmulMPS(a, b)
    let r2 = try GPUEngine.shared.matmulMPS(r1, c)
    let r3 = try GPUEngine.shared.matmulMPS(r2, a)
    GPUEngine.shared.commitBatch()
    GPUEngine.shared.sync()
}

let batch_speedup = unbatched_ms / batched_ms
print("  3×Matmul chain (\(size)×\(size)):")
print("    Unbatched (3 syncs):  \(String(format: "%.2f", unbatched_ms)) ms")
print("    Batched   (1 sync):   \(String(format: "%.2f", batched_ms)) ms")
print("    Speedup: \(String(format: "%.2f", batch_speedup))x")

// ─────────────────────────────────────────────────────────────
// TEST 3: SIMD FP32<->FP16 Conversion Speed
// ─────────────────────────────────────────────────────────────
print("\n📊 TEST 3: SIMD vs Scalar FP32→FP16 Conversion")
print("-" .padding(toLength: 70, withPad: "-", startingAt: 0))

let convSize = 1024 * 1024  // 1M elements
let srcBuf = UnsafeMutablePointer<Float>.allocate(capacity: convSize)
let dstBuf = UnsafeMutablePointer<Float16>.allocate(capacity: convSize)

for i in 0..<convSize { srcBuf[i] = Float.random(in: -1...1) }

// SIMD conversion (our new path)
let simd_ms = try! benchmarkMatmul(label: "SIMD", iterations: 50) {
    GPUEngine.convertFP32ToFP16(src: srcBuf, dst: dstBuf, count: convSize)
}

// Scalar conversion (old path)
let scalar_ms = try! benchmarkMatmul(label: "Scalar", iterations: 50) {
    for i in 0..<convSize {
        dstBuf[i] = Float16(srcBuf[i])
    }
}

print("  1M elements FP32→FP16:")
print("    SIMD:   \(String(format: "%.3f", simd_ms)) ms")
print("    Scalar: \(String(format: "%.3f", scalar_ms)) ms")
print("    Speedup: \(String(format: "%.2f", scalar_ms / simd_ms))x")

srcBuf.deallocate()
dstBuf.deallocate()

// ─────────────────────────────────────────────────────────────
// TEST 4: GPU vs CPU GELU Backward
// ─────────────────────────────────────────────────────────────
print("\n📊 TEST 4: GPU vs CPU GELU Backward")
print("-" .padding(toLength: 70, withPad: "-", startingAt: 0))

let geluSize = 4096 * 4096
let geluInput = try! Tensor.random([4096, 4096])
let geluGrad = try! Tensor.random([4096, 4096])

// GPU kernel
let gpu_gelu_ms = try! benchmarkMatmul(label: "GPU GELU Backward", iterations: 20) {
    let _ = try GPUEngine.shared.geluBackward(geluInput, geluGrad)
    GPUEngine.shared.sync()
}

// CPU scalar (simulating old path)
let cpu_gelu_ms = try! benchmarkMatmul(label: "CPU GELU Backward", iterations: 3) {
    let result = try Tensor(shape: [4096, 4096], dtype: .float32)
    let xPtr = geluInput.buffer.pointer.bindMemory(to: Float.self, capacity: geluSize)
    let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: geluSize)
    let upPtr = geluGrad.buffer.pointer.bindMemory(to: Float.self, capacity: geluSize)
    
    for i in 0..<geluSize {
        let xi = xPtr[i]
        let cdf = 0.5 * (1 + tanh(0.7978845608 * Double(xi + 0.044715 * xi * xi * xi)))
        let pdf = exp(-0.5 * Double(xi) * Double(xi)) / sqrt(2 * .pi)
        let grad = Float(cdf + Double(xi) * pdf)
        rPtr[i] = upPtr[i] * grad
    }
}

print("  GELU Backward (4096×4096 = \(geluSize / 1_000_000)M elements):")
print("    GPU kernel: \(String(format: "%.2f", gpu_gelu_ms)) ms")
print("    CPU scalar: \(String(format: "%.2f", cpu_gelu_ms)) ms")
print("    Speedup: \(String(format: "%.0f", cpu_gelu_ms / gpu_gelu_ms))x 🔥")

// ─────────────────────────────────────────────────────────────
// TEST 5: Batched Forward + Backward MLP
// ─────────────────────────────────────────────────────────────
print("\n📊 TEST 5: MLP Forward+Backward (Batched vs Unbatched)")
print("-" .padding(toLength: 70, withPad: "-", startingAt: 0))

let mlpInput = GradTensor(try! Tensor.random([64, 4096]), requiresGrad: true)
let w1 = GradTensor(try! Tensor.random([4096, 4096]), requiresGrad: true)
w1.data.isParameter = true
let w2 = GradTensor(try! Tensor.random([4096, 4096]), requiresGrad: true)
w2.data.isParameter = true

// Unbatched
let mlp_unbatched_ms = try! benchmarkMatmul(label: "MLP Unbatched", iterations: 10, warmup: 3) {
    let h = try GradTensor.matmul(mlpInput, w1)
    let act = try h.gelu()
    let out = try GradTensor.matmul(act, w2)
    let loss = try out.sum()
    try loss.backward()
    GPUEngine.shared.sync()
}

// Batched
let mlp_batched_ms = try! benchmarkMatmul(label: "MLP Batched", iterations: 10, warmup: 3) {
    IntelligentRouter.shared.startForwardPass()
    let h = try GradTensor.matmul(mlpInput, w1)
    let act = try h.gelu()
    let out = try GradTensor.matmul(act, w2)
    IntelligentRouter.shared.endForwardPass()
    GPUEngine.shared.sync()
    
    let loss = try out.sum()
    try loss.backward()
    GPUEngine.shared.sync()
}

let mlp_speedup = mlp_unbatched_ms / mlp_batched_ms
print("  MLP (64×4096 → 4096 → GELU → 4096):")
print("    Unbatched: \(String(format: "%.2f", mlp_unbatched_ms)) ms")
print("    Batched:   \(String(format: "%.2f", mlp_batched_ms)) ms")
print("    Speedup:   \(String(format: "%.2f", mlp_speedup))x")

// ─────────────────────────────────────────────────────────────
// SUMMARY
// ─────────────────────────────────────────────────────────────
print("\n" + "=" .padding(toLength: 70, withPad: "=", startingAt: 0))
print("  SUMMARY")
print("=" .padding(toLength: 70, withPad: "=", startingAt: 0))
print("  FP16 Matmul:       Available ✅")
print("  Batched Commands:  \(String(format: "%.2f", batch_speedup))x speedup ✅")
print("  SIMD Conversion:   \(String(format: "%.2f", scalar_ms / simd_ms))x speedup ✅")
print("  GPU GELU Backward: \(String(format: "%.0f", cpu_gelu_ms / gpu_gelu_ms))x speedup ✅")
print("  MLP Batched:       \(String(format: "%.2f", mlp_speedup))x speedup ✅")
print("=" .padding(toLength: 70, withPad: "=", startingAt: 0))
