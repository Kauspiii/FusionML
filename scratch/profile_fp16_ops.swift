// Profile individual FP16 operations in the LLaMA-3 block to find the 25ms gap vs MLX
import FusionML
import Foundation

func measureMicros(_ label: String, warmup: Int = 10, iterations: Int = 50, _ op: () throws -> Void) rethrows -> Double {
    for _ in 0..<warmup { try op() }
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations { try op() }
    let seconds = (CFAbsoluteTimeGetCurrent() - start) / Double(iterations)
    return seconds * 1_000_000 // Microseconds
}

let BS = 4096
let H = 4096
let FFN = 14336

print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
print("  FusionML FP16 Op-by-Op Profiler")
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))

// Tensors
let wq = try! Tensor(shape: [H, H], dtype: .float16)
let wg = try! Tensor(shape: [H, FFN], dtype: .float16)
let wd = try! Tensor(shape: [FFN, H], dtype: .float16)
let ln_gamma = try! Tensor(shape: [H], dtype: .float16)
let ln_beta = try! Tensor(shape: [H], dtype: .float16)
let x = try! Tensor(shape: [BS, H], dtype: .float16)

// Outputs
let ln1 = try! GPUEngine.shared.layerNorm(x, gamma: ln_gamma, beta: ln_beta)
let q = try! GPUEngine.shared.matmulMPS(ln1, wq)
let gate = try! GPUEngine.shared.matmulMPS(ln1, wg)
let act = try! GPUEngine.shared.gelu(gate)

// 1. LayerNorm
let ln_us = try! measureMicros("LayerNorm FP16") {
    let _ = try GPUEngine.shared.layerNorm(x, gamma: ln_gamma, beta: ln_beta)
    GPUEngine.shared.sync()
}
print(String(format: "  LayerNorm:     %8.2f us", ln_us))

// 2. Matmul QKV [4096,4096] x [4096,4096]
let matmul_qkv_us = try! measureMicros("Matmul QKV") {
    let _ = try GPUEngine.shared.matmulMPS(ln1, wq)
    GPUEngine.shared.sync()
}
print(String(format: "  Matmul QKV:    %8.2f us", matmul_qkv_us))

// 3. Matmul Up/Gate [4096,4096] x [4096,14336]
let matmul_up_us = try! measureMicros("Matmul Up") {
    let _ = try GPUEngine.shared.matmulMPS(ln1, wg)
    GPUEngine.shared.sync()
}
print(String(format: "  Matmul Up:     %8.2f us", matmul_up_us))

// 4. Matmul Down [4096,14336] x [14336,4096]
let matmul_dn_us = try! measureMicros("Matmul Down") {
    let _ = try GPUEngine.shared.matmulMPS(gate, wd)
    GPUEngine.shared.sync()
}
print(String(format: "  Matmul Down:   %8.2f us", matmul_dn_us))

// 5. GELU
let gelu_us = try! measureMicros("GELU FP16") {
    let _ = try GPUEngine.shared.gelu(gate)
    GPUEngine.shared.sync()
}
print(String(format: "  GELU:          %8.2f us", gelu_us))

// 6. Mul (Element-wise)
let mul_us = try! measureMicros("Mul FP16") {
    let _ = try GPUEngine.shared.mul(act, act)
    GPUEngine.shared.sync()
}
print(String(format: "  Mul:           %8.2f us", mul_us))

// 7. Add (Element-wise)
let add_us = try! measureMicros("Add FP16") {
    let _ = try GPUEngine.shared.add(x, x)
    GPUEngine.shared.sync()
}
print(String(format: "  Add:           %8.2f us", add_us))

// Calculate total theoretical sum of ops
let total_ops_us = (ln_us * 2) + (matmul_qkv_us * 4) + (matmul_up_us * 2) + matmul_dn_us + gelu_us + mul_us + (add_us * 2)
print("\n" + "-" .padding(toLength: 75, withPad: "-", startingAt: 0))
print(String(format: "  Sum of Individual Ops:  %8.2f ms", total_ops_us / 1000.0))
print("=" .padding(toLength: 75, withPad: "=", startingAt: 0))
