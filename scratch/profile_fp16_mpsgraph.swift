import FusionML
import Foundation

let BS = 4096
let H = 4096
let FFN = 14336

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

let ln1 = graph.addNode(op: .layerNorm, inputs: [x_in, ln1_gamma, ln1_beta], outputShape: [BS, H])
let q = graph.addNode(op: .matmul, inputs: [ln1, wq], outputShape: [BS, H])
let k = graph.addNode(op: .matmul, inputs: [ln1, wk], outputShape: [BS, H])
let v = graph.addNode(op: .matmul, inputs: [ln1, wv], outputShape: [BS, H])

let attn_out = graph.addNode(op: .matmul, inputs: [q, wo], outputShape: [BS, H])
let attn_add = graph.addNode(op: .add, inputs: [x_in, attn_out], outputShape: [BS, H])

let ln2 = graph.addNode(op: .layerNorm, inputs: [attn_add, ln2_gamma, ln2_beta], outputShape: [BS, H])
let gate = graph.addNode(op: .matmul, inputs: [ln2, wg], outputShape: [BS, FFN])
let up = graph.addNode(op: .matmul, inputs: [ln2, wu], outputShape: [BS, FFN])

let gated = graph.addNode(op: .geluMul, inputs: [gate, up], outputShape: [BS, FFN])
let down = graph.addNode(op: .matmul, inputs: [gated, wd], outputShape: [BS, H])
let out = graph.addNode(op: .add, inputs: [attn_add, down], outputShape: [BS, H])

graph.markOutput(out)

print("⚙️ Compiling Conductor FP16 Execution Plan...")
let plan = try! Fusion.conductor.compile(graph: graph, dtype: .float16)

let x = try! Tensor(shape: [BS, H], dtype: .float16)
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

let inputs = [
    x, ln1_gamma_16, ln1_beta_16,
    wq_16, wk_16, wv_16, wo_16,
    ln2_gamma_16, ln2_beta_16,
    wg_16, wu_16, wd_16
]

// Warmup
print("Warming up...")
for _ in 0..<3 {
    let _ = try! Fusion.conductor.execute(plan: plan, inputs: inputs)
}

print("Profiling wave by wave:")
// Let's execute the waves manually and profile each step!
// First setup inputs
for (i, inputSlot) in plan.inputSlots.enumerated() {
    let src = inputs[i]
    plan.bufferPool[inputSlot].buffer = src.metalBuffer
    plan.bufferPool[inputSlot].tensor = src
}

func getTensorName(_ slot: Int) -> String {
    if slot < plan.bufferPool.count {
        return "Slot \(slot) (shape: \(plan.bufferPool[slot].shape))"
    }
    return "Slot \(slot)"
}

for (wIdx, wave) in plan.waves.enumerated() {
    print("\n🌊 Wave \(wIdx):")
    for step in wave {
        let start = CFAbsoluteTimeGetCurrent()
        // Execute single step and synchronize GPU to measure its duration
        try! ConductorEngine.shared.execute(plan: plan, inputs: inputs) // Wait, we can't easily call private executeStep from outside, but we can do a mock run or profile by timing the whole execution of smaller plans, or we can read ConductorEngine.swift and temporarily add profiling print lines there!
    }
}
