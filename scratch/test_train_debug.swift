import Foundation
import FusionML

// MARK: - Llama3 Transformer Block (approximated with GeGLU using gelu)
public final class Llama3Block: Module {
    public var training: Bool = true {
        didSet {
            qkvProj.training = training
            oProj.training = training
            ln1.training = training
            gateUpProj.training = training
            downProj.training = training
            ln2.training = training
        }
    }
    
    let qkvProj: GradLinear
    let oProj: GradLinear
    let ln1: GradLayerNorm
    let gateUpProj: GradLinear
    let downProj: GradLinear
    let ln2: GradLayerNorm
    let scoreScale: GradTensor

    public init(dim: Int = 4096, hiddenDim: Int = 14336, seqLen: Int = 1024) throws {
        self.qkvProj = try GradLinear(inFeatures: dim, outFeatures: dim * 3, bias: false)
        self.oProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: false)
        self.ln1 = try GradLayerNorm(normalizedShape: [dim])
        self.gateUpProj = try GradLinear(inFeatures: dim, outFeatures: hiddenDim * 2, bias: false)
        self.downProj = try GradLinear(inFeatures: hiddenDim, outFeatures: dim, bias: false)
        self.ln2 = try GradLayerNorm(normalizedShape: [dim])
        // 1/sqrt(D) attention scale, materialized once (constant, no grad) to avoid
        // per-forward-call allocation; Tensor.mul requires exact shape match (no broadcast).
        let scaleValue = Float(1.0 / Double(dim).squareRoot())
        let scaleData = [Float](repeating: scaleValue, count: seqLen * seqLen)
        self.scoreScale = GradTensor.from(try Tensor(scaleData, shape: [seqLen, seqLen]), requiresGrad: false)
    }

    public func forward(_ input: GradTensor) throws -> GradTensor {
        // Pre-LN Self-Attention
        let norm1 = try ln1.forward(input)
        let qkv = try qkvProj.forward(norm1)
        let splits = try GradTensor.split(qkv, parts: 3)
        let q = splits[0]
        let k = splits[1]
        let v = splits[2]

        let rawScores = try GradTensor.matmul(q, k, transposeRight: true, forceGPU: true)
        let scaledScores = try GradTensor.mul(rawScores, scoreScale)
        let scores = try Fusion.nn.functional.softmax(scaledScores, dim: -1)
        let attended = try GradTensor.matmul(scores, v, forceGPU: true)
        let projected = try oProj.forward(attended)
        
        let h1 = try GradTensor.add(input, projected)
        
        // Pre-LN SwiGLU MLP (approximated with GeGLU using gelu)
        let norm2 = try ln2.forward(h1)
        let gateUp = try gateUpProj.forward(norm2)
        let mlpSplits = try GradTensor.split(gateUp, parts: 2)
        let gate = mlpSplits[0]
        let up = mlpSplits[1]
        
        let activatedGate = try gate.gelu()
        let intermediate = try GradTensor.mul(activatedGate, up)
        let output = try downProj.forward(intermediate)
        
        return try GradTensor.add(h1, output)
    }
    
    public func parameters() -> [GradTensor] {
        return qkvProj.parameters() + oProj.parameters() +
               ln1.parameters() + gateUpProj.parameters() + downProj.parameters() +
               ln2.parameters()
    }
    
    public func namedParameters() -> [(String, GradTensor)] {
        return qkvProj.namedParameters() + oProj.namedParameters() +
               ln1.namedParameters() + gateUpProj.namedParameters() + downProj.namedParameters() +
               ln2.namedParameters()
    }
}

func test() {
    Fusion.initialize()
    print("Testing Llama-3 block training step...")
    
    do {
        let llama = try Llama3Block()
        let llamaX = try Fusion.rand([1024, 4096])
        let llamaY = try Tensor(shape: [1024], dtype: .float32)
        let llamaInput = GradTensor(llamaX, requiresGrad: false)
        let llamaOpt = Fusion.optim.adam(llama.parameters(), lr: 0.01)
        
        print("Set training mode...")
        llama.train()
        
        print("Forcing CPU backend...")
        IntelligentRouter.shared.forcedBackend = .cpu
        IntelligentRouter.shared.enableSplitting = false
        
        print("Running zeroGrad...")
        llamaOpt.zeroGrad()
        
        print("Running forward pass...")
        let output = try llama.forward(llamaInput)
        print("Forward pass completed. Output shape: \(output.shape)")
        
        print("Computing cross entropy loss...")
        let loss = try Fusion.nn.functional.crossEntropy(output, llamaY)
        print("Loss computed: \(loss.data.toArray()[0])")
        
        print("Running backward pass...")
        try Fusion.autograd.backward(loss)
        print("Backward pass completed!")
        
        print("Running optimizer step...")
        try llamaOpt.step()
        print("Optimizer step completed successfully!")
        
        print("All tests completed successfully!")
    } catch {
        print("CATCH ERROR: \(error)")
    }
}

test()
