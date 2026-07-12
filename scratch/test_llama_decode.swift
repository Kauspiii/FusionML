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

    public init(dim: Int = 4096, hiddenDim: Int = 14336, seqLen: Int = 1) throws {
        self.qkvProj = try GradLinear(inFeatures: dim, outFeatures: dim * 3, bias: false)
        self.oProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: false)
        self.ln1 = try GradLayerNorm(normalizedShape: [dim])
        self.gateUpProj = try GradLinear(inFeatures: dim, outFeatures: hiddenDim * 2, bias: false)
        self.downProj = try GradLinear(inFeatures: hiddenDim, outFeatures: dim, bias: false)
        self.ln2 = try GradLayerNorm(normalizedShape: [dim])
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

func benchmarkDecode() {
    Fusion.initialize()
    print("=========================================================")
    print("🚀 FusionML Native Swift Llama-3 Autoregressive Decode Benchmark")
    print("   Batch: 1 | Sequence Length: 1 | Hidden: 4096 | FFN: 14336")
    print("=========================================================")
    
    do {
        let llama = try Llama3Block(seqLen: 1)
        llama.eval()
        let inputX = try Fusion.zeros([1, 4096])
        let input = GradTensor(inputX, requiresGrad: false)
        
        // Warmup
        print("Warming up GPU and compiler...")
        for _ in 0..<20 {
            _ = try llama.forward(input)
        }
        GPUEngine.shared.sync()
        
        // 1. GPU Only
        print("Testing GPU Only Decode...")
        IntelligentRouter.shared.forcedBackend = .gpu
        IntelligentRouter.shared.enableSplitting = false
        
        var gpuTimes: [Double] = []
        for _ in 0..<100 {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try llama.forward(input)
            GPUEngine.shared.sync()
            gpuTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
        }
        let avgGpu = gpuTimes.reduce(0, +) / Double(gpuTimes.count)
        let tpsGpu = 1000.0 / avgGpu
        print(String(format: "   GPU Decode: %.2f ms (%.1f tokens/sec)", avgGpu, tpsGpu))
        
        // 2. CPU Only
        print("Testing CPU Only Decode...")
        IntelligentRouter.shared.forcedBackend = .cpu
        IntelligentRouter.shared.enableSplitting = false
        
        var cpuTimes: [Double] = []
        for _ in 0..<100 {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try llama.forward(input)
            cpuTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
        }
        let avgCpu = cpuTimes.reduce(0, +) / Double(cpuTimes.count)
        let tpsCpu = 1000.0 / avgCpu
        print(String(format: "   CPU Decode: %.2f ms (%.1f tokens/sec)", avgCpu, tpsCpu))
        
        // 3. Smart Split
        print("Testing Smart Split Decode...")
        IntelligentRouter.shared.forcedBackend = nil
        IntelligentRouter.shared.enableSplitting = true
        
        var smartTimes: [Double] = []
        for _ in 0..<100 {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try llama.forward(input)
            GPUEngine.shared.sync()
            smartTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
        }
        let avgSmart = smartTimes.reduce(0, +) / Double(smartTimes.count)
        let tpsSmart = 1000.0 / avgSmart
        print(String(format: "   Smart Split Decode: %.2f ms (%.1f tokens/sec)", avgSmart, tpsSmart))
        
    } catch {
        print("Error: \(error)")
    }
}

benchmarkDecode()
