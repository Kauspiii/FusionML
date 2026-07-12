import Foundation
import CoreML

// Define a simple Tensor-like structure to hold buffer data
class FloatTensor {
    let shape: [Int]
    let count: Int
    let pointer: UnsafeMutablePointer<Float>
    
    init(shape: [Int]) {
        self.shape = shape
        self.count = shape.reduce(1, *)
        self.pointer = UnsafeMutablePointer<Float>.allocate(capacity: count)
        self.pointer.initialize(repeating: 0.0, count: count)
    }
    
    deinit {
        self.pointer.deallocate()
    }
}

func main() {
    print("🚀 Starting Native Swift ANE Linear Benchmark...")
    
    let basePath = "/Users/ommohite/Documents/Programming/FusionML/models"
    let mlpackageURL = URL(fileURLWithPath: "\(basePath)/matmul_2048.mlpackage") // Or static_linear_2048
    let staticLinearURL = URL(fileURLWithPath: "\(basePath)/static_linear_2048.mlpackage")
    
    do {
        print("Compiling CoreML model...")
        let compiledURL = try MLModel.compileModel(at: staticLinearURL)
        print("Compiled URL: \(compiledURL)")
        
        let config = MLModelConfiguration()
        config.computeUnits = .all // Force ANE if possible
        
        print("Loading Model onto ANE...")
        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        
        let B = 512
        let K = 2048
        let O = 2048
        
        print("Creating input MLMultiArray shape [\(B), \(K)]...")
        let xArray = try MLMultiArray(shape: [B, K] as [NSNumber], dataType: .float16)
        
        // Populate input with dummy float16 values
        xArray.withUnsafeMutableBytes { ptr, strides in
            let dest = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
            for i in 0..<(B * K) {
                dest[i] = Float16(1.0)
            }
        }
        
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: xArray)
        ])
        
        print("Warming up ANE (5 iterations)...")
        for _ in 0..<5 {
            _ = try model.prediction(from: input)
        }
        
        print("Benchmarking ANE (20 iterations)...")
        var times: [Double] = []
        for _ in 0..<20 {
            let start = CFAbsoluteTimeGetCurrent()
            let output = try model.prediction(from: input)
            let end = CFAbsoluteTimeGetCurrent()
            times.append((end - start) * 1000)
            
            // Touch output to avoid compiler optimization dead-code elimination
            _ = output.featureValue(for: "var_5") // var_5 is the renamed output name in CoreML
        }
        
        times.sort()
        let medianLat = times[times.count / 2]
        let ops = 2.0 * Double(B * K * O)
        let gflops = (ops / medianLat) / 1_000_000
        let tflops = gflops / 1000.0
        
        print("------------------------------------------------------------")
        print(String(format: "  Median ANE Latency: %.3f ms", medianLat))
        print(String(format: "  Throughput:         %.1f GFLOPS (%.3f TFLOPS)", gflops, tflops))
        print("------------------------------------------------------------")
        
    } catch {
        print("❌ Error: \(error)")
    }
}

main()
