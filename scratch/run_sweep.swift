
import Foundation
import CoreML

func runBenchmark(batchSize: Int) throws {
    let basePath = "/Users/ommohite/Documents/Programming/FusionML/models"
    let modelURL = URL(fileURLWithPath: "\(basePath)/static_linear_\(batchSize).mlpackage")
    
    let compiledURL = try MLModel.compileModel(at: modelURL)
    let config = MLModelConfiguration()
    config.computeUnits = .all
    
    let model = try MLModel(contentsOf: compiledURL, configuration: config)
    
    let B = batchSize
    let K = 2048
    let O = 2048
    
    let xArray = try MLMultiArray(shape: [B, K] as [NSNumber], dataType: .float16)
    xArray.withUnsafeMutableBytes { ptr, strides in
        let dest = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
        for i in 0..<(B * K) {
            dest[i] = Float16(1.0)
        }
    }
    
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "x": MLFeatureValue(multiArray: xArray)
    ])
    
    // Warmup
    for _ in 0..<5 {
        _ = try model.prediction(from: input)
    }
    
    var times: [Double] = []
    for _ in 0..<30 {
        let start = CFAbsoluteTimeGetCurrent()
        let output = try model.prediction(from: input)
        let end = CFAbsoluteTimeGetCurrent()
        times.append((end - start) * 1000)
        _ = output.featureValue(for: "var_5")
    }
    
    times.sort()
    let medianLat = times[times.count / 2]
    let ops = 2.0 * Double(B * K * O)
    let gflops = (ops / medianLat) / 1_000_000
    let tflops = gflops / 1000.0
    
    print(String(format: "  %4d | %8.3f ms | %10.1f GFLOPS | %8.3f TFLOPS", B, medianLat, gflops, tflops))
}

print("==============================================================")
print("  Batch|   Latency  |    GFLOPS     |   TFLOPS   ")
print("==============================================================")
try? runBenchmark(batchSize: 128)
try? runBenchmark(batchSize: 256)
try? runBenchmark(batchSize: 512)
try? runBenchmark(batchSize: 1024)
print("==============================================================")
