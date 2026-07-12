// Verification of Custom GEMM correctness
import FusionML
import Foundation

func verifyCorrectness() throws {
    print("Verifying Custom GEMM correctness against CPU (cblas)...")
    
    let sizes = [64, 128, 256, 512, 1024]
    
    for size in sizes {
        let a = try Tensor.random([size, size])
        let b = try Tensor.random([size, size])
        
        let cpuResult = try Tensor.matmul(a, b)
        let customResult = try GPUEngine.shared.matmul(a, b)
        GPUEngine.shared.sync()
        
        let cpuPtr = cpuResult.buffer.pointer.bindMemory(to: Float.self, capacity: size * size)
        let customPtr = customResult.buffer.pointer.bindMemory(to: Float.self, capacity: size * size)
        
        var maxDiff: Float = 0.0
        for i in 0..<(size * size) {
            let diff = abs(cpuPtr[i] - customPtr[i])
            if diff > maxDiff {
                maxDiff = diff
            }
        }
        
        print("Size: \(size)x\(size) - Max absolute difference: \(maxDiff)")
        if maxDiff > 1e-4 {
            print("❌ Correctness check failed at size \(size)!")
            // Let's print some sample elements
            print("First 5 elements of CPU result:")
            for i in 0..<min(5, size*size) {
                print("  [\(i)]: CPU=\(cpuPtr[i]) Custom=\(customPtr[i])")
            }
            throw NSError(domain: "CorrectnessError", code: 1, userInfo: nil)
        } else {
            print("✅ Size \(size) matches perfectly!")
        }
    }
}

do {
    try verifyCorrectness()
    print("Verification completed successfully!")
} catch {
    print("Verification failed with error: \(error)")
}
