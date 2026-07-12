import Foundation
import Accelerate

func main() {
    let M = 1
    let K = 4096
    let N = 4096
    
    let a = [Float](repeating: 1.0, count: M * K)
    let b = [Float](repeating: 1.0, count: K * N)
    var c = [Float](repeating: 0.0, count: M * N)
    
    // Warmup
    for _ in 0..<100 {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(M), Int32(N), Int32(K), 1.0,
                    a, Int32(K), b, Int32(N), 0.0, &c, Int32(N))
    }
    
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<1000 {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(M), Int32(N), Int32(K), 1.0,
                    a, Int32(K), b, Int32(N), 0.0, &c, Int32(N))
    }
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 / 1000
    print("Time for 1 sgemm [1, 4096] x [4096, 4096]: \(elapsed) ms")
}
main()
