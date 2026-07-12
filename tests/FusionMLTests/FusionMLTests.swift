import XCTest
@testable import FusionML

final class FusionMLTests: XCTestCase {
    
    override func setUp() {
        Fusion.initialize()
    }
    
    // MARK: - Tensor Tests
    
    func testTensorCreation() throws {
        let t = try Fusion.zeros([2, 3])
        XCTAssertEqual(t.shape, [2, 3])
        XCTAssertEqual(t.count, 6)
    }
    
    func testTensorRandom() throws {
        let t = try Fusion.rand([10])
        XCTAssertEqual(t.count, 10)
    }
    
    // MARK: - Linear Algebra Tests
    
    func testMatmul() throws {
        let a = try Fusion.ones([2, 3])
        let b = try Fusion.ones([3, 4])
        let c = try Fusion.linalg.matmul(a, b)
        
        XCTAssertEqual(c.shape, [2, 4])
        XCTAssertEqual(c.toArray()[0], 3.0, accuracy: 0.001)
    }
    
    // MARK: - Neural Network Tests
    
    func testLinearLayer() throws {
        let layer = try Fusion.nn.linear(10, 5)
        let input = GradTensor(try Fusion.rand([4, 10]), requiresGrad: false)
        let output = try layer.forward(input)
        
        XCTAssertEqual(output.shape, [4, 5])
    }
    
    func testSequential() throws {
        let model = Fusion.nn.sequential(
            try Fusion.nn.linear(10, 5),
            Fusion.nn.relu()
        )
        
        let input = GradTensor(try Fusion.rand([2, 10]), requiresGrad: false)
        let output = try model.forward(input)
        
        XCTAssertEqual(output.shape, [2, 5])
    }
    
    // MARK: - Autograd Tests
    
    func testBackward() throws {
        let x = GradTensor(try Fusion.rand([2, 3]), requiresGrad: true)
        let y = try x.sum()
        try Fusion.autograd.backward(y)
        
        XCTAssertNotNil(x.grad)
    }
    
    func testSplit() throws {
        let x = GradTensor(try Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3]), requiresGrad: true)
        let splits = try GradTensor.split(x, parts: 3)
        
        XCTAssertEqual(splits.count, 3)
        XCTAssertEqual(splits[0].shape, [2, 1])
        XCTAssertEqual(splits[1].shape, [2, 1])
        XCTAssertEqual(splits[2].shape, [2, 1])
        
        XCTAssertEqual(splits[0].data.toArray(), [1.0, 4.0])
        XCTAssertEqual(splits[1].data.toArray(), [2.0, 5.0])
        XCTAssertEqual(splits[2].data.toArray(), [3.0, 6.0])
        
        // Backward test: sum(splits[0]) * 2 + sum(splits[1]) * 5 + sum(splits[2]) * 10
        let s0 = try splits[0].sum()
        let s1 = try splits[1].sum()
        let s2 = try splits[2].sum()
        
        let loss1 = try GradTensor.mul(s0, GradTensor(try Tensor([2.0])))
        let loss2 = try GradTensor.mul(s1, GradTensor(try Tensor([5.0])))
        let loss3 = try GradTensor.mul(s2, GradTensor(try Tensor([10.0])))
        
        let loss = try GradTensor.add(try GradTensor.add(loss1, loss2), loss3)
        try Fusion.autograd.backward(loss)
        
        XCTAssertNotNil(x.grad)
        let grad = x.grad!.toArray()
        // Expected gradient: row 1 [2, 5, 10], row 2 [2, 5, 10]
        XCTAssertEqual(grad, [2.0, 5.0, 10.0, 2.0, 5.0, 10.0])
    }
}
