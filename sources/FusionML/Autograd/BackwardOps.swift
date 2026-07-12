// BackwardOps - Backward implementations for automatic differentiation
// Each operation has a corresponding gradient function

import Foundation
import Accelerate

// MARK: - Differentiable Operations on GradTensor

extension GradTensor {
    
    // MARK: - Matrix Multiplication
    
    /// Matrix multiply with gradient tracking: C = A @ B
    public static func matmul(_ a: GradTensor, _ b: GradTensor, transposeLeft: Bool = false, transposeRight: Bool = false, forceGPU: Bool = false, forceCPU: Bool = false) throws -> GradTensor {
        // Forward pass - use intelligent router
        let result = try IntelligentRouter.shared.matmul(a.data, b.data, transposeLeft: transposeLeft, transposeRight: transposeRight, isWeightGrad: false, forceGPU: forceGPU, forceCPU: forceCPU)
        let output = GradTensor(result, requiresGrad: a.requiresGrad || b.requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [a, b],
                gradFn: { inputs, gradOutput in
                    let aData = inputs[0]
                    let bData = inputs[1]
                    
                    var gradA: Tensor
                    var gradB: Tensor
                    do {
                        if transposeLeft && transposeRight {
                            let aT = try Tensor.transpose(aData)
                            let bT = try Tensor.transpose(bData)
                            gradA = try IntelligentRouter.shared.matmul(bT, gradOutput, transposeRight: true)
                            let gradOutputT = try Tensor.transpose(gradOutput)
                            gradB = try IntelligentRouter.shared.matmul(gradOutputT, aT)
                        } else if transposeLeft {
                            gradA = try IntelligentRouter.shared.matmul(gradOutput, bData, transposeRight: true)
                            gradB = try IntelligentRouter.shared.matmul(aData, gradOutput)
                        } else if transposeRight {
                            gradA = try IntelligentRouter.shared.matmul(gradOutput, bData)
                            let gradOutputT = try Tensor.transpose(gradOutput)
                            gradB = try IntelligentRouter.shared.matmul(gradOutputT, aData)
                        } else {
                            gradA = try IntelligentRouter.shared.matmul(gradOutput, bData, transposeRight: true)
                            let gradOutputT = try Tensor.transpose(gradOutput)
                            let temp = try IntelligentRouter.shared.matmul(gradOutputT, aData)
                            gradB = try Tensor.transpose(temp)
                        }
                    } catch {
                        gradA = gradOutput
                        gradB = gradOutput
                    }
                    
                    return [gradA, gradB]
                },
                name: "matmul"
            )
        }
        
        return output
    }
    
    // MARK: - Element-wise Addition
    
    /// Element-wise add with gradient tracking
    public static func add(_ a: GradTensor, _ b: GradTensor) throws -> GradTensor {
        let result = try IntelligentRouter.shared.add(a.data, b.data)
        let output = GradTensor(result, requiresGrad: a.requiresGrad || b.requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [a, b],
                gradFn: { _, gradOutput in
                    // Gradient flows through unchanged for addition
                    return [gradOutput, gradOutput]
                },
                name: "add"
            )
        }
        
        return output
    }
    
    // MARK: - Element-wise Multiplication
    
    /// Element-wise multiply with gradient tracking
    public static func mul(_ a: GradTensor, _ b: GradTensor) throws -> GradTensor {
        let result = try IntelligentRouter.shared.mul(a.data, b.data)
        let output = GradTensor(result, requiresGrad: a.requiresGrad || b.requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [a, b],
                gradFn: { inputs, gradOutput in
                    // dL/dA = gradOutput * B
                    // dL/dB = gradOutput * A
                    do {
                        let gradA = try IntelligentRouter.shared.mul(gradOutput, inputs[1])
                        let gradB = try IntelligentRouter.shared.mul(gradOutput, inputs[0])
                        return [gradA, gradB]
                    } catch {
                        return [gradOutput, gradOutput]
                    }
                },
                name: "mul"
            )
        }
        
        return output
    }
    
    // MARK: - ReLU Activation
    
    /// ReLU with gradient tracking
    public func relu() throws -> GradTensor {
        let result = try IntelligentRouter.shared.relu(data)
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { inputs, gradOutput in
                    // dL/dx = gradOutput * (x > 0 ? 1 : 0)
                    let input = inputs[0]
                    do {
                        return [try GPUEngine.shared.reluBackward(input, gradOutput)]
                    } catch {
                        return [gradOutput]
                    }
                },
                name: "relu"
            )
        }
        
        return output
    }
    
    // MARK: - GELU Activation
    
    /// GELU with gradient tracking
    public func gelu() throws -> GradTensor {
        let result = try IntelligentRouter.shared.gelu(data)
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { inputs, gradOutput in
                    let x = inputs[0]
                    do {
                        return [try GPUEngine.shared.geluBackward(x, gradOutput)]
                    } catch {
                        return [gradOutput]
                    }
                },
                name: "gelu"
            )
        }
        
        return output
    }
    
    // MARK: - Sum Reduction
    
    /// Sum all elements with gradient tracking
    public func sum() throws -> GradTensor {
        let data = self.data.toArray()
        let sumValue = data.reduce(0, +)
        let result = try Tensor(shape: [1], dtype: self.data.dtype)
        result.buffer.pointer.bindMemory(to: Float.self, capacity: 1).pointee = sumValue
        
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            let inputShape = self.shape
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { _, gradOutput in
                    // Gradient of sum is ones * upstream
                    var grad = try! Tensor.ones(inputShape)
                    let scale = gradOutput.toArray()[0]
                    let ptr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    for i in 0..<grad.count {
                        ptr[i] = scale
                    }
                    return [grad]
                },
                name: "sum"
            )
        }
        
        return output
    }
    
    // MARK: - Mean Reduction
    
    /// Mean of all elements with gradient tracking
    public func mean() throws -> GradTensor {
        let dataArr = self.data.toArray()
        let meanValue = dataArr.reduce(0, +) / Float(dataArr.count)
        let result = try Tensor(shape: [1], dtype: self.data.dtype)
        result.buffer.pointer.bindMemory(to: Float.self, capacity: 1).pointee = meanValue
        
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            let inputShape = self.shape
            let inputCount = self.count
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { _, gradOutput in
                    // Gradient of mean is (1/n) * ones * upstream
                    var grad = try! Tensor(shape: inputShape, dtype: .float32)
                    let scale = gradOutput.toArray()[0] / Float(inputCount)
                    let ptr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    for i in 0..<grad.count {
                        ptr[i] = scale
                    }
                    return [grad]
                },
                name: "mean"
            )
        }
        
        return output
    }
    
    // MARK: - Power
    
    /// Element-wise power with gradient tracking
    public func pow(_ exponent: Float) throws -> GradTensor {
        var result = try Tensor(shape: shape, dtype: data.dtype)
        let inputData = data.toArray()
        let resultPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: count)
        
        for i in 0..<count {
            resultPtr[i] = Foundation.pow(inputData[i], exponent)
        }
        
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            let exp = exponent
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { inputs, gradOutput in
                    // d/dx(x^n) = n * x^(n-1)
                    let x = inputs[0]
                    let xData = x.toArray()
                    var grad = try! Tensor(shape: x.shape, dtype: x.dtype)
                    let gradPtr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    let upstreamData = gradOutput.toArray()
                    
                    for i in 0..<x.count {
                        gradPtr[i] = upstreamData[i] * exp * Foundation.pow(xData[i], exp - 1)
                    }
                    
                    return [grad]
                },
                name: "pow"
            )
        }
        
        return output
    }
    
    // MARK: - Split
    
    /// Split a GradTensor along its second dimension
    public static func split(_ x: GradTensor, parts: Int) throws -> [GradTensor] {
        let splitTensors = try x.data.split(parts: parts)
        let outputs = splitTensors.map { GradTensor($0, requiresGrad: x.requiresGrad) }
        
        if x.requiresGrad {
            for (p, output) in outputs.enumerated() {
                output.isLeaf = false
                output.gradNode = GradNode(
                    inputs: [x],
                    gradFn: { inputs, gradOutput in
                        do {
                            if x.grad == nil {
                                x.grad = try Tensor.zeros(x.shape, dtype: x.data.dtype)
                            }
                            if gradOutput.isDirty {
                                try GPUEngine.shared.splitBackward2D(y: gradOutput, x: x.grad!, parts: parts, partIdx: p)
                            } else {
                                let dstPtr = x.grad!.buffer.pointer.bindMemory(to: Float.self, capacity: x.grad!.count)
                                let srcPtr = gradOutput.buffer.pointer.bindMemory(to: Float.self, capacity: gradOutput.count)
                                let M = x.shape[0]
                                let N = x.shape[1]
                                let partWidth = N / parts
                                
                                for i in 0..<M {
                                    let dstOffset = i * N + p * partWidth
                                    let srcOffset = i * partWidth
                                    memcpy(dstPtr.advanced(by: dstOffset), srcPtr.advanced(by: srcOffset), partWidth * 4)
                                }
                            }
                            return []
                        } catch {
                            return []
                        }
                    },
                    name: "split_\(p)"
                )
            }
        }
        
        return outputs
    }
}
