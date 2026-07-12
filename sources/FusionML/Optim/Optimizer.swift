// Optimizer - Base optimizer protocol and implementations
// SGD, Adam, AdamW with learning rate scheduling

import Foundation

// MARK: - Parameter Protocol

/// A trainable parameter
public protocol Parameter: AnyObject {
    var gradTensor: GradTensor { get }
    var name: String { get }
}

/// Simple parameter wrapper
public final class Param: Parameter {
    public let gradTensor: GradTensor
    public let name: String
    
    public init(_ tensor: GradTensor, name: String = "") {
        self.gradTensor = tensor
        self.name = name
    }
}

// MARK: - Optimizer Protocol

public protocol Optimizer {
    var learningRate: Float { get set }
    var parameters: [GradTensor] { get }
    
    func step() throws
    func zeroGrad()
}

// MARK: - SGD Optimizer

/// Stochastic Gradient Descent with momentum
public final class SGD: Optimizer {
    public var learningRate: Float
    public let parameters: [GradTensor]
    public let momentum: Float
    public let weightDecay: Float
    public let nesterov: Bool
    
    private var velocities: [Tensor]
    
    public init(
        parameters: [GradTensor],
        lr: Float = 0.01,
        momentum: Float = 0.0,
        weightDecay: Float = 0.0,
        nesterov: Bool = false
    ) {
        self.parameters = parameters
        self.learningRate = lr
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.nesterov = nesterov
        
        // Initialize velocities to zeros
        self.velocities = parameters.map { param in
            try! Tensor.zeros(param.shape)
        }
    }
    
    public func step() throws {
        GPUEngine.shared.sync()
        for (i, param) in parameters.enumerated() {
            guard let grad = param.grad else { continue }
            
            let gPtr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: param.count)
            let ptr = param.data.buffer.pointer.bindMemory(to: Float.self, capacity: param.count)
            let vPtr = velocities[i].buffer.pointer.bindMemory(to: Float.self, capacity: param.count)
            
            let count = param.count
            let blockSize = 65536
            if count >= blockSize {
                let numBlocks = (count + blockSize - 1) / blockSize
                DispatchQueue.concurrentPerform(iterations: numBlocks) { blockIdx in
                    let start = blockIdx * blockSize
                    let end = min(start + blockSize, count)
                    for j in start..<end {
                        var gj = gPtr[j]
                        let dataVal = ptr[j]
                        
                        // Weight decay (L2 regularization)
                        if weightDecay != 0 {
                            gj += weightDecay * dataVal
                        }
                        
                        if momentum != 0 {
                            // Update velocity
                            let vj = momentum * vPtr[j] + gj
                            vPtr[j] = vj
                            
                            if nesterov {
                                gj = gj + momentum * vj
                            } else {
                                gj = vj
                            }
                        }
                        
                        // Update parameter
                        ptr[j] = dataVal - learningRate * gj
                    }
                }
            } else {
                for j in 0..<count {
                    var gj = gPtr[j]
                    let dataVal = ptr[j]
                    
                    if weightDecay != 0 {
                        gj += weightDecay * dataVal
                    }
                    
                    if momentum != 0 {
                        let vj = momentum * vPtr[j] + gj
                        vPtr[j] = vj
                        
                        if nesterov {
                            gj = gj + momentum * vj
                        } else {
                            gj = vj
                        }
                    }
                    
                    ptr[j] = dataVal - learningRate * gj
                }
            }
        }
    }
    
    public func zeroGrad() {
        for param in parameters {
            param.zeroGrad()
        }
    }
}

// MARK: - Adam Optimizer

/// Adam optimizer with bias correction
public final class Adam: Optimizer {
    public var learningRate: Float
    public let parameters: [GradTensor]
    public let beta1: Float
    public let beta2: Float
    public let epsilon: Float
    public let weightDecay: Float
    public let amsgrad: Bool
    
    private var m: [Tensor]  // First moment
    private var v: [Tensor]  // Second moment
    private var vMax: [Tensor]?  // For AMSGrad
    private var step_t: Int = 0
    
    public init(
        parameters: [GradTensor],
        lr: Float = 0.001,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        weightDecay: Float = 0.0,
        amsgrad: Bool = false
    ) {
        self.parameters = parameters
        self.learningRate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weightDecay = weightDecay
        self.amsgrad = amsgrad
        
        self.m = parameters.map { try! Tensor.zeros($0.shape) }
        self.v = parameters.map { try! Tensor.zeros($0.shape) }
        if amsgrad {
            self.vMax = parameters.map { try! Tensor.zeros($0.shape) }
        }
    }
    
    public func step() throws {
        step_t += 1
        
        let biasCorrection1 = 1 - Foundation.pow(beta1, Float(step_t))
        let biasCorrection2 = 1 - Foundation.pow(beta2, Float(step_t))
        
        for (i, param) in parameters.enumerated() {
            guard let grad = param.grad else { continue }
            
            // Run Adam step updates on GPU if CPU is not forced and AMSGrad is not enabled
            if IntelligentRouter.shared.forcedBackend != .cpu && !amsgrad {
                try GPUEngine.shared.adamStep(
                    w: param.data.metalBuffer,
                    g: grad.metalBuffer,
                    m: m[i].metalBuffer,
                    v: v[i].metalBuffer,
                    count: param.count,
                    lr: learningRate,
                    beta1: beta1,
                    beta2: beta2,
                    eps: epsilon,
                    weightDecay: weightDecay,
                    biasCorrection1: biasCorrection1,
                    biasCorrection2: biasCorrection2
                )
                // Mark weight and moment buffers as dirty so CPU knows they were updated on GPU
                param.data.isDirty = true
                m[i].isDirty = true
                v[i].isDirty = true
                continue
            }
            
            let gPtr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: param.count)
            let ptr = param.data.buffer.pointer.bindMemory(to: Float.self, capacity: param.count)
            let mPtr = m[i].buffer.pointer.bindMemory(to: Float.self, capacity: param.count)
            let vPtr = v[i].buffer.pointer.bindMemory(to: Float.self, capacity: param.count)
            
            let count = param.count
            let blockSize = 65536
            if count >= blockSize {
                let numBlocks = (count + blockSize - 1) / blockSize
                DispatchQueue.concurrentPerform(iterations: numBlocks) { blockIdx in
                    let start = blockIdx * blockSize
                    let end = min(start + blockSize, count)
                    
                    for j in start..<end {
                        let gj = gPtr[j]
                        var dataVal = ptr[j]
                        
                        // Weight decay (AdamW style - decoupled)
                        if weightDecay != 0 {
                            dataVal = dataVal - learningRate * weightDecay * dataVal
                        }
                        
                        // Update biased first moment
                        let mj = beta1 * mPtr[j] + (1 - beta1) * gj
                        mPtr[j] = mj
                        
                        // Update biased second moment
                        let vj = beta2 * vPtr[j] + (1 - beta2) * gj * gj
                        vPtr[j] = vj
                        
                        // Bias correction
                        let mHat = mj / biasCorrection1
                        var vHat = vj / biasCorrection2
                        
                        if amsgrad, let vMaxPtr = vMax?[i].buffer.pointer.bindMemory(to: Float.self, capacity: param.count) {
                            vMaxPtr[j] = max(vMaxPtr[j], vHat)
                            vHat = vMaxPtr[j]
                        }
                        
                        // Update parameter
                        ptr[j] = dataVal - learningRate * mHat / (sqrt(vHat) + epsilon)
                    }
                }
            } else {
                for j in 0..<count {
                    let gj = gPtr[j]
                    var dataVal = ptr[j]
                    
                    if weightDecay != 0 {
                        dataVal = dataVal - learningRate * weightDecay * dataVal
                    }
                    
                    let mj = beta1 * mPtr[j] + (1 - beta1) * gj
                    mPtr[j] = mj
                    
                    let vj = beta2 * vPtr[j] + (1 - beta2) * gj * gj
                    vPtr[j] = vj
                    
                    let mHat = mj / biasCorrection1
                    var vHat = vj / biasCorrection2
                    
                    if amsgrad, let vMaxPtr = vMax?[i].buffer.pointer.bindMemory(to: Float.self, capacity: param.count) {
                        vMaxPtr[j] = max(vMaxPtr[j], vHat)
                        vHat = vMaxPtr[j]
                    }
                    
                    ptr[j] = dataVal - learningRate * mHat / (sqrt(vHat) + epsilon)
                }
            }
        }
    }
    
    public func zeroGrad() {
        for param in parameters {
            param.zeroGrad()
        }
    }
}

// MARK: - Learning Rate Schedulers

public protocol LRScheduler {
    var optimizer: Optimizer { get }
    func step()
    func getLR() -> Float
}

/// Step decay scheduler
public final class StepLR: LRScheduler {
    public let optimizer: Optimizer
    public let stepSize: Int
    public let gamma: Float
    private var epoch: Int = 0
    private let baseLR: Float
    
    public init(optimizer: Optimizer, stepSize: Int, gamma: Float = 0.1) {
        self.optimizer = optimizer
        self.stepSize = stepSize
        self.gamma = gamma
        self.baseLR = optimizer.learningRate
    }
    
    public func step() {
        epoch += 1
        if epoch % stepSize == 0 {
            var opt = optimizer
            opt.learningRate = baseLR * Foundation.pow(gamma, Float(epoch / stepSize))
        }
    }
    
    public func getLR() -> Float {
        return optimizer.learningRate
    }
}

/// Cosine annealing scheduler
public final class CosineAnnealingLR: LRScheduler {
    public let optimizer: Optimizer
    public let tMax: Int
    public let etaMin: Float
    private var epoch: Int = 0
    private let baseLR: Float
    
    public init(optimizer: Optimizer, tMax: Int, etaMin: Float = 0) {
        self.optimizer = optimizer
        self.tMax = tMax
        self.etaMin = etaMin
        self.baseLR = optimizer.learningRate
    }
    
    public func step() {
        epoch += 1
        var opt = optimizer
        opt.learningRate = etaMin + (baseLR - etaMin) * (1 + cos(Float.pi * Float(epoch) / Float(tMax))) / 2
    }
    
    public func getLR() -> Float {
        return optimizer.learningRate
    }
}
