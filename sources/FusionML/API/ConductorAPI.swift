// ConductorAPI.swift — User-facing Swift API for the Conductor
// Exposes Fusion.conductor to compile and run execution graphs.

import Foundation

extension Fusion {
    
    /// Conductor: Sub-zero runtime orchestrator targeting parallel GPU/CPU/ANE execution.
    public enum conductor {
        
        /// Compile a computation graph into a frozen ExecutionPlan.
        /// Profiles the graph on available hardware and designs an optimal schedule.
        ///
        /// - Parameter graph: The DAG of operations to compile.
        /// - Returns: A frozen ExecutionPlan ready for zero-allocation execution.
        public static func compile(graph: ConductorCompiler.ComputationGraph, dtype: DType = .float32) throws -> ExecutionPlan {
            return try ConductorCompiler.shared.compile(graph: graph, dtype: dtype)
        }
        
        /// Create a new ComputationGraph builder instance.
        public static func newGraph() -> ConductorCompiler.ComputationGraph {
            return ConductorCompiler.ComputationGraph()
        }
        
        /// Get the shared CostModel instance.
        public static var costModel: CostModel {
            return ConductorCompiler.shared.costModel
        }
        
        /// Replay/execute a pre-compiled plan with a set of input tensors.
        /// Zero allocations, zero runtime decision overhead.
        ///
        /// - Parameters:
        ///   - plan: The pre-compiled plan.
        ///   - inputs: Input tensors mapped to the plan's input slots.
        /// - Returns: The resulting output tensors.
        @discardableResult
        public static func execute(plan: ExecutionPlan, inputs: [Tensor]) throws -> [Tensor] {
            return try ConductorEngine.shared.execute(plan: plan, inputs: inputs)
        }
    }
}
