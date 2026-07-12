// swift-tools-version: 5.9
// FusionML - High-Performance ML Framework for Apple Silicon

import PackageDescription

let package = Package(
    name: "FusionML",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        // Main library
        .library(
            name: "FusionML",
            targets: ["FusionML"]
        ),
        
        // Examples
        .executable(
            name: "QuickStart",
            targets: ["QuickStart"]
        ),
        .executable(
            name: "TrainingExample",
            targets: ["TrainingExample"]
        ),
        .executable(
            name: "BenchmarkExample",
            targets: ["BenchmarkExample"]
        )
    ],
    targets: [
        // Main FusionML library
        .target(
            name: "FusionML",
            dependencies: [],
            path: "sources/FusionML",
            resources: [
                .process("Metal/Kernels.metal")
            ]
        ),
        
        // Examples
        .executableTarget(
            name: "QuickStart",
            dependencies: ["FusionML"],
            path: "examples/quickstart"
        ),
        .executableTarget(
            name: "TrainingExample",
            dependencies: ["FusionML"],
            path: "examples/training"
        ),
        .executableTarget(
            name: "BenchmarkExample",
            dependencies: ["FusionML"],
            path: "examples/benchmark"
        ),
        
        .executableTarget(
            name: "ConductorVerify",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["test_conductor_correctness.swift"]
        ),
        
        .executableTarget(
            name: "TriComputeBenchmark",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["run_tri_compute_benchmark.swift"]
        ),
        .executableTarget(
            name: "TestTrainDebug",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["test_train_debug.swift"]
        ),
        .executableTarget(
            name: "TestLlamaDecode",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["test_llama_decode.swift"]
        ),
        .executableTarget(
            name: "BenchPerfFixes",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["bench_perf_fixes.swift"]
        ),
        .executableTarget(
            name: "DeepProfiler",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["deep_profiler.swift"]
        ),
        .executableTarget(
            name: "TaskParallelBench",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["task_parallel_bench.swift"]
        ),
        .executableTarget(
            name: "OverheadIsolator",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["overhead_isolator.swift"]
        ),
        .executableTarget(
            name: "BenchCustomGEMM",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["bench_custom_gemm.swift"]
        ),
        .executableTarget(
            name: "VerifyCustomGEMM",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["verify_custom_gemm.swift"]
        ),
        .executableTarget(
            name: "BenchFP16Decoder",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["bench_fp16_decoder.swift"]
        ),
        .executableTarget(
            name: "ProfileFP16Ops",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["profile_fp16_ops.swift"]
        ),
        .executableTarget(
            name: "BenchConductorDecoder",
            dependencies: ["FusionML"],
            path: "scratch",
            sources: ["bench_conductor_decoder.swift"]
        ),
        
        // Tests
        .testTarget(
            name: "FusionMLTests",
            dependencies: ["FusionML"],
            path: "tests/FusionMLTests"
        )
    ]
)
