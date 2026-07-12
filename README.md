# FusionML

**High-Performance Machine Learning Framework for Apple Silicon**

FusionML delivers PyTorch-like ease of use with a unique advantage: **intelligent parallel execution across GPU, CPU, and Neural Engine** — measured **1.13–1.25×** faster transformer blocks than a precision-matched MLX baseline on M4, and up to **1.79×** on batch-scale matmuls, with a runtime gate that never regresses below baseline. All numbers fair-baseline and reproducible: see [Performance](#performance).

## Features

- 🔥 **PyTorch-Style API** - Familiar `Fusion.nn`, `Fusion.optim`, `Fusion.autograd`
- ⚡ **Intelligent Routing** - Automatic work distribution across GPU + CPU
- 🧠 **Full Autograd** - Computation graph with backpropagation
- 🎯 **Apple Silicon Optimized** - Metal, Accelerate, and Neural Engine
- 📦 **Zero Dependencies** - Pure Swift with system frameworks only

## Installation

### Swift Package Manager (Local)

```swift
dependencies: [
    .package(path: "../FusionML")
]
```

## Quick Start

```swift
import FusionML

// Initialize
Fusion.initialize()

// Create tensors
let x = try Fusion.rand([32, 784])
let y = try Fusion.randint(0, 10, [32])

// Build model
let model = Fusion.nn.sequential(
    Fusion.nn.linear(784, 256),
    Fusion.nn.relu(),
    Fusion.nn.linear(256, 10)
)

// Optimizer
let optimizer = Fusion.optim.adam(model.parameters(), lr: 0.001)

// Training loop
for epoch in 0..<10 {
    optimizer.zeroGrad()
    
    let output = try model.forward(GradTensor(x, requiresGrad: false))
    let loss = try Fusion.nn.functional.crossEntropy(output, y)
    
    try Fusion.autograd.backward(loss)
    try optimizer.step()
    
    print("Epoch \(epoch): Loss = \(loss.data.toArray()[0])")
}
```

## API Reference

### Neural Network (`Fusion.nn`)
```swift
Fusion.nn.linear(inFeatures, outFeatures)
Fusion.nn.relu()
Fusion.nn.gelu()
Fusion.nn.dropout(0.5)
Fusion.nn.layerNorm([hiddenSize])
Fusion.nn.sequential(layer1, layer2, ...)
```

### Functional (`Fusion.nn.functional`)
```swift
Fusion.nn.functional.relu(tensor)
Fusion.nn.functional.softmax(tensor)
Fusion.nn.functional.crossEntropy(predictions, targets)
Fusion.nn.functional.mse(predictions, targets)
```

### Optimizers (`Fusion.optim`)
```swift
Fusion.optim.sgd(parameters, lr: 0.01, momentum: 0.9)
Fusion.optim.adam(parameters, lr: 0.001)
Fusion.optim.adamw(parameters, lr: 0.001, weightDecay: 0.01)
```

### Hardware Backends
```swift
Fusion.linalg.matmul(a, b)  // Intelligent routing (fastest!)
Fusion.cpu.matmul(a, b)      // Force CPU
Fusion.gpu.matmul(a, b)      // Force GPU
```

## Performance

Fair-baseline, precision-matched (FP16 vs FP16), correctness-checked. Full
methodology, per-cell numbers, and honest negatives: [`benchmarks/PAPER_READINESS.md`](benchmarks/PAPER_READINESS.md).

| Workload | vs fair MLX-FP16 baseline | Hardware |
|-----------|--------------------------|----------|
| Transformer decoder block, per-layer CPU+GPU split (seq 1024–8192) | **1.13–1.25×** | M4 24GB (replicated) |
| Same, on 8GB fanless M1 | 1.06–1.18× | M1 8GB |
| Batch-scale matmul, contention-aware 3-way split (4096–8192 rows) | up to **1.79×** vs GPU-only | M1 |
| Dynamic runtime gate (split/nosplit/eager) | never below baseline (≥1.0× every cell) | M4 |

Known honest negatives (disclosed, not hidden): single-block *training* is
0.86–0.97× vs MLX-FP16 on both chips, and ANE dispatch overhead (~20 ms via
CoreML) makes per-layer ANE routing non-viable on current stacks.

### Contribute benchmark results (M2/M3/M4 wanted!)

We're collecting clean-protocol results across Apple Silicon generations —
especially **M2, M3 Pro, and M4 Pro**. One command, ~2–3 unattended hours:

```bash
git clone https://github.com/ommo007/FusionML.git && cd FusionML
git checkout benchmarks
./benchmarks/run_clean_suite.sh
```

Then open a PR with your `benchmarks/results/<your-chip>/` folder — the PR
template guides you. Details: [`benchmarks/README.md`](benchmarks/README.md).

## How It Works

FusionML's **IntelligentRouter** analyzes each operation and distributes work:

```
Traditional:  GPU ────────────────→ Result

FusionML:     GPU (68%) ────┬─────→ Result (faster!)
              CPU (32%) ────┘
```

The split ratio is calibrated per-device for optimal throughput.

## Requirements

- macOS 12.0+
- Apple Silicon (M series)
- Swift 5.9+

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
