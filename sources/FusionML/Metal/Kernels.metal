// SiliconML Custom Matrix Multiplication Shader
// Optimized tiled matrix multiplication for Apple Silicon GPU

#include <metal_stdlib>
using namespace metal;

// Tile size for shared memory optimization
constant int TILE_SIZE = 32;

// Basic matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
kernel void matmul_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int row = gid.y;
    int col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// High-Performance Register-Blocked GEMM Kernel
// Each thread computes a TM×TN sub-tile of output using register accumulators
// This achieves much higher arithmetic intensity than the naive 1-element-per-thread approach
//
// Configuration:
//   BM=64, BN=64: threadgroup tile size
//   TM=8, TN=8:   per-thread output tile (64 register accumulators)
//   BK=8:         K-dimension tile size
//   Threads per threadgroup: (BM/TM)×(BN/TN) = 8×8 = 64
//   Arithmetic intensity: 16 FLOP/byte from global memory
//
// This is the same strategy used by MLX, CUTLASS, and other high-perf GEMM libraries.

constant int BM = 64;   // Threadgroup tile rows
constant int BN = 64;   // Threadgroup tile cols
constant int BK = 8;    // K-dimension tile
constant int TM = 8;    // Per-thread rows
constant int TN = 8;    // Per-thread cols

kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Shared memory tiles
    threadgroup float As[BM * BK];  // 64×8 = 512 floats = 2KB
    threadgroup float Bs[BK * BN];  // 8×64 = 512 floats = 2KB
    
    // Which thread am I in the 8×8 thread grid?
    const int threadRow = (int)tid / 8;  // 0..7
    const int threadCol = (int)tid % 8;  // 0..7
    
    // Base row/col for this threadgroup's output tile
    const int blockRowStart = (int)tgid.y * BM;
    const int blockColStart = (int)tgid.x * BN;
    
    // Register accumulators — each thread accumulates TM×TN = 64 results
    float acc[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        acc[i] = 0.0f;
    }
    
    // Register caches for A and B tiles
    float regA[TM];
    float regB[TN];
    
    // Number of threads in this threadgroup
    const int numThreads = 64;
    
    // Loop over K-dimension tiles
    const int numKTiles = (K + BK - 1) / BK;
    
    for (int bk = 0; bk < numKTiles; bk++) {
        // ── Cooperative load: A tile [BM × BK] ──────────────────
        // 64 threads loading 512 elements = 8 elements/thread
        for (int loadOffset = 0; loadOffset < BM * BK; loadOffset += numThreads) {
            int idx = loadOffset + (int)tid;
            if (idx < BM * BK) {
                int r = idx / BK;
                int c = idx % BK;
                int globalRow = blockRowStart + r;
                int globalCol = bk * BK + c;
                As[r * BK + c] = (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
            }
        }
        
        // ── Cooperative load: B tile [BK × BN] ──────────────────
        for (int loadOffset = 0; loadOffset < BK * BN; loadOffset += numThreads) {
            int idx = loadOffset + (int)tid;
            if (idx < BK * BN) {
                int r = idx / BN;
                int c = idx % BN;
                int globalRow = bk * BK + r;
                int globalCol = blockColStart + c;
                Bs[r * BN + c] = (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // ── Compute: each thread does TM×TN×BK FMAs ─────────────
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // Load TM values from A_shared into registers
            for (int i = 0; i < TM; i++) {
                regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            // Load TN values from B_shared into registers
            for (int j = 0; j < TN; j++) {
                regB[j] = Bs[dotIdx * BN + threadCol * TN + j];
            }
            // Outer product: TM × TN FMAs
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    acc[i * TN + j] += regA[i] * regB[j];
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // ── Write TM×TN results to global memory ─────────────────────
    for (int i = 0; i < TM; i++) {
        int globalRow = blockRowStart + threadRow * TM + i;
        if (globalRow < M) {
            for (int j = 0; j < TN; j++) {
                int globalCol = blockColStart + threadCol * TN + j;
                if (globalCol < N) {
                    C[globalRow * N + globalCol] = acc[i * TN + j];
                }
            }
        }
    }
}

// FP16 tiled matrix multiplication (even faster)
kernel void matmul_fp16_tiled(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half As[TILE_SIZE][TILE_SIZE];
    threadgroup half Bs[TILE_SIZE][TILE_SIZE];
    
    int row = tgid.y * TILE_SIZE + tid.y;
    int col = tgid.x * TILE_SIZE + tid.x;
    
    half sum = 0.0h;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tid.x;
        if (row < M && aCol < K) {
            As[tid.y][tid.x] = A[row * K + aCol];
        } else {
            As[tid.y][tid.x] = 0.0h;
        }
        
        int bRow = t * TILE_SIZE + tid.y;
        if (bRow < K && col < N) {
            Bs[tid.y][tid.x] = B[bRow * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0h;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Element-wise operations

kernel void add_elementwise(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    C[id] = A[id] + B[id];
}

kernel void add_elementwise_fp16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    C[id] = A[id] + B[id];
}

kernel void mul_elementwise(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    C[id] = A[id] * B[id];
}

kernel void mul_elementwise_fp16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    C[id] = A[id] * B[id];
}

kernel void relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = max(input[id], 0.0f);
}

kernel void relu_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = max(input[id], (half)0.0h);
}

kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[id] = x * cdf;
}

kernel void gelu_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    half x = input[id];
    float x_f = (float)x;
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x_f + 0.044715f * x_f * x_f * x_f)));
    output[id] = (half)(x_f * cdf);
}

kernel void softmax_row(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& cols [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    int offset = row * cols;
    
    // Find max for numerical stability
    float maxVal = input[offset];
    for (int i = 1; i < cols; i++) {
        maxVal = max(maxVal, input[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        float val = exp(input[offset + i] - maxVal);
        output[offset + i] = val;
        sum += val;
    }
    
    // Normalize
    for (int i = 0; i < cols; i++) {
        output[offset + i] /= sum;
    }
}

kernel void relu_backward(
    device const float* input [[buffer(0)]],
    device const float* gradOutput [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    gradInput[id] = input[id] > 0.0f ? gradOutput[id] : 0.0f;
}

kernel void gelu_backward(
    device const float* input [[buffer(0)]],
    device const float* gradOutput [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    float c = 0.7978845608f;
    float inner = c * (x + 0.044715f * x * x * x);
    float tanh_inner = tanh(inner);
    float dgelu = 0.5f * (1.0f + tanh_inner) + 0.5f * x * (1.0f - tanh_inner * tanh_inner) * c * (1.0f + 3.0f * 0.044715f * x * x);
    gradInput[id] = gradOutput[id] * dgelu;
}

kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& lastDim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_mean[32];
    threadgroup float shared_var[32];
    
    int offset = row * lastDim;
    
    // 1. Local sum
    float local_sum = 0.0f;
    for (int i = tid; i < lastDim; i += 256) {
        local_sum += input[offset + i];
    }
    
    // 2. Reduce inside simdgroup
    float simd_sum_val = simd_sum(local_sum);
    if (lane_id == 0) {
        shared_mean[simd_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Final reduction for mean
    float mean = 0.0f;
    if (simd_id == 0) {
        float s = (lane_id < 8) ? shared_mean[lane_id] : 0.0f;
        mean = simd_sum(s) / (float)lastDim;
        if (lane_id == 0) {
            shared_mean[0] = mean;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = shared_mean[0];
    
    // 4. Local variance sum
    float local_var_sum = 0.0f;
    for (int i = tid; i < lastDim; i += 256) {
        float diff = input[offset + i] - mean;
        local_var_sum += diff * diff;
    }
    
    // 5. Reduce variance inside simdgroup
    float simd_var_val = simd_sum(local_var_sum);
    if (lane_id == 0) {
        shared_var[simd_id] = simd_var_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 6. Final reduction for variance
    float std = 0.0f;
    if (simd_id == 0) {
        float s = (lane_id < 8) ? shared_var[lane_id] : 0.0f;
        float variance = simd_sum(s) / (float)lastDim;
        std = rsqrt(variance + eps);
        if (lane_id == 0) {
            shared_var[0] = std;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    std = shared_var[0];
    
    // 7. Normalize and write
    for (int i = tid; i < lastDim; i += 256) {
        output[offset + i] = (input[offset + i] - mean) * std * gamma[i] + beta[i];
    }
}

kernel void layer_norm_fp16(
    device const half* input [[buffer(0)]],
    device const half* gamma [[buffer(1)]],
    device const half* beta [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant int& lastDim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_mean[32];
    threadgroup float shared_var[32];
    
    int offset = row * lastDim;
    
    // 1. Local sum
    float local_sum = 0.0f;
    for (int i = tid; i < lastDim; i += 256) {
        local_sum += (float)input[offset + i];
    }
    
    // 2. Reduce inside simdgroup
    float simd_sum_val = simd_sum(local_sum);
    if (lane_id == 0) {
        shared_mean[simd_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Final reduction for mean
    float mean = 0.0f;
    if (simd_id == 0) {
        float s = (lane_id < 8) ? shared_mean[lane_id] : 0.0f;
        mean = simd_sum(s) / (float)lastDim;
        if (lane_id == 0) {
            shared_mean[0] = mean;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = shared_mean[0];
    
    // 4. Local variance sum
    float local_var_sum = 0.0f;
    for (int i = tid; i < lastDim; i += 256) {
        float diff = (float)input[offset + i] - mean;
        local_var_sum += diff * diff;
    }
    
    // 5. Reduce variance inside simdgroup
    float simd_var_val = simd_sum(local_var_sum);
    if (lane_id == 0) {
        shared_var[simd_id] = simd_var_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 6. Final reduction for variance
    float std = 0.0f;
    if (simd_id == 0) {
        float s = (lane_id < 8) ? shared_var[lane_id] : 0.0f;
        float variance = simd_sum(s) / (float)lastDim;
        std = rsqrt(variance + eps);
        if (lane_id == 0) {
            shared_var[0] = std;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    std = shared_var[0];
    
    // 7. Normalize and write
    for (int i = tid; i < lastDim; i += 256) {
        output[offset + i] = (half)(((float)input[offset + i] - mean) * std * (float)gamma[i] + (float)beta[i]);
    }
}

kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= (uint)cols || gid.y >= (uint)rows) return;
    output[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
}

kernel void split_2d(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant int& M [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& partWidth [[buffer(4)]],
    constant int& partIdx [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int row = gid.y;
    int col = gid.x;
    if (row >= M || col >= partWidth) return;
    
    int srcCol = partIdx * partWidth + col;
    Y[row * partWidth + col] = X[row * N + srcCol];
}

kernel void split_backward_2d(
    device const float* Y [[buffer(0)]],
    device float* X [[buffer(1)]],
    constant int& M [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& partWidth [[buffer(4)]],
    constant int& partIdx [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int row = gid.y;
    int col = gid.x;
    if (row >= M || col >= partWidth) return;
    
    int dstCol = partIdx * partWidth + col;
    X[row * N + dstCol] = Y[row * partWidth + col];
}

kernel void gelu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = gate[id];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    float act = x * cdf;
    output[id] = act * up[id];
}

kernel void gelu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = (float)gate[id];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    float act = x * cdf;
}

kernel void cross_entropy_forward(
    device const float* logits [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* loss [[buffer(2)]],
    device float* probs [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& classes [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if ((int)id >= batch) return;
    
    int offset = id * classes;
    
    // Find max value for numerical stability
    float maxVal = -1e20f;
    for (int c = 0; c < classes; c++) {
        if (logits[offset + c] > maxVal) {
            maxVal = logits[offset + c];
        }
    }
    
    // Compute sum of exponentials
    float sumExp = 0.0f;
    for (int c = 0; c < classes; c++) {
        float val = exp(logits[offset + c] - maxVal);
        probs[offset + c] = val;
        sumExp += val;
    }
    
    // Normalize to get probabilities
    for (int c = 0; c < classes; c++) {
        probs[offset + c] /= sumExp;
    }
    
    // Compute loss for this batch element
    int targetClass = (int)targets[id];
    float p = probs[offset + targetClass];
    if (p < 1e-7f) p = 1e-7f;
    loss[id] = -log(p) / (float)batch;
}

kernel void sum_loss(
    device const float* batch_losses [[buffer(0)]],
    device float* total_loss [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id > 0) return;
    float sum = 0.0f;
    for (int i = 0; i < batch; i++) {
        sum += batch_losses[i];
    }
    total_loss[0] = sum;
}

kernel void cross_entropy_backward(
    device const float* probs [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* grad [[buffer(2)]],
    device const float* upstream [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& classes [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    int b = id / classes;
    int c = id % classes;
    
    if (b >= batch) return;
    
    int targetClass = (int)targets[b];
    float p = probs[b * classes + c];
    float t = (c == targetClass) ? 1.0f : 0.0f;
    float up = upstream[0];
    
    grad[b * classes + c] = up * (p - t) / (float)batch;
}

