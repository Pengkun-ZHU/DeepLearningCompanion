# Lesson 11 — Shared Memory and Tiled GEMM

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra (Companion Extension) |
| **Reading Assignment** | CUDA Programming Guide: Shared Memory |

This lesson transforms the naïve GEMM from Lesson 10 into a tiled implementation using shared memory.

## Objectives

By the end of this lesson you should be able to:

- explain what shared memory is in the CUDA memory hierarchy;
- understand why tiling dramatically reduces global memory traffic;
- implement a tiled GEMM kernel;
- measure the performance improvement relative to the naïve kernel;
- recognize the pattern that underlies real-world GEMM implementations.


## Motivation

The naïve kernel from Lesson 10 is correct.

It is also inefficient.

Suppose two neighboring threads compute

```
C[0, 0]

and

C[0, 1]
```

Both threads need the entire first row of `A`.

In the naïve kernel, they fetch it independently from global memory.

Global memory is slow.

This redundant fetching wastes bandwidth and limits throughput.

The fix is **shared memory**: a fast, on-chip scratchpad shared by all threads in the same thread block.


## The Tiling Idea

Instead of each thread loading data independently, the entire block cooperates.

The matrices are divided into **tiles**.

```
A               B

+---+---+       +---+---+
|   |   |       |   |   |
+---+---+       +---+---+
|   |   |       |   |   |
+---+---+       +---+---+
```

For each tile along the shared dimension:

1. Every thread in the block loads one element of the `A` tile into shared memory.
2. Every thread in the block loads one element of the `B` tile into shared memory.
3. All threads synchronize.
4. Each thread accumulates its partial dot product using shared memory values.
5. Repeat for the next tile.


## Investigation — Measuring the Bottleneck

Before implementing tiling, verify that memory access is the bottleneck in the naïve kernel.

```python
import torch
import time

M, N, K = 1024, 1024, 1024

A = torch.randn(M, K, device='cuda', dtype=torch.float32)
B = torch.randn(K, N, device='cuda', dtype=torch.float32)

# Warm-up
for _ in range(10):
    C = torch.matmul(A, B)
torch.cuda.synchronize()

# Time PyTorch (cuBLAS)
start = time.perf_counter()
for _ in range(100):
    C = torch.matmul(A, B)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100

print(f"cuBLAS: {elapsed*1000:.3f} ms")
```

cuBLAS uses tiling internally.

Your naïve kernel will be significantly slower.


## Tiled GEMM Kernel

```cpp
#define TILE 16

__global__
void tiled_gemm(const float* A,
                const float* B,
                float* C,
                int M, int N, int K)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {

        // Load tiles cooperatively
        if (row < M && t * TILE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Accumulate partial dot product
        for (int k = 0; k < TILE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### Key Changes from the Naïve Kernel

| Naïve | Tiled |
|---|---|
| Each thread loads its own data | Block cooperates to load tiles |
| Global memory access per multiply | Shared memory access per multiply |
| Redundant fetches | Values reused TILE times |


## Why Two `__syncthreads()` Calls?

The first `__syncthreads()` ensures that all threads have finished loading into shared memory before any thread reads from it.

The second `__syncthreads()` ensures that all threads have finished reading shared memory before the next tile overwrites it.

Removing either barrier introduces race conditions.


## Companion Insight

Tiling is a memory hierarchy optimization.

The key idea is:

> Load data from slow global memory once, reuse it many times from fast shared memory.

This pattern appears throughout high-performance computing:

- CPU caches do the same thing automatically.
- Shared memory in CUDA requires explicit management.
- Real GEMM libraries (cuBLAS, CUTLASS) use multi-level tiling across the full memory hierarchy.

Understanding tiled GEMM reveals why matrix multiplication on modern hardware is not just mathematically intensive but also carefully engineered to match the memory hierarchy.


## Source Reading

Read the NVIDIA CUTLASS library:

```
https://github.com/NVIDIA/cutlass
```

Compare its hierarchy of tiles (`ThreadblockMma`, `WarpMma`, `MmaTensorOp`) with the single-level tile you just implemented.

You don't need to understand every line.

Simply observe how many levels of tiling a production GEMM library uses.


## Exercises

### 1

Implement the tiled kernel.

Verify its output matches

```python
torch.matmul(A, B)
```

### 2

Benchmark the naïve and tiled kernels for matrix sizes

```
256 × 256
512 × 512
1024 × 1024
```

Plot the speedup.

### 3

Explain in your own words why shared memory reduces global memory traffic.

### 4

What happens if you remove the first `__syncthreads()`?

Design an experiment to demonstrate the problem.


## Summary

Tiled GEMM replaces independent per-thread memory loads with cooperative block-level loading into shared memory.

Each value fetched from global memory is reused `TILE` times.

This dramatically reduces memory bandwidth requirements and improves throughput.

Real GEMM libraries extend this principle to multiple levels of the memory hierarchy.


## Next Lesson

**Lesson 12 — Norms, Special Matrices, and Decompositions**

We return to the *Deep Learning* book to cover the remaining sections of Chapter 2:

- vector and matrix norms;
- special matrix types (symmetric, orthogonal, diagonal);
- eigendecomposition;
- singular value decomposition;
- their PyTorch counterparts.
