# Lesson 07 — CUDA GEMM: From Naïve to Tiled

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra (Companion Extension) |
| **Reading Assignment** | CUDA Programming Guide Chapters 1–3; CUDA Programming Guide: Shared Memory |

This lesson bridges the gap between calling `torch.matmul()` and understanding how a GPU actually computes matrix multiplication, then transforms a naïve kernel into a tiled implementation using shared memory.

## Objectives

By the end of this lesson you should be able to:

- explain why cuBLAS exists instead of launching one huge kernel;
- understand the naïve CUDA GEMM execution model;
- identify threads, blocks and grids in a matrix multiplication kernel;
- write a naïve CUDA GEMM kernel and verify its correctness;
- explain what shared memory is in the CUDA memory hierarchy;
- understand why tiling dramatically reduces global memory traffic;
- implement a tiled GEMM kernel;
- recognize the pattern that underlies real-world GEMM implementations.


## Motivation

In the previous lesson we learned:

- Matrix multiplication is computationally expensive.
- PyTorch dispatches matrix multiplication to cuBLAS on NVIDIA GPUs.

The obvious question is:

> What does cuBLAS actually do?

Before studying a highly optimized implementation, we'll first understand the simplest possible CUDA implementation, identify its bottleneck, and then fix it.


## The Basic Idea

Suppose

```
C = A × B
```

where

```
A : (M, K)
B : (K, N)
C : (M, N)
```

A naïve CUDA implementation assigns **one GPU thread** to compute **one output element**.

For example:

```
C (4 × 4)

+----+----+----+----+
| T0 | T1 | T2 | T3 |
+----+----+----+----+
| T4 | T5 | T6 | T7 |
+----+----+----+----+
| .. | .. | .. | .. |
+----+----+----+----+
```

Each thread computes exactly one value of `C`.


## What Does One Thread Do?

For output element

```
C[i, j]
```

the thread performs

```
sum = 0

for k = 0 ... K-1

    sum += A[i,k] * B[k,j]
```

This is exactly the mathematical definition from the *Deep Learning* book.

CUDA changes **who performs the work**, not **what is computed**.


## Grid and Block Mapping

A common launch configuration is

```cpp
dim3 block(16,16);

dim3 grid(
    ceil(N / 16.0),
    ceil(M / 16.0)
);
```

Conceptually,

```
Grid
 ├── Block (16×16 threads)
 │     ├── Thread (0,0)
 │     ├── Thread (0,1)
 │     └── ...
 └── ...
```

Each thread determines which matrix element it owns from its block and thread indices.


## Naïve GEMM Kernel

```cpp
__global__
void naive_gemm(const float* A,
                const float* B,
                float* C,
                int M,
                int N,
                int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;

    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = sum;
}
```

### Understanding the Indexing

Each thread computes exactly one element:

```
Thread (row, col)

↓

C[row, col]
```

Inside the loop:

```
A[row, k]

×

B[k, col]
```

This is identical to the mathematical definition from the *Deep Learning* book.

### Launch Configuration

```cpp
dim3 block(16, 16);

dim3 grid(
    (N + block.x - 1) / block.x,
    (M + block.y - 1) / block.y);

naive_gemm<<<grid, block>>>(A, B, C, M, N, K);
```

The rounding ensures every output element is assigned a thread.

### Verify Correctness

Compare the CUDA result with:

- a CPU implementation; or
- `torch.matmul()`.

Numerical differences should only be due to floating-point rounding.


## The Memory Bottleneck

Although correct, the naïve kernel is also inefficient.

Suppose two neighboring threads compute

```
C[0, 0]

and

C[0, 1]
```

Both threads need the entire first row of `A`.

In the naïve kernel, they fetch it independently from global memory.

Every thread repeatedly loads:

- one row of `A`;
- one column of `B`;

from global memory.

Neighboring threads often fetch the same values independently.

This redundant memory traffic is the primary bottleneck.


## Investigation — Measuring the Bottleneck

Verify that memory access is the bottleneck by comparing your naïve kernel to cuBLAS.

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


## Shared Memory

The fix is **shared memory**: a fast, on-chip scratchpad shared by all threads in the same thread block.

The CUDA memory hierarchy is:

```
Registers        (per-thread, fastest)

↓

Shared Memory    (per-block, on-chip)

↓

L2 Cache

↓

Global Memory    (slowest, off-chip)
```

Global memory is slow.

Shared memory is much faster — but it must be managed explicitly.


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

Read the CUDA Programming Guide sections describing:

- thread hierarchy;
- blocks;
- grids;
- kernel launch syntax;
- shared memory.

Focus on understanding the execution model.

Read the NVIDIA CUTLASS library:

```
https://github.com/NVIDIA/cutlass
```

Compare its hierarchy of tiles (`ThreadblockMma`, `WarpMma`, `MmaTensorOp`) with the single-level tile you just implemented.

You don't need to understand every line.

Simply observe how many levels of tiling a production GEMM library uses.


## Exercises

### 1

Draw a 2×2 thread block computing a 2×2 output tile.


### 2

Explain why assigning one thread per output element is logically correct.


### 3

Estimate how many multiply-add operations one thread performs when multiplying a `(1024,1024)` matrix.

How many global memory reads does it perform in the naïve kernel?


### 4

Why do neighboring threads repeatedly access the same input values in the naïve kernel?


### 5

Change the block size in the naïve kernel to `(8,8)`, `(16,16)` and `(32,8)`.

Verify correctness for each configuration.


### 6

Implement the tiled kernel.

Verify its output matches

```python
torch.matmul(A, B)
```


### 7

Benchmark the naïve and tiled kernels for matrix sizes

```
256 × 256
512 × 512
1024 × 1024
```

Plot the speedup.


### 8

Explain in your own words why shared memory reduces global memory traffic.


### 9

What happens if you remove the first `__syncthreads()`?

Design an experiment to demonstrate the problem.


## Summary

A naïve CUDA GEMM assigns one thread per output element and correctly mirrors the mathematical definition.

Its bottleneck is global memory: neighboring threads repeatedly fetch the same input values independently.

Tiled GEMM replaces independent per-thread memory loads with cooperative block-level loading into shared memory.

Each value fetched from global memory is reused `TILE` times.

This dramatically reduces memory bandwidth requirements and improves throughput.

Real GEMM libraries extend this principle to multiple levels of the memory hierarchy.


## Next Lesson

**Lesson 08 — Norms, Special Matrices, and Decompositions**

We return to the *Deep Learning* book to cover the remaining sections of Chapter 2:

- vector and matrix norms;
- special matrix types (symmetric, orthogonal, diagonal);
- eigendecomposition;
- singular value decomposition;
- their PyTorch counterparts.
