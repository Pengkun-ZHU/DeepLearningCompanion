# Lesson 10 --- Implementing a Naïve CUDA GEMM

  ------------------------ --------------------------------------------
  **Chapter**              2 --- Linear Algebra (Companion Extension)
  **Reading Assignment**   CUDA Programming Guide Chapters 1--3
  ------------------------ --------------------------------------------

This lesson implements the simplest correct CUDA matrix multiplication
kernel.

## Objectives

-   Write your first CUDA kernel.
-   Map one thread to one output element.
-   Launch a 2D grid of thread blocks.
-   Verify correctness against a CPU/PyTorch result.
-   Recognize the performance limitations of the naïve approach.

## Motivation

In Lesson 09 we saw the execution model:

> One thread computes one element of the output matrix.

Today we'll turn that idea into working CUDA code.

## Kernel

``` cpp
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

## Understanding the Indexing

Each thread computes exactly one element:

``` text
Thread (row, col)

↓

C[row, col]
```

Inside the loop:

``` text
A[row, k]

×

B[k, col]
```

This is identical to the mathematical definition from the *Deep
Learning* book.

## Launch Configuration

``` cpp
dim3 block(16, 16);

dim3 grid(
    (N + block.x - 1) / block.x,
    (M + block.y - 1) / block.y);

naive_gemm<<<grid, block>>>(A, B, C, M, N, K);
```

The rounding ensures every output element is assigned a thread.

## Verify Correctness

Compare the CUDA result with:

-   a CPU implementation; or
-   `torch.matmul()`.

Numerical differences should only be due to floating-point rounding.

## Performance Discussion

Although correct, every thread repeatedly loads:

-   one row of `A`;
-   one column of `B`;

from global memory.

Neighboring threads often fetch the same values independently.

This redundant memory traffic is the primary bottleneck.

## Source Reading

No PyTorch source reading.

Read instead:

-   CUDA Programming Guide: Kernel launches.
-   CUDA Programming Guide: Built-in thread variables.

Relate:

``` cpp
threadIdx
blockIdx
blockDim
gridDim
```

to the matrix coordinates used in the kernel.

## Exercises

1.  Change the block size to `(8,8)`, `(16,16)` and `(32,8)`. Verify
    correctness.

2.  Draw the mapping between one thread block and the corresponding
    output tile.

3.  Explain why the boundary check is necessary.

4.  Estimate how many global memory reads one thread performs when
    computing a single output element.

## Summary

The naïve GEMM kernel is mathematically correct and easy to understand.

However, it repeatedly loads the same input values from global memory,
making it memory-inefficient.

The next lesson introduces **shared memory**, allowing threads in the
same block to cooperate and reuse data instead of repeatedly fetching it
from global memory.

## Next Lesson

**Lesson 11 --- Shared Memory and Tiled GEMM**

We'll transform the naïve kernel into a tiled implementation and measure
the performance improvement.
