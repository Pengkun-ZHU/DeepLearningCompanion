# Lesson 09 --- From BLAS to CUDA Kernels

  ----------------------------------- -----------------------------------
  **Chapter**                         2 --- Linear Algebra (Companion
                                      Extension)

  **Reading Assignment**              Revisit matrix multiplication; skim
                                      CUDA Programming Guide Ch. 1--2
  ----------------------------------- -----------------------------------

This lesson bridges the gap between calling `torch.matmul()` and
understanding how a GPU actually computes matrix multiplication.

## Objectives

By the end of this lesson you should be able to:

-   explain why cuBLAS exists instead of launching one huge kernel;
-   understand the naïve CUDA GEMM execution model;
-   identify threads, blocks and grids in a matrix multiplication
    kernel;
-   see why memory access dominates performance;
-   prepare for implementing a naïve GEMM in the next lesson.

## Motivation

So far we've learned:

-   Matrix multiplication is computationally expensive.
-   PyTorch dispatches matrix multiplication to cuBLAS on NVIDIA GPUs.

The obvious question is:

> What does cuBLAS actually do?

Before studying a highly optimized implementation, we'll first
understand the simplest possible CUDA implementation.

## The Basic Idea

Suppose

``` text
C = A × B
```

where

``` text
A : (M, K)
B : (K, N)
C : (M, N)
```

A naïve CUDA implementation assigns **one GPU thread** to compute **one
output element**.

For example:

``` text
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

``` text
C[i, j]
```

the thread performs

``` text
sum = 0

for k = 0 ... K-1

    sum += A[i,k] * B[k,j]
```

This is exactly the mathematical definition from the *Deep Learning*
book.

CUDA changes **who performs the work**, not **what is computed**.

## Grid and Block Mapping

A common launch configuration is

``` cpp
dim3 block(16,16);

dim3 grid(
    ceil(N / 16.0),
    ceil(M / 16.0)
);
```

Conceptually,

``` text
Grid
 ├── Block (16×16 threads)
 │     ├── Thread (0,0)
 │     ├── Thread (0,1)
 │     └── ...
 └── ...
```

Each thread determines which matrix element it owns from its block and
thread indices.

## The First Performance Problem

The naïve kernel is correct.

It is also inefficient.

Neighboring threads repeatedly read the same rows of `A` and columns of
`B` from global memory.

Those repeated memory accesses dominate execution time.

This is why high-performance GEMM kernels use **tiling** and **shared
memory**, topics we'll study next.

## Source Reading

No PyTorch source reading for this lesson.

Instead, read the CUDA Programming Guide sections describing:

-   thread hierarchy;
-   blocks;
-   grids;
-   kernel launch syntax.

Focus on understanding the execution model rather than every language
feature.

## Exercises

1.  Draw a 2×2 thread block computing a 2×2 output tile.

2.  Explain why assigning one thread per output element is logically
    correct.

3.  Estimate how many multiply-add operations one thread performs when
    multiplying a `(1024,1024)` matrix.

4.  Why do neighboring threads repeatedly access the same input values?

## Summary

A naïve CUDA GEMM mirrors the mathematical definition of matrix
multiplication.

Each thread computes one output element by iterating over the shared
dimension.

Although correct, this implementation performs excessive global memory
accesses, leaving substantial performance on the table.

## Next Lesson

**Lesson 10 --- Implementing a Naïve CUDA GEMM**

We'll write our first CUDA kernel, launch it from host code, verify
correctness, and benchmark it against CPU execution before introducing
shared-memory tiling.
