# Lesson 08 --- BLAS, cuBLAS and How PyTorch Executes `matmul()`
| | |
|---|---|
| **Chapter** | 2 — Linear Algebra (Companion Extension) |
| **Reading Assignment** | Review the matrix multiplication section from Chapter 2 |

This lesson follows the execution path of `A @ B` inside PyTorch and
introduces the optimized libraries that actually perform the
computation.

## Objectives

By the end of this lesson you should be able to:

-   explain what BLAS is;
-   distinguish BLAS from cuBLAS;
-   understand why PyTorch delegates matrix multiplication to backend
    libraries;
-   identify the high-level execution path of `torch.matmul()`;
-   appreciate why framework developers rarely write GEMM kernels from
    scratch.

## Motivation

When you write

``` python
C = A @ B
```

it looks like PyTorch performs the multiplication itself.

In reality, PyTorch usually delegates the work to highly optimized
numerical libraries that have been refined over decades.

Understanding this delegation is essential for understanding the
architecture of modern deep learning frameworks.

## What is BLAS?

**BLAS** stands for **Basic Linear Algebra Subprograms**.

It is a standard interface for common linear algebra routines such as:

-   vector addition;
-   dot products;
-   matrix-vector multiplication;
-   matrix-matrix multiplication (GEMM).

Rather than implementing these algorithms independently, scientific
software typically calls an optimized BLAS implementation.

Common implementations include:

-   OpenBLAS
-   Intel MKL
-   Apple Accelerate

Although the interface is standardized, each implementation is heavily
optimized for its target hardware.

## What is cuBLAS?

On NVIDIA GPUs, the corresponding library is **cuBLAS**.

Conceptually,

``` text
CPU  -> BLAS implementation

GPU  -> cuBLAS
```

The API is similar, but the implementation is designed for massively
parallel GPU execution.

## Investigation

Run:

``` python
import torch

A = torch.randn(1024, 1024)
B = torch.randn(1024, 1024)

C = A @ B

print(C.shape)
```

If CUDA is available:

``` python
A = A.cuda()
B = B.cuda()

C = A @ B
```

The Python code is almost identical.

The backend performing the computation is completely different.

## How PyTorch Executes `matmul()`

Conceptually:

``` text
Python

A @ B

↓

torch.matmul()

↓

ATen

↓

Dispatcher

↓

CPU backend        CUDA backend

↓

BLAS              cuBLAS

↓

Result Tensor
```

Most users never interact directly with BLAS or cuBLAS.

PyTorch selects the appropriate backend automatically.

## Source Reading

Search the PyTorch source tree for:

``` text
matmul

mm

addmm
```

Observe how these operators are dispatched instead of implemented with
explicit triple loops.

## Framework Design

Imagine maintaining PyTorch.

Would you rather:

-   maintain a high-performance GEMM implementation for every CPU and
    GPU architecture;

or

-   integrate with mature libraries maintained by hardware vendors and
    HPC experts?

PyTorch largely chooses the second approach.

## Exercises

1.  Research one BLAS implementation (OpenBLAS, MKL or Accelerate).
    Summarize its target platform.

2.  Compare CPU and GPU execution times for a large matrix
    multiplication if you have CUDA available.

3.  Explain why `transpose()` does not require BLAS while `matmul()`
    almost always does.

## Summary

PyTorch is responsible for presenting a unified tensor API.

It is **not** responsible for implementing every numerical kernel from
scratch.

For computationally intensive operations such as matrix multiplication,
PyTorch delegates to highly optimized backend libraries:

-   BLAS on CPUs.
-   cuBLAS on NVIDIA GPUs.

This layered architecture allows PyTorch to remain portable while
benefiting from decades of optimization work.

## Next Lesson

**Lesson 09 --- From BLAS to CUDA Kernels**

We'll look beneath cuBLAS and study how a naïve CUDA matrix
multiplication kernel evolves into a tiled implementation, connecting
GPU architecture with the performance principles introduced in Lessons
6 to 8.
