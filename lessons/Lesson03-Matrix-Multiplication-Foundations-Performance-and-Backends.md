# Lesson 06 — Matrix Multiplication: Foundations, Performance, and Backends

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra |
| **Reading Assignment** | Matrix multiplication (Chapter 2) |

This lesson covers what matrix multiplication is, why naïve implementations are slow, and how PyTorch delegates the work to highly optimized backend libraries.

## Objectives

By the end of this lesson you should be able to:

- explain matrix multiplication using rows and columns;
- predict the shape of a matrix multiplication result;
- distinguish matrix multiplication from element-wise operations;
- understand why matrix multiplication almost always allocates new memory;
- identify the PyTorch APIs corresponding to mathematical matrix multiplication;
- explain why a naïve matrix multiplication is slow;
- understand the importance of memory access patterns and CPU caches;
- appreciate the role of blocking (tiling);
- explain what BLAS and cuBLAS are;
- understand why PyTorch delegates matrix multiplication to backend libraries.

## Before We Start

For readers whose linear algebra skills may have grown rusty, a review of Professor Gilbert Strang’s canonical Linear Algebra course at MIT is highly recommended. This resource offers more than a simple refresher; it deeply familiarizes the reader with the row and column perspectives of matrix multiplication—a foundational intuition that plays a vital role in mastering modern ML concepts.

## Motivation

We've spent the previous lessons studying operations like

```python
transpose()

view()

reshape()

expand()
```

One thing they have in common is that **they usually do not compute new values**.

Instead, they reinterpret existing storage.

Matrix multiplication is completely different.

```python
C = A @ B
```

Every element of `C` must be computed.

There is no way to obtain the result simply by changing tensor metadata.

Since the computation cannot be avoided, the only remaining question is:

> How efficiently can we perform it?

This shifts the focus from tensor metadata to processor architecture — and ultimately to specialized libraries.


## Book Recap

Suppose

```
A

shape = (2,3)
```

```
B

shape = (3,4)
```

The product

```
C = AB
```

has shape

```
(2,4)
```

Notice that

- the inner dimensions (`3`) must match;
- the outer dimensions determine the output shape.

This is the fundamental compatibility rule for matrix multiplication.


## Investigation 1 — Matrix Multiplication

```python
import torch

A = torch.tensor([
    [1,2,3],
    [4,5,6]
])

B = torch.tensor([
    [1,2],
    [3,4],
    [5,6]
])

C = A @ B

print(C)
print(C.shape)
```

Output

```
tensor([[22, 28],
        [49, 64]])

torch.Size([2,2])
```

Before verifying the result, compute the first element by hand.

```
22

=

1×1

+

2×3

+

3×5
```

Each output element is the **dot product** of one row of `A` and one column of `B`.


## Investigation 2 — New Storage

Compare the storage pointers.

```python
print(A.untyped_storage().data_ptr())
print(B.untyped_storage().data_ptr())
print(C.untyped_storage().data_ptr())
```

Notice that

```
C
```

owns **different storage**.

Unlike `transpose()` or `view()`, matrix multiplication must allocate memory for newly computed values.


## Element-wise vs Matrix Multiplication

These two expressions look similar.

```python
A * B
```

```python
A @ B
```

They perform completely different operations.

### Element-wise multiplication

```
1 2

*

5 6

=

5 12
```

Each element interacts only with the element in the same position.

### Matrix multiplication

Each output element depends on an **entire row** and an **entire column**.

This difference makes matrix multiplication computationally much more expensive.


## Investigation 3 — Counting Operations

Suppose

```
A

shape = (1000,500)
```

```
B

shape = (500,800)
```

The output shape is

```
(1000,800)
```

How many output elements are produced?

```
1000 × 800

=

800,000
```

Each output element requires

```
500

multiplications

+

499 additions
```

The total work is enormous.

This is why matrix multiplication dominates the runtime of many deep learning models.


## Why Naïve Implementations Are Slow

Suppose we implement matrix multiplication ourselves.

```python
def matmul(A, B):
    M, K = A.shape
    K2, N = B.shape

    C = torch.zeros(M, N)

    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]

    return C
```

Mathematically, this is perfectly correct.

Unfortunately, it is also **extremely slow**.


## Investigation 4 — Measuring Performance

```python
import time
import torch

N = 1024

A = torch.randn(N, N)
B = torch.randn(N, N)

t0 = time.time()

C = A @ B

print(time.time() - t0)
```

Compare this with a simple Python implementation.

Even though both perform exactly the same mathematical operation, PyTorch is often **hundreds or thousands of times faster**.

A common misconception is that the speed comes from PyTorch being written in C++.

That is only part of the story.

Replacing Python loops with C++ loops helps, but it does **not** explain differences of hundreds or thousands of times.

Something much more important is happening.


## CPU Cache

Modern CPUs are much faster than main memory.

To bridge this gap, CPUs contain several levels of cache.

```
Registers

↓

L1 Cache

↓

L2 Cache

↓

L3 Cache

↓

Main Memory
```

Reading data already in cache is dramatically faster than fetching it from RAM.

Efficient matrix multiplication is largely about **keeping useful data in the cache**.


## Why Loop Order Matters? The Memory Problem

Consider these two loop orders.

```cpp
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
            C[i][j] += A[i][k] * B[k][j]; 
```

`C[i][j]`: Spatial locality is perfect because it doesn't change based on k. It can stay in a CPU register.  
`A[i][k]`: Sequential access (`A[i][0]`, `A[i][1]`, `A[i][2]`). This is very cache-friendly.  
`B[k][j]`: Disaster. As `k` increments, you are accessing `B[0][j]`, `B[1][j]`, `B[2][j]`. You are jumping a distance of N elements in memory on every single iteration. This causes constant cache misses because the CPU is pulling down whole cache lines but only using one element from each before throwing it away.

versus

```cpp
for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
            C[i][j] += A[i][k] * B[k][j]; 
```

`A[i][k]`: Independent of j. It stays completely constant for the entire duration of the inner loop and can be cached in a single register.  
`B[k][j]`: Sequential access (`B[k][0]`, `B[k][1]`, `B[k][2]`). Perfect spatial locality, The CPU loads a cache line once and instantly processes the next 8–16 elements directly from the cache.  
`C[i][j]`: Sequential access (`C[i][0]`, `C[i][1]`, `C[i][2]`). Also perfect spatial locality!

On large matrices, this difference can dramatically affect performance.


## Blocking (Tiling)

A better strategy is to divide the matrices into smaller blocks.

Instead of multiplying

```
1024 × 1024
```

all at once,

we compute many smaller multiplications.

```
┌────┬────┐
│    │    │
├────┼────┤
│    │    │
└────┴────┘
```

Each block fits into cache.

The same data can be reused many times before it is evicted.

This technique is called **blocking** (or **tiling**).

It is one of the most important optimization techniques in high-performance computing.

The same idea appears on GPUs using shared memory — a topic covered in the next lesson.


## What is BLAS?

**BLAS** stands for **Basic Linear Algebra Subprograms**.

It is a standard interface for common linear algebra routines such as:

- vector addition;
- dot products;
- matrix-vector multiplication;
- matrix-matrix multiplication (GEMM).

Rather than implementing these algorithms independently, scientific software typically calls an optimized BLAS implementation.

Common implementations include:

- OpenBLAS
- Intel MKL
- Apple Accelerate

Although the interface is standardized, each implementation is heavily optimized for its target hardware.


## What is cuBLAS?

On NVIDIA GPUs, the corresponding library is **cuBLAS**.

Conceptually,

```
CPU  -> BLAS implementation

GPU  -> cuBLAS
```

The API is similar, but the implementation is designed for massively parallel GPU execution.


## Investigation 5 — CPU vs GPU Dispatch

```python
import torch

A = torch.randn(1024, 1024)
B = torch.randn(1024, 1024)

C = A @ B

print(C.shape)
```

If CUDA is available:

```python
A = A.cuda()
B = B.cuda()

C = A @ B
```

The Python code is almost identical.

The backend performing the computation is completely different.


## How PyTorch Executes `matmul()`

Conceptually:

```
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


## Companion Insight

Matrix multiplication introduces an important distinction.

Some tensor operations are **metadata operations**.

Examples include:

- `transpose()`
- `view()`
- `expand()`
- slicing

These are often O(1).

Matrix multiplication is a **computational operation**.

It creates entirely new values.

Its cost grows roughly with

```
rows × columns × shared_dimension
```

Efficient implementations depend on:

- cache locality;
- memory access patterns;
- blocking (tiling);
- vector instructions;
- highly optimized numerical libraries (BLAS on CPU, cuBLAS on GPU).

As models become larger, this operation quickly becomes the dominant computational expense.


## PyTorch APIs

PyTorch provides several interfaces.

```python
torch.mm(A, B)
```

Matrix × Matrix


```python
torch.matmul(A, B)
```

General matrix multiplication.

Works with vectors, matrices and batches.


```python
A @ B
```

Python syntax for `torch.matmul()`.

For most user code,

```python
@
```

is the preferred notation.


## Source Reading

Search the PyTorch source tree for

```
matmul

mm

addmm
```

Follow the call chain.

Notice that PyTorch quickly dispatches to optimized backend libraries rather than performing nested loops directly.

This is intentional.

Writing a competitive matrix multiplication kernel is an enormous engineering challenge that has been refined over decades.


## Framework Design

Imagine you are implementing your own deep learning framework.

Would you:

- write your own matrix multiplication kernel for every CPU and GPU architecture;
- call existing optimized libraries maintained by hardware vendors and HPC experts?

PyTorch largely chooses the second approach.

This allows it to benefit from decades of optimization work without maintaining separate implementations for every architecture.


## Exercises

### 1

Predict the output shape.

```
(5,3)

@

(3,7)
```


### 2

Which of the following are valid?

```
(3,4)

@

(4,5)
```

```
(3,4)

@

(3,4)
```

```
(5,2)

@

(2,1)
```

Explain your reasoning.


### 3

Verify experimentally that

```python
A @ B
```

allocates new storage.

### 4

Estimate the number of multiplications required for

```
(512,1024)

@

(1024,2048)
```

Why does this motivate GPU acceleration?


### 5

Write a naïve matrix multiplication using three Python loops.

Compare its runtime with

```python
torch.matmul()
```

How large is the difference?


### 6

Research cache locality.

Explain, in your own words, why sequential memory access is generally faster than scattered access.


### 7

Research one BLAS implementation (OpenBLAS, MKL or Accelerate).

Summarize its target platform.


### 8

Compare CPU and GPU execution times for a large matrix multiplication if you have CUDA available.


### 9

Explain why `transpose()` does not require BLAS while `matmul()` almost always does.


## Summary

Unlike the tensor operations studied previously, matrix multiplication creates new values rather than reinterpreting existing ones.

This requires:

- allocating new storage;
- performing a large number of arithmetic operations;
- using highly optimized implementations for acceptable performance.

Efficient implementations exploit cache locality, blocking (tiling), vector instructions, and — through BLAS and cuBLAS — decades of vendor optimization work.

PyTorch exposes a single unified `@` operator that automatically selects the right backend for the current device.


## Next Lesson

**Lesson 07 — CUDA GEMM: From Naïve to Tiled**

We'll look beneath cuBLAS and study how GPU hardware executes matrix multiplication.

Starting from a simple kernel where each thread computes one output element, we'll identify the memory bottleneck and fix it using shared-memory tiling — the same technique that underlies real GEMM libraries.
