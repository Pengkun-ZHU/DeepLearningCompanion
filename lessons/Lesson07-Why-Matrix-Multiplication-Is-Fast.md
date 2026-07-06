# Lesson 07 — Why Matrix Multiplication Is Fast

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra (Companion Extension) |
| **Reading Assignment** | Review the matrix multiplication section from Chapter 2 |

The *Deep Learning* book explains what matrix multiplication is.

This lesson explores how modern deep learning frameworks make it fast.

## Objectives

By the end of this lesson you should be able to:

- explain why a naïve matrix multiplication is slow;
- understand the importance of memory access patterns;
- appreciate the role of CPU caches;
- understand why PyTorch delegates matrix multiplication to optimized libraries;
- see why GPUs are particularly effective for matrix multiplication.


## Motivation

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

Why?


## Investigation 1 — Measuring Performance

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

Now compare this with a simple Python implementation.

Even though both perform exactly the same mathematical operation, PyTorch is often **hundreds or thousands of times faster**.

Where does that speed come from?


## A Common Misconception

Many newcomers assume

> PyTorch is fast because it is written in C++.

That is only part of the story.

Replacing Python loops with C++ loops helps.

It does **not** explain performance differences of hundreds or thousands of times.

Something much more important is happening.


## The Memory Problem

Consider this loop.

```cpp
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
            C[i][j] += A[i][k] * B[k][j];
```

Accesses to `A` look like

```
A[0][0]
A[0][1]
A[0][2]
A[0][3]
...
```

These values are adjacent in memory.

The CPU likes this.

Accesses to `B` look like

```
B[0][0]
B[1][0]
B[2][0]
B[3][0]
...
```

These values are **not** adjacent in row-major storage.

Every iteration jumps to a different location in memory.

This is much less efficient.


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


## Why Loop Order Matters

Consider these two loop orders.

```cpp
i
j
k
```

versus

```cpp
i
k
j
```

Both produce exactly the same mathematical result.

However, they access memory differently.

One may repeatedly reuse cached values.

The other may constantly reload data from memory.

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


## Investigation 2

Open a system monitor while running

```python
A @ B
```

Observe:

- CPU usage;
- memory usage.

Notice that CPU utilization becomes very high, while memory usage changes relatively little.

This suggests that the computation is dominated by arithmetic rather than memory allocation.


## Companion Insight

The previous lessons taught us that operations like

- `transpose()`
- `view()`
- `expand()`

are fast because they only modify metadata.

Matrix multiplication is fundamentally different.

Every output element must be computed.

Since the computation cannot be avoided, the only remaining question is:

> How efficiently can we perform it?

This shifts the focus from tensor metadata to processor architecture.


## Source Reading

Search the PyTorch source tree for

```
mm

matmul
```

Follow the call chain.

Notice that PyTorch quickly dispatches to optimized backend libraries.

It does **not** perform the triple loop directly.

This is intentional.

Writing a competitive matrix multiplication kernel is an enormous engineering challenge that has been refined over decades.


## Framework Design

Imagine you are implementing your own deep learning framework.

Would you:

- write your own matrix multiplication kernel;
- call an existing optimized library?

PyTorch chooses the second approach on most platforms.

This allows it to benefit from decades of optimization work without maintaining separate implementations for every CPU architecture.


## Exercises

### 1

Write a naïve matrix multiplication using three Python loops.

Compare its runtime with

```python
torch.matmul()
```

How large is the difference?


### 2

Read about BLAS.

What problem was BLAS originally designed to solve?

Why is it still widely used today?


### 3

Research cache locality.

Explain, in your own words, why sequential memory access is generally faster than scattered access.


### 4

Suppose you have two correct matrix multiplication implementations.

One is twice as fast as the other.

List three possible reasons for the performance difference that have nothing to do with the mathematical algorithm itself.


## Summary

The mathematical definition of matrix multiplication is simple.

The implementation is not.

Efficient implementations depend on:

- cache locality;
- memory access patterns;
- blocking (tiling);
- vector instructions;
- highly optimized numerical libraries.

Understanding these ideas prepares us to study GPU implementations, where the same principles appear again in a different form.


## Next Lesson

**Lesson 08 — BLAS, cuBLAS and How PyTorch Executes `matmul()`**

We'll trace the execution of

```python
A @ B
```

through PyTorch and discover:

- what BLAS is;
- why every deep learning framework relies on it;
- how PyTorch dispatches to different backends;
- where cuBLAS enters the picture on NVIDIA GPUs.