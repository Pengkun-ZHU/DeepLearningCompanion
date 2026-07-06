# Lesson 06 — Matrix Multiplication in PyTorch

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra |
| **Reading Assignment** | Matrix multiplication (Chapter 2) |

This lesson connects the mathematical definition of matrix multiplication with its implementation in PyTorch.

## Objectives

By the end of this lesson you should be able to:

- explain matrix multiplication using rows and columns;
- predict the shape of a matrix multiplication result;
- distinguish matrix multiplication from element-wise operations;
- understand why matrix multiplication almost always allocates new memory;
- identify the PyTorch APIs corresponding to mathematical matrix multiplication.


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
```

and

```
mm
```

Notice that the implementation quickly dispatches into optimized backend libraries rather than performing nested loops directly.

Why?

Because matrix multiplication is one of the most heavily optimized operations in numerical computing.


## Framework Design

Suppose you're implementing

```python
A @ B
```

Would you write

```cpp
for (...)

    for (...)

        for (...)
```

inside PyTorch?

Technically you could.

But would that be competitive?

Think about:

- CPU cache
- SIMD instructions
- GPU acceleration
- Vendor libraries

We'll revisit this question when we study CUDA.


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


## Summary

Unlike the tensor operations studied previously, matrix multiplication creates new values rather than reinterpreting existing ones.

This requires:

- allocating new storage;
- performing a large number of arithmetic operations;
- using highly optimized implementations for acceptable performance.

Understanding this distinction is the first step toward understanding why deep learning frameworks invest so much effort into optimizing matrix multiplication.


## Next Lesson

**Lesson 07 — Why Matrix Multiplication Is Fast**

The mathematical algorithm is simple.

The implementation is not.

We'll investigate why a naïve triple-loop implementation is dramatically slower than modern implementations, introducing concepts such as:

- cache locality;
- memory access patterns;
- BLAS libraries;
- why PyTorch delegates matrix multiplication to highly optimized backends.

That lesson will serve as our bridge from PyTorch into CUDA and GPU programming.