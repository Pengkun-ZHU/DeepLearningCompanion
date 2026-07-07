# Lesson 02 — Tensor Indexing and Broadcasting

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra |
| **Reading Assignment** | Continue Chapter 2 (Operations on Tensors) |

This lesson assumes you have completed Lesson 01 and are familiar with tensor storage, strides, and views.

## Objectives

By the end of this lesson you should be able to:

- explain how tensor indexing works internally;
- derive the address calculation formula used by PyTorch;
- explain PyTorch's broadcasting rules;
- predict the result shape of a broadcasted operation;
- understand why broadcasting usually does not allocate memory;
- relate both indexing and broadcasting to tensor strides.


## Motivation

Two closely related questions arise from the concepts in Lesson 01.

**Question 1.** What does PyTorch actually compute when you write

```python
x[2, 1]
```

Inside PyTorch, that expression eventually becomes something like

```cpp
*(base_pointer + ?)
```

What is the `?`

**Question 2.** How can two tensors with different shapes be added?

```python
x = torch.arange(12).reshape(3, 4)   # shape (3, 4)

b = torch.tensor([100, 200, 300, 400])  # shape (4,)

y = x + b                               # shape (3, 4)
```

Both answers rely on the same idea: **strides**.


## Part 1 — Address Calculation

### Looking at Strides

```python
import torch

x = torch.arange(12).reshape(3, 4)

print(x)

print("shape :", x.shape)
print("stride:", x.stride())
```

Output

```
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

shape  = (3, 4)

stride = (4, 1)
```

Suppose we ask for `x[2, 1]`. The answer is `9`.

How does PyTorch locate that value?

### From Indices to Storage

The underlying storage is simply

```
Index

 0  1  2  3  4  5  6  7  8  9 10 11

Value

 0  1  2  3  4  5  6  7  8  9 10 11
```

The tensor is merely a *view* of this storage.

The question is:

> Which storage index corresponds to `(2, 1)`?

### Deriving the Formula

The stride tells us

```
stride = (4, 1)
```

Moving

- one row advances **4** storage elements.
- one column advances **1** storage element.

Therefore

```
row = 2     →   2 × 4 = 8

column = 1  →   1 × 1 = 1

Total offset = 8 + 1 = 9
```

`x[2, 1]` → `storage[9]` → `9`. ✓

### The General Formula

For an N-dimensional tensor:

```
offset = Σ  index[d] × stride[d]
```

If the tensor begins in the middle of a storage:

```
offset += storage_offset
```

Finally:

```
address = base_pointer + offset
```

This is the essential indexing algorithm used by tensor libraries.

### Companion Insight

One surprising observation: **shape never appears in the address calculation**.

Shape is used only for:

- validating indices;
- determining tensor dimensions;
- iteration.

Once the indices are known, only stride and storage offset are needed to compute the address.

This explains why transposing a tensor is cheap.

Transpose changes the stride. The indexing algorithm remains exactly the same.

### Investigation — Storage Offset

```python
x = torch.arange(12).reshape(3, 4)

y = x[1:]

print(y)

print("storage_offset:", y.storage_offset())
print("stride:", y.stride())
```

`y` starts at the second row, so `storage_offset = 4`.

No memory was copied. Only the metadata changed.

### Verify the Formula

```python
x = torch.arange(24).reshape(2, 3, 4)

print(x.shape)
print(x.stride())
```

Before indexing, predict the storage offset for `x[1, 2, 3]` using only shape, stride, and storage offset.

Then verify your answer in Python.


## Part 2 — Broadcasting

### Broadcasting Rules

Consider

```python
x = torch.arange(12).reshape(3, 4)   # shape (3, 4)

b = torch.tensor([100, 200, 300, 400])  # shape (4,)

y = x + b
```

PyTorch compares dimensions from **right to left**.

Two dimensions are compatible if they are:

- equal; or
- one of them is `1`.

```
(3, 4)
(1, 4)    ← b is prepended with 1

3 vs 1  ✓
4 vs 4  ✓

Result: (3, 4)
```

### Investigation 1

```python
import torch

x = torch.arange(12).reshape(3, 4)

b = torch.tensor([100, 200, 300, 400])

y = x + b

print(y.shape)

print(b.shape)
```

Notice that `b` still has shape `(4,)`.

PyTorch did **not** permanently convert it into `(3, 4)`.

### The Secret: Zero Strides

Broadcasting does not duplicate data.

Instead, PyTorch creates a temporary **view** that behaves *as if* the data had been repeated.

The underlying values are still stored only once.

A normal tensor might have

```
stride  (4, 1)
```

An expanded tensor has

```
(0, 1)
```

A stride of zero means:

> Moving along that dimension does **not** move in memory.

Every row points to the same four values.

Conceptually:

```
100 200 300 400
100 200 300 400    ← illusion
100 200 300 400    ← illusion
```

Physically:

```
100 200 300 400
```

The repeated rows are an illusion created entirely by metadata.

### Investigation 2 — `expand()`

```python
b = torch.tensor([100, 200, 300, 400])

e = b.expand(3, 4)

print(e)

print(e.stride())

print(e.is_contiguous())
```

Observe:

- no new values were created;
- the expanded tensor is not contiguous;
- the first stride is `0`.

### Investigation 3 — Verify Shared Storage

```python
b = torch.tensor([1, 2, 3])

e = b.expand(5, 3)

print(e.stride())

print(e.storage().data_ptr())

print(b.storage().data_ptr())
```

Verify that both tensors share storage and only the metadata differs.


## The Unified Theme

Across this lesson and the previous one, the same design principle appears repeatedly:

| Operation | What changes? | Memory allocated? |
|---|---|---|
| `transpose()` | strides | No |
| slicing | stride, storage offset | No |
| `view()` | shape, stride | No |
| `expand()` / broadcast | stride (zero) | No |
| `reshape()` | shape, stride | Only if required |
| `.contiguous()` | — | Yes |

**PyTorch first asks: can this be expressed by changing metadata alone?**

If yes, the operation is essentially free.


## Source Reading

Open

```
aten/src/ATen/ExpandUtils.cpp
```

Browse the implementation of `expand()`.

Notice how the function computes new sizes and new strides rather than allocating a larger tensor.

Identify where the expanded stride (including zero strides) is constructed.

You don't need to understand every line.

Simply confirm where the zero stride is introduced.


## Framework Design

Suppose you were implementing a tensor library from scratch.

For indexing, would you store stride explicitly, or recompute it from shape each time?

For broadcasting, would you allocate expanded tensors, or implement it via zero strides?

Consider the performance implications for

- memory usage;
- cache efficiency;
- GPU workloads.


## Exercises

### 1

For each tensor below, compute the storage offset of the indexed element **by hand** before running the code.

```python
x = torch.arange(12).reshape(3, 4)

x[2, 3]

x[1, 0]

x[0, 2]
```


### 2

Repeat Exercise 1 after

```python
x = x.t()
```

Notice that only the stride changes. The indexing algorithm remains identical.


### 4

Predict whether each pair of shapes can be broadcast.

```
(3, 4)   +  (4,)
(5, 1)   +  (5, 7)
(2, 3)   +  (3, 2)
(8, 1, 6)  +  (7, 6)
```

Explain your reasoning.



### 5

Can an expanded tensor be modified in place?

```python
e.add_(1)
```

What happens? Why?


### 6

Without referring to the documentation, explain broadcasting using only the concepts introduced in Lessons 1–2. Do **not** use the word "copy."


## Summary

Tensor indexing converts logical indices into a memory address using a simple formula:

```
offset = storage_offset + Σ(index[d] * stride[d]) for d = 0 to n-1
```

Broadcasting achieves operations on differently-shaped tensors by setting certain strides to zero, creating the illusion of a larger tensor without allocating additional memory.

Both mechanisms depend entirely on the stride abstraction.

Shape, stride, and storage offset together form the complete description of how a tensor maps onto memory.


## Next Lesson

**Lesson 03 — Matrix Multiplication**

The *Deep Learning* book now begins using matrix multiplication extensively.

We'll connect the mathematical notation `AB` to `torch.matmul()` and investigate why matrix multiplication is fundamentally different from `transpose()` or `expand()`: it **must** compute new values and therefore almost always allocates new memory.
