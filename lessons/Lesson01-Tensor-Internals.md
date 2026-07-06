# Lesson 01 — Tensor Internals: Storage, Strides, Views, and Contiguous Memory

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra |
| **Reading Assignment** | Sections 2.1–2.4 (Scalars, Vectors, Matrices, Tensors) |

This lesson assumes you have already completed the assigned reading from **Deep Learning**.

## Objectives

By the end of this lesson you should be able to:

- explain why PyTorch uses a single `Tensor` abstraction;
- explain what tensor storage is;
- explain the purpose of shape, stride, and storage offset;
- predict whether an operation allocates memory;
- distinguish a view from a copy;
- explain what a contiguous tensor is;
- understand why `view()` sometimes fails;
- know when `.contiguous()` is required.


## Motivation

The *Deep Learning* book introduces four mathematical objects:

- scalar
- vector
- matrix
- tensor

From a mathematical perspective they are different objects.

PyTorch deliberately ignores this distinction.

Everything—from a single number to the weights of a trillion-parameter language model—is represented by one class:

```python
torch.Tensor
```

Before we explain why, run this program.

```python
import torch

b = torch.arange(12).reshape(3, 4)

print("Tensor b")
print(b)
print("shape       :", b.shape)
print("stride      :", b.stride())
print("contiguous  :", b.is_contiguous())

print()

c = b.t()

print("Tensor c = b.t()")
print(c)
print("shape       :", c.shape)
print("stride      :", c.stride())
print("contiguous  :", c.is_contiguous())

print()

print("Same storage:",
      b.untyped_storage().data_ptr()
      ==
      c.untyped_storage().data_ptr())
```

You should observe:

- `b` and `c` have different shapes.
- They have different strides.
- One is contiguous while the other is not.
- They still share exactly the same storage.

The important observation is not that the matrix became transposed.

The important observation is that **the memory never moved**.

PyTorch simply changed **how that memory is interpreted**.

This tells us something fundamental: a tensor is **not** just data. It also contains metadata.


## The Memory Model

### Storage

A tensor does **not** own its values directly. Instead, it describes **how to interpret** a block of memory.

```python
import torch

a = torch.arange(12)

b = a.reshape(3, 4)

print(a.untyped_storage().data_ptr())
print(b.untyped_storage().data_ptr())

print(a.storage_offset())
print(b.storage_offset())
```

The storage pointers are identical because `reshape()` does not allocate memory.

It creates a new view of the same data.

### Stride

```python
import torch

b = torch.arange(12).reshape(3, 4)

print(b)

print("shape :", b.shape)
print("stride:", b.stride())
```

Output

```
shape : torch.Size([3, 4])

stride: (4, 1)
```

The underlying storage is simply

```
0 1 2 3 4 5 6 7 8 9 10 11
```

The first stride (`4`) means:

> Move four elements to advance one row.

The second stride (`1`) means:

> Move one element to advance one column.

Stride is therefore **the mapping between logical indices and physical memory**.

It is purely an implementation detail — not part of linear algebra.

### Transpose Only Changes Metadata

```python
b = torch.arange(12).reshape(3, 4)

c = b.t()

print(c.shape)

print(c.stride())
```

Output

```
torch.Size([4, 3])

(1, 4)
```

Only the metadata changed. The storage remained untouched.

Transpose is therefore essentially:

```
old shape  →  new shape

old stride →  new stride
```

No elements are copied. That is why `transpose()` is essentially O(1).

### Storage Offset

```python
b = torch.arange(12).reshape(3, 4)

c = b[1:]

print(c)

print(c.storage_offset())
```

The storage pointer is unchanged, but the storage offset is no longer zero.

The tensor now starts somewhere in the middle of the shared storage.

### Tensor Structure

A PyTorch tensor is conceptually

```
Tensor

├── Storage
├── Shape
├── Stride
├── Storage Offset
├── DType
└── Device
```

The actual numerical values live inside **Storage**.

Everything else describes how those values should be interpreted.


## Views and Contiguous Memory

Suppose you transpose a tensor.

```python
x = torch.arange(12).reshape(3, 4)

y = x.t()
```

Everything still works. Now try

```python
z = y.view(12)
```

PyTorch raises an exception.

Why? Didn't we just learn that transpose only changes metadata?

### A Successful View

```python
import torch

x = torch.arange(12)

y = x.view(3, 4)

print(x.data_ptr() == y.data_ptr())
```

Output

```
True
```

No memory was allocated.

`view()` simply created another tensor describing the same storage.

### A Failed View

```python
import torch

x = torch.arange(12).reshape(3, 4)

y = x.t()

print(y.is_contiguous())

z = y.view(12)
```

Output

```
RuntimeError:
view size is not compatible with
input tensor's size and stride...
```

### What Does "Contiguous" Mean?

Let's visualize the storage.

```
Storage

0 1 2 3 4 5 6 7 8 9 10 11
```

Tensor `x`

```
0  1  2  3
4  5  6  7
8  9 10 11
```

Stride

```
(4, 1)
```

After transpose

```
0 4 8
1 5 9
2 6 10
3 7 11
```

Stride

```
(1, 4)
```

A tensor is **contiguous** when its logical order matches the order in memory.

For `x`, memory is `0 1 2 3 4 5 6 7 8 9 10 11` — everything lines up perfectly.

After transpose, the logical order no longer matches the physical order.

The tensor is therefore **non-contiguous**.

### Why Does `view()` Fail?

`view()` never rearranges memory. It only changes metadata.

Flattening

```
0 4 8
1 5 9
2 6 10
3 7 11
```

into

```
0 4 8 1 5 ...
```

would require physically moving data.

Since `view()` refuses to copy memory, it throws an exception.

### `reshape()` vs `view()`

```python
x = torch.arange(12).reshape(3, 4)

y = x.t()

z = y.reshape(12)

print(z.is_contiguous())
```

This works because `reshape()` is allowed to copy memory when necessary.

A useful mental model:

```
view()    →  Never copies

reshape() →  Copies only if required
```

### `.contiguous()`

```python
x = torch.arange(12).reshape(3, 4)

y = x.t()

z = y.contiguous()

print(z.is_contiguous())
```

Now `z.view(12)` works.

`.contiguous()` creates a new tensor whose data has been physically rearranged into contiguous memory.


## Source Reading

Open

```
c10/core/TensorImpl.h
```

Locate

```cpp
sizes_

strides_

storage_offset_
```

Notice that none of these fields contain numerical values. They are all metadata.

Next, open

```
c10/core/Storage.h
```

Observe that the raw memory is managed separately from the tensor metadata.

This separation allows multiple tensors to reference the same storage while presenting different logical views.

Finally, open

```
aten/src/ATen/native/TensorShape.cpp
```

Locate the implementations of `view`, `reshape`, and `contiguous`. Answer:

- When does `reshape()` return a view?
- When does it allocate memory?
- Why is `view()` stricter?


## Framework Design

PyTorch chose to separate Storage from Tensor metadata:

```
Tensor  →  points to Storage
```

rather than

```
Tensor  →  owns memory directly
```

Can you identify three operations that become essentially free because of this design?

(Hint: transpose, slicing, views.)


## Exercises

### 1

Create

- a scalar
- a vector
- a matrix
- a 4-D tensor

Print `shape`, `ndim`, and `stride()` for each.


### 2

Predict the output before running.

```python
a = torch.arange(12)

b = a.reshape(3, 4)

c = b[:, 1:]

print(c.shape)

print(c.stride())

print(c.storage_offset())
```


### 3

Create three tensors sharing one storage. Verify experimentally.


### 4

Predict the result before running.

```python
x = torch.arange(24).reshape(2, 3, 4)

y = x.transpose(1, 2)

print(y.shape)

print(y.stride())

print(y.is_contiguous())
```


### 5

Find five tensor operations. Classify each as:

- always returns a view
- may return a view
- always allocates memory


### 6

Without referring to the documentation, explain the difference between `view`, `reshape`, `transpose`, and `contiguous` using only the concepts from this lesson.


### 7

In one paragraph, explain why PyTorch does not provide separate `Vector` and `Matrix` classes.


## Summary

A tensor is **not** the data itself.

Instead, it is a lightweight object describing

- where the data lives;
- how large each dimension is;
- how indices map onto memory;
- where the logical tensor begins.

Many PyTorch operations merely change this interpretation.

Others require rearranging memory.

Understanding the difference explains a large class of PyTorch behaviors that otherwise seem arbitrary.

The core principle: **metadata changes are free; memory copies are not**.


## Next Lesson

**Lesson 02 — Tensor Indexing and Broadcasting**

We'll answer two more fundamental questions:

- How does PyTorch convert `x[i, j, k]` into a memory address?
- How can shapes `(3, 1)` and `(1, 4)` be added together to produce `(3, 4)` without copying data?

Both answers rely on the stride abstraction built in this lesson.
