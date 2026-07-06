# Lesson 04 — Tensor Indexing and Address Calculation

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra |
| **Reading Assignment** | Review Sections 2.1–2.4 |

This lesson explains how a logical tensor index is converted into a physical memory address.

## Objectives

By the end of this lesson you should be able to:

- explain how tensor indexing works internally;
- derive the address calculation formula used by PyTorch;
- understand the purpose of `storage_offset`;
- predict which memory locations an indexing operation accesses;
- connect tensor indexing to the metadata stored in `TensorImpl`.


## Motivation

In Python, indexing feels trivial.

```python
x[2, 1]
```

returns a single element.

But inside PyTorch, that expression eventually becomes something like

```cpp
*(base_pointer + ?)
```

What is the `?`

How does PyTorch know where element `(2,1)` is located?

Today's lesson answers that question.


## Investigation 1 — Looking at Strides

```python
import torch

x = torch.arange(12).reshape(3,4)

print(x)

print("shape :", x.shape)
print("stride:", x.stride())
```

Output

```
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

shape  = (3,4)

stride = (4,1)
```

Suppose we ask for

```python
x[2,1]
```

The answer is `9`.

How does PyTorch locate that value?


## From Indices to Storage

Ignore the matrix for a moment.

The underlying storage is simply

```
Index

 0  1  2  3  4  5  6  7  8  9 10 11

Value

 0  1  2  3  4  5  6  7  8  9 10 11
```

The tensor is merely a *view* of this storage.

```
0  1  2  3
4  5  6  7
8  9 10 11
```

The question becomes:

> Which storage index corresponds to `(2,1)`?


## Deriving the Formula

The stride tells us

```
stride = (4,1)
```

Moving

- one row advances **4** storage elements.
- one column advances **1** storage element.

Therefore

```
row = 2

↓

2 × 4 = 8
```

```
column = 1

↓

1 × 1 = 1
```

Total offset

```
8 + 1 = 9
```

Therefore

```
x[2,1]

↓

storage[9]
```

which indeed contains

```
9
```


## The General Formula

For a two-dimensional tensor,

```
address_offset

=

row × stride[0]

+

column × stride[1]
```

For an N-dimensional tensor,

```
offset =

Σ

index[d] × stride[d]
```

If the tensor begins in the middle of a storage,

```
offset += storage_offset
```

Finally,

```
address

=

base_pointer

+

offset
```

This is the essential indexing algorithm used by tensor libraries.


## Investigation 2 — Storage Offset

```python
x = torch.arange(12).reshape(3,4)

y = x[1:]

print(y)

print("storage_offset:", y.storage_offset())
print("stride:", y.stride())
```

Output

```
storage_offset = 4
```

Why?

Because `y` starts at the second row.

Instead of beginning at storage element `0`, it begins at storage element `4`.

No memory was copied.

Only the metadata changed.


## Investigation 3 — Verify the Formula

```python
x = torch.arange(24).reshape(2,3,4)

print(x.shape)
print(x.stride())
```

Before indexing,

predict the storage offset for

```python
x[1,2,3]
```

Use only

- shape
- stride
- storage offset

Then verify your answer in Python.


## Companion Insight

One surprising observation is that **shape never appears in the address calculation**.

Shape is used only for:

- validating indices;
- determining tensor dimensions;
- iteration.

Once the indices are known, only

- stride;
- storage offset;

are needed to compute the address.

This explains why transposing a tensor is cheap.

Transpose changes the stride.

The indexing algorithm remains exactly the same.


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

Notice how each field participates in indexing.

Then browse

```
aten/src/ATen
```

Search for

```
stride
```

Observe how frequently stride information appears throughout the implementation.

Tensor indexing is a fundamental operation, so these metadata fields are consulted almost everywhere.


## Framework Design

Suppose you were designing a tensor library.

Would you

```
recompute addresses

using shape
```

or

```
store stride explicitly
```

PyTorch stores stride.

Why?

Think about

- transpose;
- slicing;
- broadcasting;
- views.

Would recomputing stride every time still work?


## Exercises

### 1

For each tensor below, compute the storage offset of the indexed element **by hand** before running the code.

```python
x = torch.arange(12).reshape(3,4)

x[2,3]

x[1,0]

x[0,2]
```


### 2

Repeat the exercise after

```python
x = x.t()
```

Notice that only the stride changes.

The indexing algorithm remains identical.


### 3

Write a Python function

```python
offset(indices, stride, storage_offset)
```

that computes the storage offset exactly as PyTorch does.

Test it on several tensors.


### 4

Explain why shape is not sufficient to compute an address.

What additional information is required?


## Summary

A tensor is fundamentally a mapping

```
logical indices

↓

storage offset

↓

memory address
```

The mapping is controlled by three pieces of metadata:

- stride;
- storage offset;
- base storage pointer.

This simple idea explains tensor indexing, slicing, transpose, and many other PyTorch operations.


## Next Lesson

**Lesson 05 — Broadcasting**

The *Deep Learning* book frequently relies on operations between tensors of different shapes.

How can

```python
(3,1)

+

(1,4)
```

produce

```python
(3,4)
```

without copying data?

We'll discover that broadcasting is another example of PyTorch changing metadata rather than allocating new tensors whenever possible.