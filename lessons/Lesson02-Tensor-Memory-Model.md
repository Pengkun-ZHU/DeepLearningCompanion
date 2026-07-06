# Lesson 02 — Tensor Memory Model

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra |
| **Reading Assignment** | Review Sections 2.1–2.4 |

This lesson extends the mathematical concepts in the book by exploring how tensors are represented in memory.

## Objectives

By the end of this lesson you should be able to:

- explain what tensor storage is;
- explain the purpose of shape, stride and storage offset;
- predict whether an operation allocates memory;
- distinguish a tensor from its underlying storage;
- explain why multiple tensors can share the same storage.


## Motivation

Recall the experiment from Lesson 1.

```python
b = torch.arange(12).reshape(3, 4)

c = b.t()
```

We observed that

- `b` and `c` have different shapes;
- they have different strides;
- they share the same storage.

How is this possible?

The answer is that a tensor does **not** own its values directly.

Instead, it describes **how to interpret** a block of memory.


## Investigation 1 — Storage

```python
import torch

a = torch.arange(12)

b = a.reshape(3,4)

print(a.untyped_storage().data_ptr())
print(b.untyped_storage().data_ptr())

print(a.storage_offset())
print(b.storage_offset())
```

### Questions

1. Why are the storage pointers identical?
2. Why does `reshape()` not allocate memory?
3. What exactly belongs to the tensor?
4. What belongs to the storage?


## Investigation 2 — Stride

```python
import torch

b = torch.arange(12).reshape(3,4)

print(b)

print("shape :", b.shape)
print("stride:", b.stride())
```

Output

```
shape : torch.Size([3,4])

stride: (4,1)
```

What does `(4,1)` mean?

Imagine the tensor flattened in memory.

```
0 1 2 3 4 5 6 7 8 9 10 11
```

The first stride (`4`) means:

> Move four elements to advance one row.

The second stride (`1`) means:

> Move one element to advance one column.

Stride is therefore **the mapping between logical indices and physical memory**.

It is **not** part of linear algebra.

It is purely an implementation detail.


## Investigation 3 — Transpose

```python
b = torch.arange(12).reshape(3,4)

c = b.t()

print(c.shape)

print(c.stride())
```

Output

```
torch.Size([4,3])

(1,4)
```

Notice something remarkable.

Only the metadata changed.

The storage remained untouched.

Transpose is therefore essentially:

```
old shape
↓

new shape

old stride
↓

new stride
```

No elements are copied.

That is why `transpose()` is essentially O(1).


## Investigation 4 — Storage Offset

```python
b = torch.arange(12).reshape(3,4)

c = b[1:]

print(c)

print(c.storage_offset())
```

Observe that

- storage pointer is unchanged;
- storage offset is no longer zero.

The tensor now starts somewhere in the middle of the shared storage.


## Putting Everything Together

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

Now these names should make much more sense.

Notice that none of them contain numerical values.

They are all metadata.

Next, open

```
c10/core/Storage.h
```

Observe that the raw memory is managed separately.

This separation allows multiple tensors to reference the same storage while presenting different logical views.


## Framework Design

Suppose you were implementing your own tensor library.

Would you

```
Tensor

↓

owns memory
```

or

```
Tensor

↓

points to Storage
```

PyTorch chose the second design.

Why?

Can you think of three advantages?

(Hint: transpose, slicing and views.)


## Exercises

### 1.

Predict the output before running.

```python
a = torch.arange(12)

b = a.reshape(3,4)

c = b[:,1:]

print(c.shape)

print(c.stride())

print(c.storage_offset())
```


### 2.

Create three tensors sharing one storage.

Verify experimentally.


### 3.

Explain the difference between

- storage
- tensor
- shape
- stride
- storage offset

without referring to the PyTorch documentation.


### 4.

Find one tensor operation that allocates memory.

Find another that only creates a view.

Explain the difference.


## Summary

Today's lesson introduced the implementation details hidden beneath the mathematical definition of a tensor.

A tensor is **not** the data itself.

Instead, it is a lightweight object describing

- where the data lives;
- how large each dimension is;
- how indices map onto memory;
- where the logical tensor begins.

Once this idea becomes intuitive, many seemingly mysterious PyTorch behaviors become obvious.


## Next Lesson

**Lesson 03 — Views, Copies and Contiguous Memory**

We'll answer questions such as:

- Why does `view()` sometimes fail?
- What exactly is a contiguous tensor?
- Why does `reshape()` sometimes allocate memory?
- Why does calling `.contiguous()` fix certain errors?