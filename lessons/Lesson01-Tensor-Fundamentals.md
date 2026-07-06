# Lesson 01 — Tensor Fundamentals

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra |
| **Reading Assignment** | Sections 2.1–2.4 (Scalars, Vectors, Matrices, Tensors) |

This lesson assumes you have already completed the assigned reading from **Deep Learning**.

## Objectives

By the end of this lesson you should be able to:

- explain why PyTorch uses a single `Tensor` abstraction;
- distinguish **shape**, **stride**, **storage**, and **storage offset**;
- explain why `transpose()` is essentially O(1);
- locate the core tensor metadata inside the PyTorch source tree.


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

That design decision is one of the reasons PyTorch's API feels remarkably consistent.

Today's lesson begins investigating **why**.


## Investigation

Run the following program **before reading any further**.

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

At first glance, these observations seem contradictory.


## Think Before Reading On

Try answering these questions without searching online.

1. Why didn't `transpose()` allocate another tensor?
2. Why is `transpose()` almost instantaneous?
3. What actually changed when `b` became `c`?
4. How can two tensors share memory while presenting different layouts?
5. If tensors only stored numbers, would any of this be possible?

Write down your own answers.

We'll revisit them in the next lesson.


## Companion Insight

The important observation from today's experiment is **not** that the matrix became transposed.

The important observation is that **the memory never moved**.

PyTorch simply changed **how that memory is interpreted**.

This tells us something fundamental:

A tensor is **not** just data.

A tensor also contains metadata.

At a minimum, PyTorch must somehow record:

- the tensor's dimensions;
- the size of each dimension;
- how indices map onto memory;
- where the underlying storage begins.

The *Deep Learning* book intentionally leaves these implementation details aside because they are not part of linear algebra.

As framework developers, however, these details explain why some tensor operations are effectively free while others require allocating and copying memory.


## Source Reading

Open the following file in the PyTorch repository.

```
c10/core/TensorImpl.h
```

Locate these member variables.

```
sizes_

strides_

storage_offset_
```

Don't try to understand the entire class.

Simply answer one question:

> Why does a tensor need all three?

Next, open:

```
c10/core/Storage.h
```

Question:

> Why are `Tensor` and `Storage` separate classes?

We'll answer both questions in Lesson 02.


## Exercises

### 1.

Create

- a scalar
- a vector
- a matrix
- a 4-D tensor

Print

```python
shape
ndim
stride()
```

for each.


### 2.

Create a transpose.

Verify experimentally that both tensors share the same storage.


### 3.

Slice a tensor.

Does slicing allocate memory?

How can you verify your answer?


### 4.

In one paragraph, explain why PyTorch does not provide separate `Vector` and `Matrix` classes.


## Summary

The *Deep Learning* book introduces tensors as mathematical objects.

PyTorch turns that mathematical idea into a single universal data structure.

Today's experiment shows that a tensor is much more than a block of numbers.

It is a combination of:

- storage;
- metadata;
- rules describing how that storage should be interpreted.

Understanding those rules is the key to understanding PyTorch itself.


## Next Lesson

**Lesson 02 — Tensor Memory Model**

We'll answer every question raised today.

Topics include:

- Storage
- Shape
- Stride
- Storage Offset
- Views vs. Copies
- Contiguous Memory
- Why `view()` sometimes fails