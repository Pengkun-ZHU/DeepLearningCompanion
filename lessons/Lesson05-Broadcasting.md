# Lesson 05 — Broadcasting

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra |
| **Reading Assignment** | Continue Chapter 2 (Operations on Tensors) |

This lesson explains how PyTorch performs operations on tensors of different shapes without physically duplicating data.

## Objectives

By the end of this lesson you should be able to:

- explain PyTorch's broadcasting rules;
- predict the result shape of a broadcasted operation;
- understand why broadcasting usually does not allocate memory;
- distinguish logical expansion from physical copying;
- relate broadcasting to tensor strides.


## Motivation

Consider two tensors.

```python
x = torch.arange(12).reshape(3,4)

b = torch.tensor([100,200,300,400])
```

PyTorch happily computes

```python
x + b
```

producing

```
[[100,201,302,403],
 [104,205,306,407],
 [108,209,310,411]]
```

At first glance this looks impossible.

`x` has shape

```
(3,4)
```

while `b` has shape

```
(4,)
```

How can they be added together?


## Broadcasting Rules

PyTorch compares dimensions from **right to left**.

Two dimensions are compatible if they are:

- equal; or
- one of them is `1`.

For example,

```
(3,4)

(4,)
```

becomes

```
(3,4)

(1,4)
```

Dimension by dimension,

```
3 vs 1

✓

4 vs 4

✓
```

Therefore the result has shape

```
(3,4)
```


## Investigation 1

```python
import torch

x = torch.arange(12).reshape(3,4)

b = torch.tensor([100,200,300,400])

y = x + b

print(y.shape)
```

Now inspect

```python
print(b.shape)
```

Notice that

```
b

still has shape

(4,)
```

PyTorch did **not** permanently convert it into `(3,4)`.


## Did PyTorch Copy the Data?

Imagine broadcasting by physically copying.

```
100 200 300 400

↓

100 200 300 400
100 200 300 400
100 200 300 400
```

This would require allocating new memory.

Instead, PyTorch does something much cleverer.

It creates a temporary **view** that behaves *as if* the data had been repeated.

The underlying values are still stored only once.


## Investigation 2 — `expand()`

```python
b = torch.tensor([100,200,300,400])

e = b.expand(3,4)

print(e)

print(e.stride())

print(e.is_contiguous())
```

Observe:

- no new values were created;
- the expanded tensor is not contiguous;
- the stride looks unusual.


## The Secret: Zero Strides

A normal tensor might have

```
stride

(4,1)
```

An expanded tensor can have

```
(0,1)
```

What does a stride of zero mean?

It means:

> Moving along that dimension does **not** move in memory.

Every row points to the same four values.

Conceptually,

```
100 200 300 400

↓

100 200 300 400
100 200 300 400
100 200 300 400
```

Physically,

```
100 200 300 400
```

exists only once.

The repeated rows are an illusion created entirely by metadata.


## Investigation 3

```python
b = torch.tensor([1,2,3])

e = b.expand(5,3)

print(e.stride())

print(e.storage().data_ptr())

print(b.storage().data_ptr())
```

Verify that

- both tensors share storage;
- only the metadata differs.


## Companion Insight

Broadcasting is another example of a recurring design principle we've seen throughout these lessons.

Instead of copying data,

PyTorch first asks:

> Can this be expressed by changing metadata alone?

If the answer is yes,

the operation becomes almost free.

We've already seen this with

- `transpose()`;
- slicing;
- views.

Broadcasting follows exactly the same philosophy.


## Source Reading

Open

```
aten/src/ATen/ExpandUtils.cpp
```

Browse the implementation of

```
expand()
```

Notice how the function computes

- new sizes;
- new strides;

rather than allocating a larger tensor.

You don't need to understand every line.

Simply identify where the expanded stride is constructed.


## Framework Design

Suppose you were implementing broadcasting.

Would you

```
allocate

↓

repeat values
```

or

```
reuse storage

↓

change strides
```

PyTorch chooses the second approach whenever possible.

Think about why this matters for

- memory usage;
- cache efficiency;
- GPU workloads.


## Exercises

### 1

Predict whether each pair of shapes can be broadcast.

```
(3,4)

(4,)
```

```
(5,1)

(5,7)
```

```
(2,3)

(3,2)
```

```
(8,1,6)

(7,6)
```

Explain your reasoning.


### 2

Create a tensor with

```python
expand()
```

Inspect

- `shape`
- `stride`
- `storage_offset`
- `is_contiguous()`

How do they differ from the original tensor?


### 3

Can an expanded tensor be modified in place?

Try

```python
e.add_(1)
```

What happens?

Why?


### 4

Without referring to the documentation,

explain broadcasting using only the concepts introduced in Lessons 1–5.

Do **not** use the word "copy."


## Summary

Broadcasting does not usually duplicate data.

Instead,

PyTorch creates a new interpretation of the existing storage.

The key observation is that broadcasting can be implemented by changing tensor metadata—particularly the stride—rather than allocating larger arrays.

This continues the central theme of the previous lessons:

> Many tensor operations are inexpensive because they manipulate metadata instead of memory.


## Next Lesson

**Lesson 06 — Tensor Operations and Linear Algebra**

The *Deep Learning* book now begins using matrix multiplication extensively.

We'll connect the mathematical notation

```
AB
```

to

```python
torch.matmul()

@

torch.mm()
```

and investigate why matrix multiplication is fundamentally different from operations like `transpose()` or `expand()`: it **must** compute new values and therefore almost always allocates new memory.