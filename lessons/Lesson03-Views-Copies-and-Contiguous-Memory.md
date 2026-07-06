# Lesson 03 — Views, Copies and Contiguous Memory

| | |
|---|---|
| **Chapter** | 2 — Linear Algebra |
| **Reading Assignment** | Review Sections 2.1–2.4 |

This lesson builds directly on Lessons 1 and 2 by exploring how tensor memory layout affects PyTorch operations.

## Objectives

By the end of this lesson you should be able to:

- distinguish a view from a copy;
- explain what a contiguous tensor is;
- predict whether an operation allocates memory;
- understand why `view()` sometimes fails;
- know when `.contiguous()` is required.


## Motivation

Suppose you transpose a tensor.

```python
x = torch.arange(12).reshape(3,4)

y = x.t()
```

Everything still works.

Now try

```python
z = y.view(12)
```

PyTorch raises an exception.

Why?

Didn't we just learn that transpose only changes metadata?


## Investigation 1 — A Successful View

```python
import torch

x = torch.arange(12)

y = x.view(3,4)

print(x.data_ptr() == y.data_ptr())
```

Output

```
True
```

No memory was allocated.

`view()` simply created another tensor describing the same storage.


## Investigation 2 — A Failed View

```python
import torch

x = torch.arange(12).reshape(3,4)

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

This surprises many people.


## Why?

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
(4,1)
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
(1,4)
```

Notice something important.

The values are **not arranged sequentially** anymore.

They are reached by "jumping" through storage.


## What Does "Contiguous" Mean?

A tensor is **contiguous** when its logical order matches the order in memory.

For the tensor

```
0 1 2 3
4 5 6 7
8 9 10 11
```

memory is

```
0 1 2 3 4 5 6 7 8 9 10 11
```

Everything lines up perfectly.

After transpose

```
0 4 8
1 5 9
2 6 10
3 7 11
```

the logical order no longer matches the physical order.

The tensor is therefore **non-contiguous**.


## Why Does `view()` Fail?

`view()` never rearranges memory.

It only changes metadata.

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


## Investigation 3 — `.reshape()`

```python
x = torch.arange(12).reshape(3,4)

y = x.t()

z = y.reshape(12)

print(z.is_contiguous())
```

This works.

Why?

Because `reshape()` is allowed to copy memory when necessary.

A useful mental model is:

```
view()

↓

Never copies

reshape()

↓

Copies only if required
```


## Investigation 4 — `.contiguous()`

```python
x = torch.arange(12).reshape(3,4)

y = x.t()

z = y.contiguous()

print(z.is_contiguous())
```

Now

```python
z.view(12)
```

works.

Why?

Because `.contiguous()` creates a new tensor whose data has been physically rearranged into contiguous memory.


## Source Reading

Open

```
aten/src/ATen/native/TensorShape.cpp
```

Locate the implementations of

```
view

reshape

contiguous
```

You don't need to understand every line.

Instead, answer these questions.

- When does `reshape()` return a view?
- When does it allocate memory?
- Why is `view()` stricter?


## Framework Design

Suppose you are implementing your own tensor library.

Would you make

```
reshape()
```

always copy?

Or would you adopt PyTorch's strategy?

```
Try view first

↓

Copy only if necessary
```

Think about the performance implications.


## Exercises

### 1

Predict the result before running.

```python
x = torch.arange(24).reshape(2,3,4)

y = x.transpose(1,2)

print(y.shape)

print(y.stride())

print(y.is_contiguous())
```


### 2

Find five tensor operations.

Classify them as

- always returns a view
- may return a view
- always allocates memory


### 3

Without looking at the documentation,

explain the difference between

```
view

reshape

transpose

contiguous
```

using only the concepts introduced in Lessons 1–3.


## Summary

By now, the tensor abstraction should be much less mysterious.

A tensor consists of

- storage;
- metadata;
- an interpretation of that storage.

Many PyTorch operations merely change the interpretation.

Others require rearranging memory.

Understanding the difference explains a large class of PyTorch behaviors that otherwise seem arbitrary.


## Next Lesson

**Lesson 04 — Tensor Indexing and Memory Access**

We'll answer another important question:

> How does PyTorch convert

```python
x[i, j, k]
```

into a memory address?

We'll derive the indexing formula from the concepts of

- shape,
- stride,
- and storage offset,

then connect it directly to the implementation inside `TensorImpl`.
