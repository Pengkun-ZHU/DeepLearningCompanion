# Lesson 14 — Computational Graphs and Autograd

| | |
|---|---|
| **Chapter** | 4 — Numerical Computation |
| **Reading Assignment** | Section 4.3 (revisit) and Chapter 6.5 (Back-Propagation — preview) |

This lesson investigates how PyTorch's automatic differentiation engine computes gradients.

## Objectives

By the end of this lesson you should be able to:

- explain what a computational graph is;
- trace how PyTorch builds a graph during a forward pass;
- understand reverse-mode automatic differentiation;
- explain the role of `requires_grad`, `grad_fn`, and `backward()`;
- understand when gradients do and do not flow.


## Motivation

In Lesson 17, we called `loss.backward()` without asking how it worked.

PyTorch does not use symbolic differentiation (like SymPy).

It does not use finite differences.

It uses **reverse-mode automatic differentiation**, also called **backpropagation**.

Understanding this mechanism is essential for:

- debugging gradient flow;
- implementing custom operations;
- reasoning about memory usage during training.


## Computational Graphs

A computational graph is a directed acyclic graph (DAG).

Each node represents an operation.

Edges represent data flow.

Consider:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

z = x * y + x**2
```

The graph is:

```
x ──┬─── * ──── + ──── z
y ──┘         │
x ─── **2 ───┘
```


## Investigation 1 — Inspecting the Graph

```python
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

a = x * y
b = x ** 2
z = a + b

print(f"z      : {z}")
print(f"grad_fn: {z.grad_fn}")          # AddBackward0
print(f"inputs : {z.grad_fn.next_functions}")
```

PyTorch records the operations as it executes them.

Each tensor produced by an operation carries a `grad_fn` pointer to the operation that created it.

Leaf tensors have `grad_fn = None`.

```python
print(f"x grad_fn: {x.grad_fn}")   # None — x is a leaf
print(f"a grad_fn: {a.grad_fn}")   # MulBackward0
```


## Forward Pass and Reverse Pass

The **forward pass** computes the output.

The **backward pass** traverses the graph in reverse, applying the chain rule at each node.

```python
z.backward()

print(f"∂z/∂x = {x.grad}")   # 4 (from x*y) + 6 (from x**2) = 4 + 6 = 10
print(f"∂z/∂y = {y.grad}")   # x = 3
```

Verify manually:

```
z = x*y + x²
∂z/∂x = y + 2x = 4 + 6 = 10 ✓
∂z/∂y = x = 3             ✓
```


## Investigation 2 — Gradient Accumulation

PyTorch accumulates gradients rather than overwriting them.

```python
x = torch.tensor(2.0, requires_grad=True)

for _ in range(3):
    y = x ** 2
    y.backward()
    print(f"x.grad = {x.grad}")   # 4, 8, 12 — accumulates!
```

Always call `optimizer.zero_grad()` (or `x.grad.zero_()`) before the backward pass.

Failing to zero gradients is one of the most common bugs in PyTorch code.


## Investigation 3 — Stopping Gradient Flow

Sometimes you want to compute a value but not backpropagate through it.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

y_detached = y.detach()   # new tensor, no grad_fn

z = y_detached + 1.0

print(f"z.grad_fn: {z.grad_fn}")   # None
```

`torch.no_grad()` is used to disable gradient computation entirely:

```python
x = torch.tensor(3.0, requires_grad=True)

with torch.no_grad():
    y = x ** 2

print(f"y.requires_grad: {y.requires_grad}")   # False
```

This is commonly used during:

- inference (evaluation);
- metric computation;
- parts of the model that should not be trained.


## The Chain Rule in Code

Autograd applies the chain rule at every node.

For a node computing `f(a, b)` during the backward pass, autograd:

1. receives the gradient of the loss with respect to the node's output: `∂L/∂f`;
2. computes the gradient with respect to each input using the local derivative;
3. passes it upstream.

```python
# Custom scalar function: f(x) = x^3
x = torch.tensor(2.0, requires_grad=True)

y = x ** 3

y.backward()

print(f"dy/dx at x=2: {x.grad}")   # 3 * x^2 = 12
```

For each operation, PyTorch has a pre-defined backward function.

You can also define your own with `torch.autograd.Function`.


## Investigation 4 — retain_graph

By default, PyTorch frees the computational graph after calling `backward()`.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

y.backward()

try:
    y.backward()   # Error: graph was freed
except RuntimeError as e:
    print(e)
```

Use `retain_graph=True` to keep the graph:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

y.backward(retain_graph=True)

x.grad.zero_()

y.backward()

print(f"dy/dx = {x.grad}")
```

Retaining the graph uses more memory.


## Companion Insight

Reverse-mode automatic differentiation is efficient because the cost of computing all partial derivatives of a scalar output is proportional to the cost of the forward pass — regardless of the number of inputs.

This is why training is feasible even when models have hundreds of millions of parameters.

Forward-mode differentiation (the alternative) would cost one forward pass per input dimension.

For a model with 100M parameters and a scalar loss, reverse mode requires one pass while forward mode would require 100M passes.

This asymmetry is why modern deep learning frameworks universally use reverse-mode differentiation.


## Source Reading

Open:

```
torch/autograd/__init__.py
```

Find the `backward()` function.

Observe that it calls `torch._C._EngineBase.run_backward()`.

The actual gradient computation engine is implemented in C++:

```
torch/csrc/autograd/engine.cpp
```

You do not need to read the C++ code in detail.

Simply appreciate that what appears to be Python-level magic is a carefully engineered C++ traversal of the computational graph.


## Exercises

### 1

Manually verify that autograd computes the correct gradient for:

```python
f(x) = sin(x) * exp(-x)
```

at `x = 1.0`.

Compare the autograd result with the analytical derivative.

### 2

Implement a function that takes a neural network and checks whether gradients are flowing to all parameters.

Hint: after a backward pass, check `param.grad` for each parameter.

### 3

What happens to gradients if you use `.detach()` on a tensor in the middle of a computation?

Design an experiment to demonstrate this.

### 4

Implement a simple layer using `torch.autograd.Function` with custom `forward` and `backward` methods for:

```
f(x) = max(0, x)   (ReLU)
```

Verify that autograd computes the correct gradient.


## Summary

PyTorch builds a computational graph during the forward pass and traverses it in reverse during `backward()`.

Key concepts:

- `requires_grad=True` marks a tensor as requiring gradients.
- `grad_fn` records the operation that produced a tensor.
- `backward()` traverses the graph and accumulates gradients.
- `detach()` and `torch.no_grad()` stop gradient flow.
- Gradients accumulate — always zero them before each backward pass.

Understanding autograd is prerequisite to implementing custom operations, debugging training, and reasoning about memory usage.


## Next Lesson

**Lesson 15 — Learning Algorithms and the ML Pipeline**

Chapter 5 of the *Deep Learning* book introduces machine learning from a rigorous perspective.

We'll connect the mathematical definitions of learning algorithms, training and test sets, and generalization to their PyTorch implementations.
