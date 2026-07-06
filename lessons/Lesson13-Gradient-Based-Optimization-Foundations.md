# Lesson 13 — Gradient-Based Optimization Foundations

| | |
|---|---|
| **Chapter** | 4 — Numerical Computation |
| **Reading Assignment** | Sections 4.3–4.5 (Gradient-Based Optimization, Constrained Optimization, Example: Linear Least Squares) |

This lesson connects the mathematical theory of gradient-based optimization to PyTorch's optimizer API.

## Objectives

By the end of this lesson you should be able to:

- explain what a gradient measures and why it points toward steepest ascent;
- implement gradient descent from scratch;
- use `torch.optim.SGD` to minimize a function;
- understand the role of the learning rate;
- recognize saddle points and local minima and explain why they matter for deep learning.


## Motivation

Gradient descent is the algorithm that makes deep learning work.

Every neural network training procedure is, at its core, a gradient descent procedure.

The *Deep Learning* book formalizes the mathematics.

This lesson investigates the implementation: how PyTorch computes gradients, how the optimizer updates parameters, and what can go wrong.


## The Gradient

The gradient of a scalar function `f` at point `x` is the vector of partial derivatives:

```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

It points in the direction of steepest ascent.

Gradient descent takes steps in the opposite direction:

```
x ← x - α ∇f(x)
```

where `α` is the learning rate.


## Investigation 1 — Manual Gradient Descent

```python
import torch

# Minimize f(x) = x^2 + 4x + 4 = (x+2)^2
# Minimum at x = -2

x = torch.tensor(5.0, requires_grad=True)

learning_rate = 0.1

for step in range(30):
    f = x**2 + 4*x + 4

    f.backward()

    with torch.no_grad():
        x -= learning_rate * x.grad

    x.grad.zero_()

    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, f(x) = {f.item():.6f}")

print(f"\nMinimum found at x = {x.item():.6f}")
print(f"True minimum at x = -2.0000")
```

Observe how `x` converges to `-2`.


## PyTorch Optimizer API

In practice, you do not implement the update step manually.

PyTorch provides `torch.optim.SGD` and other optimizers.

```python
x = torch.tensor(5.0, requires_grad=True)

optimizer = torch.optim.SGD([x], lr=0.1)

for step in range(30):
    optimizer.zero_grad()

    f = x**2 + 4*x + 4

    f.backward()

    optimizer.step()

    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, f(x) = {f.item():.6f}")
```

The pattern:

```python
optimizer.zero_grad()   # clear old gradients
loss.backward()         # compute new gradients
optimizer.step()        # update parameters
```

is the standard PyTorch training loop.


## Investigation 2 — Learning Rate Sensitivity

The learning rate controls the step size.

```python
def optimize(lr, steps=50):
    x = torch.tensor(5.0, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=lr)

    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        f = x**2 + 4*x + 4
        f.backward()
        optimizer.step()
        history.append(x.item())

    return history

for lr in [0.01, 0.1, 0.5, 1.1]:
    h = optimize(lr)
    print(f"lr={lr}: final x = {h[-1]:.4f}")
```

Observe:

- small learning rate converges slowly;
- large learning rate diverges;
- there is a sweet spot.

For `f(x) = x²`, the critical learning rate is 1.

Beyond that, gradient descent oscillates or diverges.


## The Loss Surface

For neural networks, the function being minimized is the loss surface.

Unlike the simple quadratic above, the loss surface is:

- high-dimensional;
- non-convex;
- full of saddle points.


## Investigation 3 — Saddle Points

A saddle point is a critical point that is neither a local minimum nor a maximum.

```python
# f(x, y) = x^2 - y^2
# Saddle point at (0, 0)

x = torch.tensor(0.1, requires_grad=True)
y = torch.tensor(0.1, requires_grad=True)

optimizer = torch.optim.SGD([x, y], lr=0.1)

for step in range(20):
    optimizer.zero_grad()
    f = x**2 - y**2
    f.backward()
    optimizer.step()

    print(f"Step {step:2d}: x={x.item():.4f}, y={y.item():.4f}, f={f.item():.6f}")
```

Observe that:

- `x` converges toward 0 (a minimum in the x-direction);
- `y` diverges (a maximum in the y-direction).

Gradient descent escapes saddle points because random initialization places the optimizer away from the saddle.


## Directional Derivative and the Jacobian

For vector-valued functions, gradients generalize to Jacobians.

```python
def f(x):
    return torch.stack([x[0]**2, x[1]**3])

x = torch.tensor([2.0, 3.0], requires_grad=True)

output = f(x)

# Jacobian row 0
output[0].backward(retain_graph=True)
print(f"∂f₀/∂x = {x.grad}")  # [4, 0]

x.grad.zero_()

# Jacobian row 1
output[1].backward()
print(f"∂f₁/∂x = {x.grad}")  # [0, 27]
```

PyTorch also provides `torch.autograd.functional.jacobian()` for full Jacobian computation.


## Companion Insight

Gradient descent is deceptively simple.

The update rule is a single line.

The difficulty is entirely in:

- the shape of the loss surface;
- the choice of learning rate;
- the number of steps;
- what you do when gradients are noisy (stochastic settings).

Modern deep learning adds **momentum**, **adaptive learning rates**, and **learning rate schedules** on top of the basic gradient descent rule.

We'll study these in Lesson 23 when we cover stochastic gradient descent in detail.


## Source Reading

Open:

```
torch/optim/sgd.py
```

Find the `step()` method.

The core update is:

```python
p.add_(grad, alpha=-lr)
```

This is:

```
x ← x - lr * ∇f(x)
```

in one line of Python.

Compare this to the manual update you wrote in Investigation 1.


## Exercises

### 1

Minimize `f(x) = (x - 3)² + (x - 3)⁴` manually using gradient descent.

Does the minimum lie at `x = 3`?

Plot `x` vs. step for learning rates `0.01`, `0.1`, and `0.5`.

### 2

Implement gradient descent for a 2D function:

```python
f(x, y) = (x - 1)^2 + (y + 2)^2
```

What is the minimum?

How many steps does it take to get within 0.001 of the minimum with `lr = 0.1`?

### 3

Using `torch.optim.SGD`:

- Start from `x = 10.0`.
- Minimize `f(x) = x^4 - 4x^2`.
- There are two local minima. Find both by using different initializations.

### 4

Read the documentation for `torch.optim.SGD`.

What is the `momentum` parameter?

Without using momentum, manually implement momentum in the gradient descent loop.


## Summary

Gradient-based optimization is the engine of deep learning.

The update rule is:

```
x ← x - α ∇f(x)
```

PyTorch's optimizer API wraps this into:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The challenges in practice are not the algorithm but the properties of the loss surface and the sensitivity to hyperparameters like the learning rate.


## Next Lesson

**Lesson 14 — Computational Graphs and Autograd**

The previous lessons assumed that gradients were available.

Now we'll investigate how PyTorch computes them automatically through its automatic differentiation engine.
