# Lesson 07 — Gradient-Based Optimization Foundations

|Book|Chapter|Section|
|---|---|---|
| Deep Learning | 4 — Numerical Computation  | Sections 4.3–4.5 |

This lesson develops the optimization machinery behind neural network training. We begin with the mathematical role of the gradient, move to gradient descent as the core optimization algorithm, and then connect that algorithm to the practical constraints of training deep models: learning rate sensitivity, non-convex loss surfaces, and stochastic updates.

## Objectives

By the end of this lesson you should be able to:

- explain what the gradient measures and why it points in the steepest ascent direction;
- implement gradient descent from scratch in PyTorch;
- reason about the effect of the learning rate on convergence and stability;
- distinguish a minimum, a saddle point, and a flat region on a loss surface;
- explain why stochastic gradient descent is used in practice;
- connect the optimization loop to the training code used in real PyTorch models.


## Motivation

In the previous lesson we saw how a loss function is built from probability models, log-likelihoods, entropy, and stability-aware numerical operations.

That gives us a scalar objective. The next problem is operational:

> how do we actually find the parameter values that make that objective small?

The answer is gradient-based optimization.

In deep learning, the optimization problem is rarely solved analytically. Instead, models are trained iteratively by repeatedly computing a gradient and taking a step in the opposite direction. This is the central mechanism behind all supervised learning.


## The Gradient

For a scalar function $f(x)$, the derivative tells us the slope. For a vector-valued input, the gradient generalizes this idea:

$$
\nabla f(x) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

The gradient points in the direction of steepest ascent. To minimize $f$, we move in the opposite direction.

The basic update rule is:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta f(\theta_t)
$$

where $\alpha$ is the learning rate.

This update is the foundation of training in neural networks.


## Investigation 1 — Manual Gradient Descent

We begin with a one-dimensional quadratic:

$$
f(x) = x^2 + 4x + 4 = (x+2)^2
$$

Its minimum is at $x = -2$.

```python
import torch

x = torch.tensor(5.0, requires_grad=True)
learning_rate = 0.1

for step in range(30):
    loss = x**2 + 4*x + 4
    loss.backward()

    with torch.no_grad():
        x -= learning_rate * x.grad

    x.grad.zero_()

    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, loss = {loss.item():.6f}")

print(f"\nMinimum found at x = {x.item():.6f}")
print("True minimum at x = -2.0000")
```

The behavior is exactly what we expect:

- the gradient is large at the beginning, so the parameter moves quickly;
- as $x$ approaches the minimizer, the gradient shrinks;
- the iterates converge toward the optimum.

This is the simplest example of a training loop, but it already captures the essential algorithmic idea.


## PyTorch Optimizer API

In practice, we rarely write the update ourselves. PyTorch provides `torch.optim.SGD`, which wraps the gradient computation and parameter update in a clean API.

```python
import torch

x = torch.tensor(5.0, requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

for step in range(30):
    optimizer.zero_grad()

    loss = x**2 + 4*x + 4
    loss.backward()

    optimizer.step()

    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, loss = {loss.item():.6f}")
```

The standard pattern is:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

This is the same update rule expressed through a library abstraction.


## Investigation 2 — Learning Rate Sensitivity

The learning rate $\alpha$ determines how far we step in the direction opposite the gradient.

```python
import torch


def optimize(lr, steps=50):
    x = torch.tensor(5.0, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=lr)
    history = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = x**2 + 4*x + 4
        loss.backward()
        optimizer.step()
        history.append(x.item())

    return history

for lr in [0.01, 0.1, 0.5, 1.1]:
    hist = optimize(lr)
    print(f"lr={lr}: final x = {hist[-1]:.4f}")
```

This reveals a practical fact:

- too small a learning rate causes slow convergence;
- too large a learning rate can overshoot or diverge;
- there is a narrow range where the algorithm behaves well.

The learning rate is not a detail. It is one of the most important hyperparameters in deep learning.

For a simple convex quadratic such as $f(x) = x^2$, a learning rate above 1 often causes instability. In higher-dimensional networks, the same principle persists: the optimizer may still move, but it can become erratic or fail to descend effectively.


## The Loss Surface

For a neural network, the objective is not a simple parabola. It is a high-dimensional function of the model parameters.

The training loss surface typically has the following properties:

- high dimensionality;
- non-convex geometry;
- local minima and saddle points;
- flat regions where gradients are small;
- many directions of steep increase and shallow decrease.

This explains why optimization in deep learning is not a matter of choosing a mathematically perfect method and applying it once. It is a repeated process of navigating a complicated surface using noisy local information.

A key consequence is that we often do not find the global optimum. We settle in a good region that yields strong generalization.


## Investigation 3 — Saddle Points

A saddle point is a point where the gradient is zero but the point is neither a local minimum nor a local maximum.

For example:

$$
f(x, y) = x^2 - y^2
$$

The point $(0,0)$ is a saddle point.

```python
import torch

x = torch.tensor(0.1, requires_grad=True)
y = torch.tensor(0.1, requires_grad=True)
optimizer = torch.optim.SGD([x, y], lr=0.1)

for step in range(20):
    optimizer.zero_grad()
    loss = x**2 - y**2
    loss.backward()
    optimizer.step()

    print(f"Step {step:2d}: x={x.item():.4f}, y={y.item():.4f}, loss={loss.item():.6f}")
```

Interpretation:

- in the $x$ direction, the loss is minimized at 0;
- in the $y$ direction, the loss is maximized at 0;
- the point is not a true minimum.

This illustrates an important theme: loss landscapes in deep learning are not simple, and the optimizer must behave sensibly even when the geometry is pathological.

Random initialization and stochastic updates help prevent the optimizer from getting trapped in undesirable regions for too long.


## Why Stochastic Gradient Descent?

If the full dataset has $N$ examples, the empirical risk is:

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell_i(\theta)
$$

Computing the exact gradient over all examples can be expensive. Instead, we often compute gradients on small minibatches:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta \left(\frac{1}{B}\sum_{i\in\mathcal{B}_t} \ell_i(\theta_t)\right)
$$

This is stochastic gradient descent (SGD).

In practice, SGD is useful because:

- it is computationally cheaper than full-batch gradient descent;
- it introduces noise that can help escape poor local regions;
- it enables training on large datasets that do not fit into memory all at once;
- it often gives better generalization than exact full-batch descent on the same problem.

When combined with momentum, adaptive methods, and learning-rate schedules, SGD becomes the basis of modern optimization.


## Gradient Descent in the Training Loop

The whole training procedure is an optimization loop over parameters.

```python
import torch

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for batch_x, batch_y in data_loader:
    optimizer.zero_grad()

    preds = model(batch_x)
    loss = torch.nn.functional.mse_loss(preds, batch_y)

    loss.backward()
    optimizer.step()
```

This is the practical real-world pattern behind deep learning systems.

The optimizer repeatedly does the following:

1. zero the accumulated gradients;
2. compute predictions and loss;
3. propagate gradients backward through the graph;
4. update parameters in the direction opposite the gradient.

This simple cycle is the basis of nearly all neural network training.


## Source Reading

Open:

```text
torch/optim/sgd.py
```

Look at the `step()` method. The core update is effectively:

```python
p.add_(grad, alpha=-lr)
```

In mathematical language, this is:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}
$$

This is the exact implementation pattern behind basic stochastic gradient descent.

Compare this with the manual gradient descent loop we wrote above. The conceptual algorithm is the same; the library merely handles the bookkeeping.


## Exercises

### 1

Minimize $f(x) = (x-3)^2 + (x-3)^4$ using gradient descent.

- Does the minimum lie at $x=3$?
- Plot the trajectory for learning rates $0.01$, $0.1$, and $0.5$.

### 2

Implement gradient descent for the 2D function:

$$
f(x, y) = (x-1)^2 + (y+2)^2
$$

What is the minimum? How many steps does it take to get within $0.001$ of the minimum for $\text{lr}=0.1$?

### 3

Using `torch.optim.SGD`, minimize:

$$
f(x) = x^4 - 4x^2
$$

starting from $x=10.0$. Examine how different initializations lead to different local minima.

### 4

Read the `torch.optim.SGD` documentation and explain the role of the `momentum` argument.

Then manually implement a momentum update in a gradient descent loop and compare the behaviour to ordinary SGD.

### 5

Create a small 1D example where the learning rate is too large, and show that the iterates oscillate or diverge.

Explain why the same learning rate may be safe in one problem and unsafe in another.

### 6

Explain why stochastic gradient descent is often preferred over full-batch gradient descent in deep learning.

Your answer should mention at least three reasons.


## Summary

Gradient-based optimization is the engine of deep learning training.

The central rule is:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}
$$

The learning process is simple at the level of the update rule, but difficult in practice because:

- the loss surface is high-dimensional and non-convex;
- the learning rate changes the stability and speed of convergence;
- stochastic minibatches inject noise that can help optimization but also create variance;
- the goal is not merely to minimize training loss, but to reach a parameter region that generalizes well.

This is the bridge between the objective defined by the model and the actual training procedure used in PyTorch.


## Next Lesson

**Lesson 08 — Computational Graphs and Autograd**

The previous lessons assumed that gradients were available. Now we will investigate how PyTorch automatically computes these gradients through its automatic differentiation engine.
