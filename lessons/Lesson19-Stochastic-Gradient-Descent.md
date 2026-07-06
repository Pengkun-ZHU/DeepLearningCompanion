# Lesson 19 — Stochastic Gradient Descent in PyTorch

| | |
|---|---|
| **Chapter** | 5 — Machine Learning Basics |
| **Reading Assignment** | Section 5.9 (Stochastic Gradient Descent) |

This lesson studies SGD in depth and investigates the extensions — momentum and adaptive methods — that make it practical for training deep networks.

## Objectives

By the end of this lesson you should be able to:

- explain why stochastic gradient descent is preferred over full batch gradient descent;
- implement mini-batch SGD from scratch;
- understand momentum and its effect on convergence;
- explain how Adam differs from SGD;
- implement a learning rate schedule.


## Motivation

In the training loop from Lesson 19, we processed one mini-batch per step:

```python
for x_batch, y_batch in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(x_batch), y_batch)
    loss.backward()
    optimizer.step()
```

Why mini-batches instead of all the data at once?

The *Deep Learning* book explains that computing the gradient over the full dataset is expensive and often unnecessary.

Mini-batch gradients are noisy estimates of the true gradient.

That noise turns out to be helpful.


## Full Batch vs. Mini-Batch vs. Stochastic

| Method | Batch size | Gradient estimate | Cost per step |
|---|---|---|---|
| Full batch GD | All data | Exact | Expensive |
| Mini-batch GD | 32–512 | Noisy | Moderate |
| SGD | 1 | Very noisy | Cheap |

In practice, "SGD" in deep learning almost always means mini-batch gradient descent.


## Investigation 1 — Effect of Batch Size

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

# Dataset
n = 1000
x = torch.randn(n, 10)
w_true = torch.randn(10)
y = x @ w_true + 0.1 * torch.randn(n)

dataset = TensorDataset(x, y)


def train(batch_size, epochs=50, lr=0.01):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    return losses


losses_small = train(batch_size=8)
losses_large = train(batch_size=256)

print(f"Final loss (batch=8)  : {losses_small[-1]:.6f}")
print(f"Final loss (batch=256): {losses_large[-1]:.6f}")
```

Observe that small batch sizes may converge faster per epoch (in terms of data efficiency) but with noisier loss curves.


## Momentum

Vanilla SGD can oscillate and converge slowly.

Momentum accumulates a velocity vector:

```
v ← β·v - α·∇L(θ)
θ ← θ + v
```

This smooths the gradient updates and accelerates convergence in consistent directions.

```python
# SGD without momentum
optimizer_no_mom = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

# SGD with momentum
optimizer_mom    = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

The standard value for momentum is 0.9.


## Investigation 2 — Comparing Optimizers

```python
def compare_optimizers(optimizer_fn, epochs=100):
    torch.manual_seed(42)
    model = nn.Linear(10, 1)
    optimizer = optimizer_fn(model.parameters())
    criterion = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    return losses


sgd     = compare_optimizers(lambda p: torch.optim.SGD(p, lr=0.01))
sgd_mom = compare_optimizers(lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9))
adam    = compare_optimizers(lambda p: torch.optim.Adam(p, lr=0.01))

for name, losses in [('SGD', sgd), ('SGD+momentum', sgd_mom), ('Adam', adam)]:
    print(f"{name:>15}: final loss = {losses[-1]:.6f}")
```


## Adam

Adam (Adaptive Moment Estimation) maintains per-parameter adaptive learning rates.

It tracks:

- the first moment (mean of gradients, like momentum);
- the second moment (uncentered variance of gradients).

The update rule:

```
m ← β₁·m + (1-β₁)·g          # first moment
v ← β₂·v + (1-β₂)·g²          # second moment
m̂ = m / (1-β₁ᵗ)               # bias correction
v̂ = v / (1-β₂ᵗ)               # bias correction
θ ← θ - α · m̂ / (√v̂ + ε)
```

```python
optimizer_adam = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

The default values `betas=(0.9, 0.999)` and `eps=1e-8` work well for most problems.

Adam is the de facto default optimizer in deep learning.


## Learning Rate Schedules

The learning rate should often decrease over training.

Early training: large steps to cover broad territory.

Late training: small steps to converge precisely.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Reduce LR by factor 0.5 every 30 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

for epoch in range(90):
    # ... training loop ...

    scheduler.step()

    if epoch % 30 == 0:
        print(f"Epoch {epoch}: lr = {scheduler.get_last_lr()}")
```

Other useful schedulers:

```python
# Cosine annealing
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Reduce on plateau
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```


## Investigation 3 — Gradient Clipping

Large gradients can destabilize training.

Gradient clipping limits the gradient norm:

```python
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # Clip gradients before the optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

This is especially important for recurrent networks where gradients can explode through time.


## The Complete Training Loop

Bringing it all together:

```python
model     = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
loader    = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(60):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
```

This is the canonical deep learning training loop.


## Companion Insight

The noise in mini-batch gradients is not a flaw.

It acts as a regularizer by:

- preventing the optimizer from settling into sharp minima;
- providing an implicit signal that is different every step;
- helping escape saddle points.

The interaction between batch size, learning rate, and generalization is an active research area.

A widely-used empirical rule is:

> When multiplying batch size by `k`, multiply learning rate by `√k`.

This is a heuristic, not a theorem — but it works well in practice.


## Source Reading

Open:

```
torch/optim/adam.py
```

Find the `step()` method.

Identify where:

- the first moment estimate `m` is updated;
- the second moment estimate `v` is updated;
- bias correction is applied;
- the parameter is updated.

Compare the code to the mathematical update rule given above.


## Exercises

### 1

Train the same model with `batch_size` of 1, 32, 256, and 1000 (full batch).

Plot the training loss curve for each.

Which converges fastest? Which is smoothest?

### 2

Implement SGD with momentum from scratch (without using `torch.optim`).

Verify that your implementation matches `torch.optim.SGD(momentum=0.9)`.

### 3

Use `ReduceLROnPlateau` with a validation loss.

Monitor the learning rate over training.

At what point does the scheduler reduce the learning rate?

### 4

Find a configuration of batch size, learning rate, and optimizer that leads to training instability (NaN loss).

Then fix it using gradient clipping.


## Summary

Stochastic gradient descent computes gradient estimates from mini-batches rather than the full dataset.

Extensions to basic SGD:

- **Momentum**: accumulates gradient history to smooth updates.
- **Adam**: maintains per-parameter adaptive learning rates.
- **Learning rate schedules**: reduce the step size over training.
- **Gradient clipping**: prevents exploding gradients.

The mini-batch training loop is the practical implementation of the maximum likelihood objective from Lesson 22.

By the end of this lesson, you have all the components of a complete deep learning training pipeline:

- tensors and memory (Lessons 1–12);
- probability and information theory (Lessons 13–15);
- numerical stability and autograd (Lessons 16–18);
- datasets, generalization, and regularization (Lessons 19–22);
- stochastic optimization (this lesson).

You are now fully prepared to study Chapter 6 — Deep Feedforward Networks.


## Next Lesson

**Lesson 20 — Deep Feedforward Networks**

Chapter 6 of the *Deep Learning* book introduces the first deep architecture.

We'll implement a feedforward neural network from scratch and then using `torch.nn`, connect every component to the mathematical notation in the book, and investigate how depth changes the representational capacity.
