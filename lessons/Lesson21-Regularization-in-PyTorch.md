# Lesson 21 — Regularization in PyTorch

| | |
|---|---|
| **Chapter** | 5 — Machine Learning Basics |
| **Reading Assignment** | Section 5.2.2 (Regularization) and Section 7.1–7.2 (preview) |

This lesson implements the regularization techniques introduced in Chapter 5 and connects each to its mathematical motivation.

## Objectives

By the end of this lesson you should be able to:

- implement L1 and L2 weight decay from scratch;
- use `weight_decay` in PyTorch optimizers;
- implement and apply dropout;
- understand early stopping as a form of regularization;
- explain why regularization reduces overfitting without reducing model capacity.


## Motivation

Regularization modifies the training procedure to improve generalization.

Instead of minimizing only the loss:

```
minimize L(θ)
```

regularization minimizes a penalized objective:

```
minimize L(θ) + λ Ω(θ)
```

where `Ω(θ)` is a penalty on the parameters and `λ` controls its strength.

This discourages the model from learning overly complex solutions.


## L2 Regularization (Weight Decay)

L2 regularization adds the sum of squared weights to the loss:

```
Ω(θ) = ‖θ‖² = Σ θᵢ²
```

The combined objective is:

```
L_reg = L + λ/2 * ‖θ‖²
```

The gradient of the penalty term is `λθ`, which shrinks weights toward zero at each step.

This is why L2 regularization is also called **weight decay**.


## Investigation 1 — L2 Regularization from Scratch

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# Overfit setup: many parameters, few samples
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

x_train = torch.linspace(-1, 1, 20).unsqueeze(1)
y_train = x_train ** 2 + 0.1 * torch.randn_like(x_train)

x_test  = torch.linspace(-1, 1, 100).unsqueeze(1)
y_test  = x_test ** 2

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

lambda_l2 = 0.001

for epoch in range(500):
    optimizer.zero_grad()

    pred = model(x_train)
    loss = criterion(pred, y_train)

    # Add L2 penalty manually
    l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
    total_loss = loss + lambda_l2 * l2_penalty

    total_loss.backward()
    optimizer.step()

with torch.no_grad():
    test_loss = criterion(model(x_test), y_test)
    print(f"Test MSE with L2 penalty: {test_loss:.6f}")
```


## Using weight_decay in Optimizers

PyTorch implements L2 regularization natively through the `weight_decay` argument:

```python
optimizer_l2 = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=0.001   # This is λ
)
```

The update step becomes:

```
θ ← θ - lr * (∇L + λ * θ)
  = θ * (1 - lr * λ) - lr * ∇L
```

The `weight_decay` argument is numerically equivalent to the manual L2 penalty above.


## L1 Regularization

L1 regularization adds the sum of absolute weights:

```
Ω(θ) = ‖θ‖₁ = Σ |θᵢ|
```

L1 promotes **sparsity**: it encourages many weights to become exactly zero.

```python
lambda_l1 = 0.001

for epoch in range(500):
    optimizer.zero_grad()

    pred = model(x_train)
    loss = criterion(pred, y_train)

    # L1 penalty
    l1_penalty = sum(p.abs().sum() for p in model.parameters())
    total_loss = loss + lambda_l1 * l1_penalty

    total_loss.backward()
    optimizer.step()
```

L1 has no direct `weight_decay` equivalent in PyTorch — it must be implemented manually.

Note: PyTorch's `weight_decay` is L2, not L1.


## Investigation 2 — Dropout

Dropout randomly sets a fraction of activations to zero during training.

```python
model_with_dropout = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Dropout(p=0.5),    # 50% of neurons zeroed each step
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(64, 1)
)
```

During training, dropout:

1. randomly zeros each neuron with probability `p`;
2. scales the remaining activations by `1 / (1-p)` to keep the expected output unchanged.

During inference, dropout is disabled:

```python
model_with_dropout.train()   # dropout active
model_with_dropout.eval()    # dropout inactive
```

**Always call `model.eval()` before evaluation.**

Forgetting this is a common bug — the model will appear to have high variance on the test set.


## Investigation 3 — Verifying Dropout Behavior

```python
model_with_dropout.train()

x = torch.randn(1, 1)

outputs = [model_with_dropout(x).item() for _ in range(5)]
print("Train mode (random):", outputs)

model_with_dropout.eval()

outputs = [model_with_dropout(x).item() for _ in range(5)]
print("Eval mode (fixed)  :", outputs)
```

In train mode, outputs vary due to dropout.

In eval mode, outputs are deterministic.


## Early Stopping

Early stopping monitors validation loss and stops training when it begins to increase.

```python
best_val_loss = float('inf')
patience      = 10
patience_counter = 0

for epoch in range(1000):
    # --- training step ---
    model.train()
    optimizer.zero_grad()
    pred = model(x_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    # --- validation step ---
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(x_test), y_test).item()

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        # Optionally save model state here
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        model.load_state_dict(best_state)
        break
```

Early stopping is equivalent to constraining the training time, which implicitly limits model complexity.


## Companion Insight

The *Deep Learning* book frames regularization as modifying the learning procedure.

From an implementation perspective, there are three mechanisms:

1. **Parameter constraints** (L1, L2): modify the loss function.
2. **Noise injection** (Dropout): modify the forward pass.
3. **Training time constraints** (Early Stopping): modify the optimization loop.

All three reduce the effective capacity of the model without changing its architecture.

This is the practical meaning of regularization:

> Make the model learn only what the data supports, not what it could memorize.


## Source Reading

Open:

```
torch/nn/modules/dropout.py
```

Read the `Dropout` class.

Find where `training` mode controls whether the dropout mask is applied.

Trace the call to `F.dropout()` in:

```
torch/nn/functional.py
```

Observe that the actual implementation is:

```python
if training:
    mask = ... Bernoulli(1 - p) ...
    return input * mask / (1 - p)
else:
    return input
```


## Exercises

### 1

Train two models on the same small dataset with and without L2 regularization.

Compare the magnitude of their weights.

Which has smaller weights?

### 2

Train a model with `Dropout(p=0.5)`.

Evaluate it in both `train()` and `eval()` modes on the test set.

Report the test loss for each.

Why is the `eval()` result the correct one?

### 3

Implement early stopping with a patience of 5 epochs.

Plot the training and validation loss curves.

Mark the epoch where training stopped.

### 4

Explain why L1 regularization tends to produce sparse weights while L2 regularization does not.

Hint: think about the gradient of each penalty at zero.


## Summary

Regularization prevents overfitting by constraining what the model learns.

PyTorch implements the main techniques through:

- `weight_decay` in optimizers (L2 regularization);
- `nn.Dropout` (stochastic activation zeroing);
- manual L1 penalty computation;
- custom early stopping logic.

The three approaches operate at different levels: parameter values, activations, and training time.


## Next Lesson

**Lesson 22 — Maximum Likelihood Estimation and Bayesian Inference**

Chapter 5 provides a rigorous probabilistic foundation for learning.

We'll connect MLE to the loss functions already studied and introduce the Bayesian perspective on learning, culminating in MAP estimation and its relationship to L2 regularization.
