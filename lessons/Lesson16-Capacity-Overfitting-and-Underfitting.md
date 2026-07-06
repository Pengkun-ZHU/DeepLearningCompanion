# Lesson 16 — Capacity, Overfitting, and Underfitting

| | |
|---|---|
| **Chapter** | 5 — Machine Learning Basics |
| **Reading Assignment** | Sections 5.2–5.4 (Capacity, Overfitting, Underfitting, Hyperparameters and Validation Sets) |

This lesson makes overfitting and underfitting directly observable through PyTorch experiments.

## Objectives

By the end of this lesson you should be able to:

- explain model capacity and how it relates to overfitting and underfitting;
- observe the training–generalization gap experimentally;
- implement a polynomial regression model in PyTorch;
- use a validation set to select model capacity;
- explain the relationship between the number of parameters and the risk of overfitting.


## Motivation

The *Deep Learning* book introduces one of the central challenges in machine learning:

> The goal of machine learning is to perform well on **new, unseen data** — not just on the training data.

A model that performs perfectly on training data but poorly on new data has **overfit**.

A model that performs poorly on both has **underfit**.

Understanding this distinction is the foundation of model design.


## The Bias-Variance Intuition

A model with too little capacity cannot fit the data (high bias, underfitting).

A model with too much capacity fits the noise in the training data (high variance, overfitting).

The task is to find the right capacity for the data.


## Investigation 1 — Polynomial Regression

We'll fit polynomials of different degrees to noisy data from `y = sin(2πx)`.

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# True function: sin(2*pi*x), x in [0, 1]
torch.manual_seed(0)
n = 50
x = torch.linspace(0, 1, n)
y = torch.sin(2 * torch.pi * x) + 0.3 * torch.randn(n)

def make_poly_features(x, degree):
    """Create polynomial features [x, x^2, ..., x^degree]."""
    return torch.stack([x**d for d in range(1, degree+1)], dim=1)


def fit_polynomial(degree, x_train, y_train, x_test, y_test, epochs=2000, lr=0.01):
    X_train = make_poly_features(x_train, degree)
    X_test  = make_poly_features(x_test,  degree)

    model = nn.Linear(degree, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_loss = criterion(model(X_train), y_train.unsqueeze(1)).item()
        test_loss  = criterion(model(X_test),  y_test.unsqueeze(1)).item()

    return train_loss, test_loss


# Split
n_train = 40
x_train, y_train = x[:n_train], y[:n_train]
x_test,  y_test  = x[n_train:], y[n_train:]

print(f"{'Degree':>8} | {'Train MSE':>12} | {'Test MSE':>12}")
print("-" * 40)

for degree in [1, 2, 3, 5, 9, 15]:
    train_mse, test_mse = fit_polynomial(degree, x_train, y_train, x_test, y_test)
    print(f"{degree:>8} | {train_mse:>12.6f} | {test_mse:>12.6f}")
```

Expected observations:

- **Degree 1**: both train and test MSE are high. Underfitting.
- **Degree 3**: train and test MSE are both low. Good fit.
- **Degree 15**: train MSE is very low, test MSE is high. Overfitting.


## The Generalization Gap

The generalization gap is:

```
test error − train error
```

A small gap means the model generalizes well.

A large gap means the model has overfit.

The *Deep Learning* book expresses this in terms of expected risk (generalization error) versus empirical risk (training error).


## Investigation 2 — Validation Set

The test set must never be used to make decisions about the model.

A **validation set** is held out from training and used for model selection.

```python
# Using a validation set to choose polynomial degree

torch.manual_seed(0)
n = 100
x = torch.linspace(0, 1, n)
y = torch.sin(2 * torch.pi * x) + 0.3 * torch.randn(n)

n_train = 60
n_val   = 20
# n_test  = 20

x_train = x[:n_train];  y_train = y[:n_train]
x_val   = x[n_train:n_train+n_val]; y_val = y[n_train:n_train+n_val]
x_test  = x[n_train+n_val:];        y_test = y[n_train+n_val:]

best_degree    = None
best_val_error = float('inf')

print(f"{'Degree':>8} | {'Train MSE':>12} | {'Val MSE':>12}")
print("-" * 40)

for degree in [1, 2, 3, 4, 5, 7, 9]:
    train_mse, val_mse = fit_polynomial(degree, x_train, y_train, x_val, y_val)
    print(f"{degree:>8} | {train_mse:>12.6f} | {val_mse:>12.6f}")

    if val_mse < best_val_error:
        best_val_error = val_mse
        best_degree    = degree

print(f"\nBest degree by validation: {best_degree}")

# Report final test error only once
_, test_mse = fit_polynomial(best_degree, x_train, y_train, x_test, y_test)
print(f"Final test MSE           : {test_mse:.6f}")
```


## Capacity and Parameters

For neural networks, capacity is approximately proportional to the number of parameters.

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

small  = nn.Sequential(nn.Linear(10, 8),   nn.ReLU(), nn.Linear(8, 1))
medium = nn.Sequential(nn.Linear(10, 64),  nn.ReLU(), nn.Linear(64, 1))
large  = nn.Sequential(nn.Linear(10, 512), nn.ReLU(), nn.Linear(512, 1))

for name, model in [('small', small), ('medium', medium), ('large', large)]:
    print(f"{name:>8}: {count_parameters(model):>8} parameters")
```

More parameters do not automatically mean better performance.

They mean higher risk of overfitting when training data is limited.


## Companion Insight

The *Deep Learning* book shows that the optimal capacity depends on the amount of training data.

This is captured by the **double descent** phenomenon:

For very high capacity, test error sometimes decreases again after increasing — a behavior first noted in modern overparameterized networks.

The takeaway is not a fixed rule but a guiding principle:

> Monitor both training and test/validation error. If they diverge, you have a capacity problem.


## Source Reading

Read the documentation for:

```
torch.utils.data.random_split
```

Then open:

```
torch/utils/data/dataset.py
```

Find `Subset` and `random_split`.

Understand how a `Subset` wraps an existing dataset without copying data.


## Exercises

### 1

Fit a polynomial of degree 1, 3, and 9 to the sine function dataset.

For each degree:

- plot the fitted curve alongside the data points;
- report train and test MSE;
- classify each as underfitting, good fit, or overfitting.

### 2

As the number of training samples increases from 10 to 500, how does the degree of polynomial that achieves the best validation performance change?

Design an experiment to answer this question.

### 3

Explain the following statement from the *Deep Learning* book:

> Underfitting and overfitting are determined by the relationship between model capacity and the complexity of the data-generating distribution.

### 4

Build two neural networks: one with 1,000 parameters and one with 1,000,000 parameters.

Train both on a small dataset (50 samples).

What do you observe on the test set?


## Summary

Model capacity controls the trade-off between underfitting and overfitting.

- Too little capacity: the model cannot express the true relationship (underfitting).
- Too much capacity: the model memorizes noise in the training set (overfitting).

The validation set allows selecting capacity without contaminating the test set.

Train error and test error tell different stories — both must be monitored.


## Next Lesson

**Lesson 17 — Regularization in PyTorch**

Chapter 5 describes regularization techniques that reduce overfitting without reducing model capacity.

We'll implement L1 and L2 weight penalties, dropout, and early stopping, connecting each technique to its mathematical motivation.
