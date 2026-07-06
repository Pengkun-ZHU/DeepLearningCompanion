# Lesson 19 — Learning Algorithms and the ML Pipeline

| | |
|---|---|
| **Chapter** | 5 — Machine Learning Basics |
| **Reading Assignment** | Sections 5.1–5.2 (Learning Algorithms, Capacity, Overfitting, Underfitting) |

This lesson connects the rigorous definitions of Chapter 5 to a concrete PyTorch data pipeline.

## Objectives

By the end of this lesson you should be able to:

- explain the components of a learning algorithm: task, performance measure, experience;
- implement a training and evaluation loop in PyTorch;
- construct a dataset and DataLoader;
- distinguish training error from generalization error;
- explain why a separate test set is necessary.


## Motivation

The *Deep Learning* book defines a machine learning algorithm as:

> A program that learns from experience E with respect to task T and performance measure P, and improves P on task T with experience E.

This definition is deliberately abstract.

In PyTorch, the components map as follows:

| Book Concept | PyTorch Implementation |
|---|---|
| Task | Model architecture |
| Experience | Dataset / DataLoader |
| Performance measure | Loss function + metrics |
| Learning | Optimizer + training loop |


## The Dataset Abstraction

PyTorch represents datasets through two classes:

```python
from torch.utils.data import Dataset, DataLoader
```

`Dataset` knows how to return one sample.

`DataLoader` knows how to batch many samples together.


## Investigation 1 — Custom Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class LinearDataset(Dataset):
    """y = 3x + 2 + noise"""

    def __init__(self, n=1000, noise=0.5):
        torch.manual_seed(0)
        self.x = torch.linspace(-3, 3, n).unsqueeze(1)
        self.y = 3.0 * self.x + 2.0 + noise * torch.randn_like(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = LinearDataset()

print(f"Dataset size: {len(dataset)}")
print(f"Sample: x={dataset[0][0].item():.4f}, y={dataset[0][1].item():.4f}")
```


## Training and Test Splits

The training set is used to fit the model.

The test set is used to measure generalization.

They must never overlap.

```python
from torch.utils.data import random_split

full_dataset = LinearDataset(n=1000)

train_size = 800
test_size  = 200

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train samples: {len(train_dataset)}")
print(f"Test  samples: {len(test_dataset)}")
```


## DataLoader

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# Inspect one batch
x_batch, y_batch = next(iter(train_loader))
print(f"Batch x shape: {x_batch.shape}")
print(f"Batch y shape: {y_batch.shape}")
```

`shuffle=True` randomizes the order each epoch.

This is essential for stochastic gradient descent to work correctly.


## Investigation 2 — A Complete Training Loop

```python
import torch.nn as nn

# Model
model = nn.Linear(1, 1)

# Loss
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(20):
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluation
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            test_loss += loss.item()

    test_loss /= len(test_loader)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d}: train={train_loss:.4f}, test={test_loss:.4f}")

# Inspect learned parameters
print(f"\nLearned weight: {model.weight.item():.4f} (true: 3.0)")
print(f"Learned bias  : {model.bias.item():.4f} (true: 2.0)")
```


## train() vs eval() mode

```python
model.train()   # enables dropout, batch normalization running stats
model.eval()    # disables dropout, uses stored batch norm statistics
```

Always call `model.eval()` before evaluation and `model.train()` before training.

This matters even for simple models that don't use dropout.

It is a good habit to develop now.


## Companion Insight

The training loop pattern is the same regardless of model complexity:

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
```

This loop will be recognizable in every neural network implementation you encounter.

The architecture, loss, and optimizer change.

The loop structure does not.


## Source Reading

Open:

```
torch/utils/data/dataloader.py
```

Find the `__iter__` method.

Observe that it returns a `_SingleProcessDataLoaderIter` or `_MultiProcessingDataLoaderIter` depending on `num_workers`.

The key operation is the `collate_fn` which converts a list of samples into a batch tensor.


## Exercises

### 1

Create a dataset for a sine function:

```python
y = sin(2πx) + noise
```

Train a linear model on it.

Does the linear model fit the data well?

Why or why not?

### 2

Implement a validation split in addition to train and test splits.

Use the validation set to monitor training and identify when the model starts to overfit.

### 3

Add a metric (e.g. mean absolute error) to the evaluation loop alongside the MSE loss.

### 4

Explain in your own words why it is incorrect to use the test set during hyperparameter tuning.

What should you use instead?


## Summary

A machine learning algorithm consists of three components: task, experience, and performance measure.

In PyTorch, these map to:

- model architecture (task);
- Dataset and DataLoader (experience);
- loss function and metrics (performance measure).

The training loop is the mechanism that connects them.

It is not complex, but it must be implemented carefully: zero gradients, compute loss, backpropagate, step, evaluate in `no_grad` mode.


## Next Lesson

**Lesson 20 — Capacity, Overfitting, and Underfitting**

Chapter 5 continues by analyzing why models succeed or fail to generalize.

We'll investigate overfitting and underfitting experimentally, connecting the book's theoretical analysis to observable behavior in PyTorch.
