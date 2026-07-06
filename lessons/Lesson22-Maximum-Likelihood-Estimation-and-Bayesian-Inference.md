# Lesson 22 — Maximum Likelihood Estimation and Bayesian Inference

| | |
|---|---|
| **Chapter** | 5 — Machine Learning Basics |
| **Reading Assignment** | Sections 5.5–5.7 (MLE, Bayesian Statistics, Supervised Learning Algorithms) |

This lesson connects maximum likelihood estimation and Bayesian inference to the loss functions and regularization techniques already implemented.

## Objectives

By the end of this lesson you should be able to:

- derive the connection between MLE and cross-entropy loss;
- implement MLE for a linear regression model in PyTorch;
- explain the Bayesian perspective on learning;
- connect L2 regularization to a Gaussian prior on weights (MAP estimation);
- evaluate log-likelihood as a metric for model comparison.


## Motivation

Most deep learning models are trained by maximum likelihood estimation, even when this is not stated explicitly.

Understanding MLE reveals:

- why cross-entropy is the right loss for classification;
- why mean squared error is the right loss for regression under Gaussian noise assumptions;
- why L2 regularization is equivalent to placing a Gaussian prior on weights;
- how to interpret the training objective probabilistically.


## Maximum Likelihood Estimation

Given data `{x₁,...,xₙ}` drawn from distribution `p_data(x)`, MLE finds the parameter `θ` that maximizes the probability of the observed data:

```
θ_MLE = argmax_θ  Π p_model(xᵢ; θ)
       = argmax_θ  Σ log p_model(xᵢ; θ)
```

The product becomes a sum in log space — a numerical stability benefit we saw in Lesson 16.


## Investigation 1 — MLE for Linear Regression

For a linear regression model with Gaussian noise:

```
y = w·x + b + ε,   ε ~ N(0, σ²)
```

the log-likelihood is:

```
log p(y|x; w, b) = -n/2 * log(2πσ²) - 1/(2σ²) * Σ (yᵢ - w·xᵢ - b)²
```

Maximizing this is equivalent to minimizing mean squared error.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# Generate data from y = 2x + 1 + noise
n = 200
x = torch.randn(n, 1)
y = 2.0 * x + 1.0 + 0.5 * torch.randn(n, 1)

model     = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
mse_loss  = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    optimizer.step()

print(f"MLE estimate: w = {model.weight.item():.4f}, b = {model.bias.item():.4f}")
print(f"True values : w = 2.0000, b = 1.0000")
```

Minimizing MSE = maximizing log-likelihood under Gaussian noise assumption.


## Negative Log-Likelihood as a Loss Function

PyTorch provides `nn.NLLLoss` for explicitly working with log-probabilities.

```python
# Manual NLL for classification
logits = torch.randn(5, 3)   # 5 samples, 3 classes
labels = torch.randint(0, 3, (5,))

log_probs = torch.log_softmax(logits, dim=1)

nll = nn.NLLLoss()
loss = nll(log_probs, labels)
print(f"NLL Loss: {loss:.4f}")

# Equivalent using CrossEntropyLoss (which combines log_softmax + nll)
ce = nn.CrossEntropyLoss()
loss2 = ce(logits, labels)
print(f"CE  Loss: {loss2:.4f}")
```


## Bayesian Perspective

The Bayesian framework adds a prior over parameters:

```
p(θ | data) ∝ p(data | θ) * p(θ)
```

MAP (Maximum A Posteriori) estimation finds:

```
θ_MAP = argmax_θ  log p(data | θ) + log p(θ)
```


## Investigation 2 — MAP Estimation and L2 Regularization

A Gaussian prior `θ ~ N(0, 1/λ)` corresponds to:

```
log p(θ) = -λ/2 * ‖θ‖²  + const
```

Therefore:

```
θ_MAP = argmax  log-likelihood - λ/2 * ‖θ‖²
      = argmin  negative log-likelihood + λ/2 * ‖θ‖²
```

This is exactly **L2 regularized loss**.

```python
# MAP estimation via L2-regularized MLE
model     = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.1)
mse_loss  = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    optimizer.step()

print(f"MAP estimate: w = {model.weight.item():.4f}, b = {model.bias.item():.4f}")
print(f"Note: MAP shrinks weights toward 0 compared to MLE")
```

The Bayesian view explains *why* L2 regularization works: it encodes the prior belief that weights should be small.


## Prior Strength and Regularization Strength

| Bayesian Language | Optimization Language |
|---|---|
| Strong prior on θ ~ N(0, σ²_prior) | Large λ (weight_decay) |
| Weak prior | Small λ |
| Prior variance → ∞ | No regularization (pure MLE) |

```python
for lambda_reg in [0.0, 0.01, 0.1, 1.0]:
    model     = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=lambda_reg)
    mse_loss  = nn.MSELoss()

    for _ in range(500):
        optimizer.zero_grad()
        loss = mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

    w = model.weight.item()
    b = model.bias.item()
    print(f"λ={lambda_reg:.2f}: w={w:.4f}, b={b:.4f}")
```

Observe that larger `weight_decay` pulls the weights closer to zero.


## Log-Likelihood as a Model Comparison Metric

```python
def log_likelihood_gaussian(model, x, y, sigma=0.5):
    with torch.no_grad():
        pred = model(x)
        dist = torch.distributions.Normal(pred, sigma)
        return dist.log_prob(y).sum().item()

# Compare two models
model_good = nn.Linear(1, 1)
model_bad  = nn.Linear(1, 1)

# Set parameters manually
model_good.weight.data.fill_(2.0)
model_good.bias.data.fill_(1.0)

model_bad.weight.data.fill_(0.0)
model_bad.bias.data.fill_(0.0)

print(f"Good model log-likelihood: {log_likelihood_gaussian(model_good, x, y):.2f}")
print(f"Bad  model log-likelihood: {log_likelihood_gaussian(model_bad,  x, y):.2f}")
```

Higher log-likelihood means the model assigns higher probability to the observed data.


## Companion Insight

The probabilistic view of deep learning unifies all the optimization objectives:

```
Task            →   Output distribution   →   Loss function
────────────────────────────────────────────────────────────
Binary class.   →   Bernoulli             →   Binary cross-entropy
Multi-class     →   Categorical           →   Cross-entropy
Regression      →   Gaussian              →   MSE
Robust regr.    →   Laplace               →   MAE
```

And regularization has a Bayesian interpretation:

```
Regularizer      →   Prior on weights
──────────────────────────────────────
L2 (weight decay) →   Gaussian prior
L1               →   Laplace prior
```

Understanding these connections makes loss function and regularization choices principled rather than arbitrary.


## Source Reading

Read Section 5.5 of the *Deep Learning* book again.

Then locate in PyTorch:

```
torch/optim/sgd.py
```

Find how `weight_decay` is added to the gradient before the parameter update.

Verify that it matches the gradient of the L2 penalty:

```
∂/∂θ (λ/2 * ‖θ‖²) = λ * θ
```


## Exercises

### 1

Show algebraically that minimizing MSE for a linear model with Gaussian noise is equivalent to maximizing the log-likelihood.

Then verify numerically using PyTorch.

### 2

Fit a logistic regression model to a two-class dataset using `nn.BCEWithLogitsLoss`.

Explain why this is MLE under a Bernoulli distribution assumption.

### 3

Train the same model with three values of `weight_decay`: 0, 0.01, and 1.0.

Compare the weight norms:

```python
torch.linalg.norm(torch.cat([p.flatten() for p in model.parameters()]))
```

### 4

A Laplace prior `θ ~ Laplace(0, 1/λ)` corresponds to L1 regularization.

Verify this claim by deriving the MAP objective for a Laplace prior.


## Summary

Maximum likelihood estimation provides the probabilistic foundation for deep learning loss functions.

MAP estimation extends MLE with a prior on parameters, which is precisely equivalent to regularized loss minimization.

Key correspondences:

- MSE ↔ MLE under Gaussian noise assumption
- Cross-entropy ↔ MLE for Categorical outputs
- L2 regularization ↔ Gaussian prior on weights
- L1 regularization ↔ Laplace prior on weights

These are not coincidences — they are the same optimization problem expressed in different languages.


## Next Lesson

**Lesson 23 — Stochastic Gradient Descent in PyTorch**

Chapter 5 concludes with a discussion of stochastic gradient descent.

We'll study why stochasticity helps, how mini-batches work, and how momentum and adaptive methods like Adam extend the basic SGD update.
