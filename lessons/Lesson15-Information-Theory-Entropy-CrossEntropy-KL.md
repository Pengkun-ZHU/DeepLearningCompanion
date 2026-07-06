# Lesson 15 — Information Theory: Entropy, Cross-Entropy, and KL Divergence

| | |
|---|---|
| **Chapter** | 3 — Probability and Information Theory |
| **Reading Assignment** | Sections 3.13–3.14 (Information Theory, Structured Probabilistic Models) |

This lesson connects information theory to the loss functions used throughout deep learning.

## Objectives

By the end of this lesson you should be able to:

- compute Shannon entropy in PyTorch;
- explain what entropy measures intuitively;
- implement cross-entropy loss from first principles;
- understand KL divergence and what it measures;
- relate these quantities to the standard loss functions in `torch.nn`.


## Motivation

Cross-entropy is one of the most commonly used loss functions in deep learning.

Most practitioners use it without understanding its origin.

It comes directly from information theory.

Understanding this connection tells you:

- why cross-entropy is the right loss for classification;
- what the model is trying to minimize;
- how to reason about KL divergence as a regularizer in variational autoencoders and other probabilistic models.


## Entropy

Shannon entropy measures the expected amount of information (surprise) in a probability distribution.

```
H(P) = -Σ P(x) log P(x)
```

A uniform distribution has high entropy — outcomes are maximally uncertain.

A peaked distribution has low entropy — outcomes are predictable.


## Investigation 1 — Computing Entropy

```python
import torch

def entropy(probs):
    # Avoid log(0) by filtering zero probabilities
    log_probs = torch.where(probs > 0,
                            torch.log(probs),
                            torch.zeros_like(probs))
    return -(probs * log_probs).sum()

# Maximally uncertain: uniform over 4 classes
uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
print(f"Entropy (uniform) : {entropy(uniform):.4f}")  # log(4) ≈ 1.386

# Certain outcome
certain = torch.tensor([1.0, 0.0, 0.0, 0.0])
print(f"Entropy (certain) : {entropy(certain):.4f}")  # 0.0

# Somewhere between
mixed = torch.tensor([0.7, 0.2, 0.08, 0.02])
print(f"Entropy (mixed)   : {entropy(mixed):.4f}")
```

Entropy is measured in **nats** when using the natural logarithm (as PyTorch does by default), or **bits** when using log base 2.


## Entropy via `torch.distributions`

```python
dist = torch.distributions.Categorical(probs=torch.tensor([0.1, 0.4, 0.5]))

print(f"Entropy: {dist.entropy():.4f}")
```

The `entropy()` method is available on every distribution in `torch.distributions`.


## Cross-Entropy

Cross-entropy measures the expected code length when encoding outcomes from distribution `P` using a code optimized for distribution `Q`:

```
H(P, Q) = -Σ P(x) log Q(x)
```

When `Q` is the model's prediction and `P` is the true distribution (usually a one-hot vector), cross-entropy becomes the standard classification loss.


## Investigation 2 — Cross-Entropy as a Loss Function

```python
# True label: class 2 (one-hot)
y_true = torch.tensor([0.0, 0.0, 1.0])

# Model predictions (probabilities after softmax)
y_pred_good = torch.tensor([0.05, 0.05, 0.90])
y_pred_bad  = torch.tensor([0.33, 0.33, 0.34])

def cross_entropy(p_true, q_pred):
    log_q = torch.log(q_pred)
    return -(p_true * log_q).sum()

print(f"Good prediction CE: {cross_entropy(y_true, y_pred_good):.4f}")  # low
print(f"Bad  prediction CE: {cross_entropy(y_true, y_pred_bad ):.4f}")  # high
```

Because `y_true` is one-hot, cross-entropy simplifies to:

```python
# -log(predicted probability of the correct class)
loss = -torch.log(y_pred_good[2])
print(f"Simplified CE: {loss:.4f}")
```

This is exactly what `torch.nn.NLLLoss` and `torch.nn.CrossEntropyLoss` compute.


## PyTorch's Cross-Entropy Loss

```python
# CrossEntropyLoss takes raw logits (not probabilities)
logits = torch.tensor([[2.0, 1.0, 4.0]])   # shape (batch, classes)
labels = torch.tensor([2])                  # correct class index

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)

print(f"CrossEntropyLoss: {loss:.4f}")

# Manual computation
probs = torch.softmax(logits, dim=1)
manual_ce = -torch.log(probs[0, 2])
print(f"Manual CE       : {manual_ce:.4f}")
```

Note that `CrossEntropyLoss` applies `log_softmax` internally for numerical stability.

Never apply `softmax` before passing logits to `CrossEntropyLoss`.


## KL Divergence

KL divergence measures how different distribution `Q` is from distribution `P`:

```
KL(P ‖ Q) = Σ P(x) log(P(x) / Q(x))
```

It is always non-negative and equals zero if and only if `P = Q`.

It is not symmetric: `KL(P ‖ Q) ≠ KL(Q ‖ P)`.

The relationship to cross-entropy is:

```
H(P, Q) = H(P) + KL(P ‖ Q)
```

Minimizing cross-entropy is equivalent to minimizing KL divergence when the true entropy `H(P)` is fixed.


## Investigation 3 — KL Divergence

```python
def kl_divergence(p, q):
    # KL(P || Q)
    return (p * (torch.log(p) - torch.log(q))).sum()

p = torch.tensor([0.4, 0.4, 0.2])
q = torch.tensor([0.3, 0.3, 0.4])

print(f"KL(P||Q) = {kl_divergence(p, q):.4f}")
print(f"KL(Q||P) = {kl_divergence(q, p):.4f}")

# Verify non-negativity
print(f"KL ≥ 0: {kl_divergence(p, q) >= 0}")

# KL = 0 when P = Q
print(f"KL(P||P) = {kl_divergence(p, p):.4f}")
```

PyTorch provides `torch.nn.functional.kl_div()`.

Note that it expects log-probabilities for the predicted distribution.


## Companion Insight

There is a deep connection between information theory and maximum likelihood estimation.

Minimizing cross-entropy loss is equivalent to:

1. Maximizing the log-likelihood of the training data.
2. Minimizing the KL divergence from the model distribution to the empirical data distribution.

This is why cross-entropy is the natural loss function for classification — not an arbitrary choice, but a direct consequence of the probabilistic interpretation of learning.

In variational autoencoders and other latent variable models, KL divergence appears explicitly as a regularization term that encourages the learned latent distribution to remain close to a prior.


## Source Reading

Open:

```
torch/nn/modules/loss.py
```

Find `CrossEntropyLoss`.

Observe that it:

1. calls `F.cross_entropy()`;
2. which calls `F.nll_loss()` with `log_softmax()` applied to the input.

The separation between `log_softmax` and `nll_loss` makes the numerical stability improvements explicit.

Compare this implementation to the formula:

```
CrossEntropy(logits, y) = -logits[y] + log(Σ exp(logits))
```


## Exercises

### 1

Verify numerically that `H(P, Q) = H(P) + KL(P ‖ Q)` for two arbitrary distributions.

### 2

For a two-class classifier, show that binary cross-entropy is a special case of cross-entropy.

Implement binary cross-entropy from scratch and compare with `torch.nn.BCELoss`.

### 3

Experiment with the following scenario:

```python
logits = torch.tensor([[10.0, 0.0, 0.0]])
labels = torch.tensor([0])
```

What happens to the loss?

Why is it important to use logits rather than post-softmax probabilities with `CrossEntropyLoss`?

### 4

Explain the difference between

```python
torch.nn.CrossEntropyLoss()
```

and

```python
torch.nn.NLLLoss()
```

When would you use each?


## Summary

Information theory provides the mathematical foundation for the most common loss functions in deep learning.

- Entropy measures the uncertainty of a distribution.
- Cross-entropy measures the cost of encoding one distribution with a code designed for another.
- KL divergence measures the information lost when approximating one distribution with another.
- Minimizing cross-entropy equals minimizing KL divergence equals maximizing likelihood.

This chain of equivalences is why cross-entropy is not merely a convenient heuristic — it is the principled choice for probabilistic models.


## Next Lesson

**Lesson 16 — Numerical Precision and Stability**

Chapter 4 of the *Deep Learning* book addresses numerical computation.

We'll investigate how floating-point arithmetic works in PyTorch, why overflow and underflow are constant concerns, and how framework implementations (including the `CrossEntropyLoss` we just studied) are designed to remain numerically stable.
