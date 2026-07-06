# Lesson 14 — Common Probability Distributions in PyTorch

| | |
|---|---|
| **Chapter** | 3 — Probability and Information Theory |
| **Reading Assignment** | Section 3.9 — Common Probability Distributions |

This lesson surveys the probability distributions most commonly used in deep learning and implements each one in PyTorch.

## Objectives

By the end of this lesson you should be able to:

- describe the Bernoulli, Categorical, Gaussian, and Laplace distributions;
- choose the appropriate output distribution for a given prediction task;
- evaluate log-probabilities and draw samples in PyTorch;
- relate distribution choice to the shape of the loss function;
- understand how mixture distributions generalize single distributions.


## Motivation

The choice of output distribution determines the loss function.

This is one of the most important design decisions in deep learning.

Consider these common choices:

| Task | Distribution | Loss |
|---|---|---|
| Binary classification | Bernoulli | Binary cross-entropy |
| Multi-class classification | Categorical | Cross-entropy |
| Regression | Gaussian | Mean squared error |
| Robust regression | Laplace | Mean absolute error |
| Image generation | Mixture of Gaussians | Negative log-likelihood |

The *Deep Learning* book covers these distributions because they are not just mathematical objects — they are model design choices.


## Bernoulli Distribution

The Bernoulli distribution models a single binary outcome.

```python
import torch

# P(y=1) = 0.7, P(y=0) = 0.3
dist = torch.distributions.Bernoulli(probs=0.7)

samples = dist.sample((10000,))
print(f"Empirical P(y=1): {samples.mean():.4f}")

# Log-probability of observing y=1
print(f"log P(y=1): {dist.log_prob(torch.tensor(1.0)):.4f}")
```

In neural network outputs, the probability parameter is typically the output of a sigmoid function.

```python
logit = torch.tensor(0.847)   # raw network output

prob = torch.sigmoid(logit)

dist = torch.distributions.Bernoulli(probs=prob)

print(f"P(y=1) = {prob:.4f}")
```


## Categorical Distribution

The Categorical distribution generalizes Bernoulli to multiple classes.

```python
probs = torch.tensor([0.1, 0.6, 0.3])

dist = torch.distributions.Categorical(probs=probs)

# Sample class indices
samples = dist.sample((10000,))

for k in range(3):
    print(f"Class {k}: {(samples == k).float().mean():.4f}")
```

In practice, neural networks produce unnormalized scores called **logits**.

The Categorical distribution accepts logits directly:

```python
logits = torch.tensor([1.0, 3.0, 2.0])

dist = torch.distributions.Categorical(logits=logits)

# Internally applies softmax
print(dist.probs)
```


## Investigation 1 — Gaussian Distribution

The Gaussian is the most prevalent distribution in deep learning.

```python
mu    = torch.tensor(3.0)
sigma = torch.tensor(1.5)

dist = torch.distributions.Normal(mu, sigma)

# Log-probability of observing x = 3.5
x = torch.tensor(3.5)
print(f"log p(x=3.5) = {dist.log_prob(x):.4f}")
print(f"p(x=3.5)     = {dist.log_prob(x).exp():.4f}")

# Sample
samples = dist.sample((5,))
print(samples)
```

Using a Gaussian output distribution is equivalent to minimizing mean squared error.

To see why, note that maximizing log P(y | x) for a Gaussian equals minimizing (y - μ)² / 2σ².


## Investigation 2 — Laplace Distribution

The Laplace distribution has heavier tails than the Gaussian.

```python
mu = torch.tensor(0.0)
b  = torch.tensor(1.0)

laplace = torch.distributions.Laplace(mu, b)
normal  = torch.distributions.Normal(mu, b)

x = torch.tensor([0.0, 1.0, 2.0, 3.0])

print("x     | Laplace log p | Normal log p")
for xi in x:
    lp_l = laplace.log_prob(xi).item()
    lp_n = normal.log_prob(xi).item()
    print(f"{xi.item():.1f}   | {lp_l:12.4f}  | {lp_n:12.4f}")
```

Observe that the Laplace distribution assigns higher probability to values far from the mean.

Using a Laplace output distribution is equivalent to minimizing mean absolute error.

This makes Laplace-based models more robust to outliers.


## Multinomial and Beta Distributions

```python
# Multinomial — counts from repeated Categorical trials
probs = torch.tensor([0.2, 0.5, 0.3])
dist = torch.distributions.Multinomial(total_count=10, probs=probs)

sample = dist.sample()
print(f"Counts: {sample}")  # sums to 10

# Beta — models a probability value in [0,1]
dist = torch.distributions.Beta(concentration1=2.0, concentration0=5.0)
samples = dist.sample((5,))
print(f"Beta samples: {samples}")
```


## Investigation 3 — Mixture of Gaussians

A single Gaussian cannot represent complex, multi-modal distributions.

A mixture of Gaussians combines several components:

```python
from torch.distributions import MixtureSameFamily, Categorical, Normal

# Two-component mixture
mix   = Categorical(probs=torch.tensor([0.3, 0.7]))
comp  = Normal(loc=torch.tensor([-2.0, 3.0]),
               scale=torch.tensor([0.5, 1.0]))

mixture = MixtureSameFamily(mix, comp)

samples = mixture.sample((5000,))

print(f"Mean of samples: {samples.mean():.4f}")
print(f"Std of samples : {samples.std():.4f}")
```

Mixture models are important in generative modeling and density estimation.


## Companion Insight

Each probability distribution encodes an assumption about the data-generating process.

When you choose a loss function, you are implicitly choosing a distribution.

```
Loss Function             Implied Distribution
----------------------------------------------
Mean Squared Error        Gaussian
Mean Absolute Error       Laplace
Binary Cross-Entropy      Bernoulli
Cross-Entropy             Categorical
```

This connection is not just theoretical.

It tells you:

- what your model assumes about the noise in the data;
- when your loss function will be robust or sensitive to outliers;
- how to extend your model when those assumptions are wrong.


## Source Reading

Open:

```
torch/distributions/__init__.py
```

List all available distributions.

Then open:

```
torch/distributions/normal.py
```

Read the implementation of `log_prob()`.

Compare it to the formula in the *Deep Learning* book:

```
log N(x; μ, σ) = -0.5 log(2π) - log(σ) - (x-μ)² / (2σ²)
```

Verify that the code matches the formula.


## Exercises

### 1

For each scenario, choose the appropriate output distribution and explain why.

- Predicting house prices.
- Predicting whether an email is spam.
- Predicting which digit (0–9) an image contains.
- Predicting tomorrow's stock return when large outliers are common.

### 2

Implement binary cross-entropy loss manually using `Bernoulli.log_prob()`.

Compare the result to `torch.nn.functional.binary_cross_entropy()`.

### 3

Generate samples from a two-component Gaussian mixture.

Compute the sample mean and variance.

Explain why the sample variance is larger than the variance of either component.

### 4

Show experimentally that minimizing mean squared error is equivalent to maximizing the log-likelihood of a Gaussian model.

Design an experiment where Laplace loss outperforms Gaussian loss in the presence of outliers.


## Summary

Probability distributions are model design choices.

PyTorch's `torch.distributions` module implements all distributions used in the *Deep Learning* book, together with methods for sampling, log-probability evaluation, and entropy computation.

The central insight is that choosing a loss function and choosing an output distribution are the same decision, expressed in different languages.


## Next Lesson

**Lesson 15 — Information Theory: Entropy, Cross-Entropy, and KL Divergence**

Chapter 3 concludes with information theory.

These concepts provide the mathematical foundation for cross-entropy loss, KL divergence regularization, and the broader connection between compression and learning.
