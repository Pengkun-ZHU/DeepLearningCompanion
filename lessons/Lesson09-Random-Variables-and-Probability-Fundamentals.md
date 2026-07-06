# Lesson 09 — Random Variables and Probability Fundamentals

| | |
|---|---|
| **Chapter** | 3 — Probability and Information Theory |
| **Reading Assignment** | Sections 3.1–3.6 (Random Variables, Probability Distributions, Marginal and Conditional Probability, Independence, Expectation, Variance, Covariance) |

This lesson connects the probability foundations of Chapter 3 to PyTorch's random number generation and sampling infrastructure.

## Objectives

By the end of this lesson you should be able to:

- distinguish discrete and continuous random variables in PyTorch;
- draw samples from distributions using `torch.distributions`;
- compute expectation, variance, and covariance experimentally;
- explain the difference between a probability mass function and a probability density function;
- understand how PyTorch's random number generator works at a high level.


## Motivation

The *Deep Learning* book introduces probability theory because deep learning models are fundamentally probabilistic.

A neural network trained with maximum likelihood implicitly defines a probability distribution over outputs.

Understanding probability is not optional background material.

It is the language in which deep learning objectives are written.

From an implementation perspective, probability connects to:

- weight initialization;
- data augmentation;
- stochastic gradient descent;
- generative models;
- Bayesian approaches to regularization.

Today's lesson focuses on the foundations.


## Random Number Generation in PyTorch

PyTorch generates random numbers using a pseudorandom number generator (PRNG).

```python
import torch

# Set seed for reproducibility
torch.manual_seed(42)

x = torch.rand(5)

print(x)
```

Setting the seed ensures that the same sequence of numbers is produced.

This matters for:

- reproducible experiments;
- debugging;
- consistent weight initialization.

```python
# Verify reproducibility
torch.manual_seed(42)
print(torch.rand(5))   # Identical to the first run
```


## Investigation 1 — Discrete Random Variables

A discrete random variable takes values from a finite or countable set.

The simplest example is a fair coin:

```python
# P(heads) = 0.5, P(tails) = 0.5
probs = torch.tensor([0.5, 0.5])

dist = torch.distributions.Categorical(probs=probs)

samples = dist.sample((1000,))

heads = (samples == 0).float().mean()
tails = (samples == 1).float().mean()

print(f"Heads: {heads:.3f}, Tails: {tails:.3f}")
```

As the number of samples increases, the empirical frequencies converge to the true probabilities.

This is the **law of large numbers** in action.


## Investigation 2 — Continuous Random Variables

A continuous random variable takes values from an uncountable set such as the real line.

PyTorch represents continuous distributions through probability density functions (PDFs), not probability mass functions (PMFs).

```python
dist = torch.distributions.Normal(loc=0.0, scale=1.0)

samples = dist.sample((10000,))

print(f"Sample mean : {samples.mean():.4f}")
print(f"Sample std  : {samples.std():.4f}")
```

The probability of any single value is zero.

We can only ask about probabilities of intervals.


## Expectation

The expectation of a random variable is its average value.

```python
# True expectation of Uniform(0, 1) is 0.5
dist = torch.distributions.Uniform(0.0, 1.0)
samples = dist.sample((100000,))

print(f"Empirical expectation: {samples.mean():.4f}")
print(f"True expectation     : 0.5000")
```

Expectation is linear:

```python
# E[a*X + b] = a * E[X] + b
a, b = 3.0, 5.0
print(f"E[3X + 5] ≈ {(a * samples + b).mean():.4f}")
print(f"True      : {a * 0.5 + b:.4f}")
```


## Variance and Standard Deviation

Variance measures how spread out a random variable is.

```python
dist = torch.distributions.Normal(loc=2.0, scale=3.0)
samples = dist.sample((100000,))

print(f"Empirical mean : {samples.mean():.4f}")  # ≈ 2.0
print(f"Empirical std  : {samples.std():.4f}")   # ≈ 3.0
print(f"Empirical var  : {samples.var():.4f}")   # ≈ 9.0
```


## Investigation 3 — Covariance

Covariance measures how two random variables change together.

```python
torch.manual_seed(0)

n = 10000

x = torch.randn(n)
y = 2 * x + torch.randn(n) * 0.5    # y correlates with x

cov = torch.cov(torch.stack([x, y]))

print("Covariance matrix:")
print(cov)
```

Positive covariance means the variables tend to increase together.

Negative covariance means they tend to move in opposite directions.

Zero covariance means they are uncorrelated (but not necessarily independent).


## The Gaussian Distribution

The Normal (Gaussian) distribution is the most important distribution in deep learning.

It arises naturally through the central limit theorem and has convenient mathematical properties.

```python
import matplotlib.pyplot as plt

dist = torch.distributions.Normal(0.0, 1.0)

x = torch.linspace(-4, 4, 200)

# Probability density
log_prob = dist.log_prob(x)
prob = log_prob.exp()

print("Peak density at x=0:", prob[100].item())
```

Deep learning frameworks work with **log probabilities** rather than raw probabilities to avoid numerical underflow.

This is why `log_prob()` is the primary interface.


## Companion Insight

One of the most important design decisions in PyTorch's probability infrastructure is working in **log space**.

Instead of:

```python
prob = p1 * p2 * p3
```

Deep learning code computes:

```python
log_prob = log_p1 + log_p2 + log_p3
```

This avoids multiplying together many small numbers, which would quickly underflow to zero in floating-point arithmetic.

We'll revisit this principle in Lesson 16 when we study numerical stability.


## Source Reading

Open:

```
torch/distributions/
```

Browse the directory structure.

Each distribution is a Python class inheriting from `torch.distributions.Distribution`.

The base class provides:

- `sample()`
- `log_prob()`
- `entropy()`

Look at `torch/distributions/normal.py`.

Observe that `log_prob()` is implemented directly as a formula rather than computing the probability first and then taking the log.

Question: why?


## Exercises

### 1

Draw 10,000 samples from a `Normal(mean=5.0, std=2.0)` distribution.

Verify that the sample mean and standard deviation match the distribution parameters.

How many samples are needed before they agree within 1%?

### 2

Compute the empirical covariance of

```python
x = torch.randn(1000)
y = -x + torch.randn(1000) * 0.1
```

Interpret the sign of the covariance.

### 3

Show that for a `Uniform(0, 1)` distribution:

- the empirical mean converges to 0.5;
- the empirical variance converges to 1/12 ≈ 0.083.

### 4

Explain the difference between a probability mass function and a probability density function.

Give a PyTorch example of each.

### 5

Explain in one paragraph why deep learning code uses `log_prob()` instead of raw probabilities.


## Summary

Probability theory provides the language of uncertainty that deep learning models express and optimize.

PyTorch implements the mathematical concepts of Chapter 3 through:

- `torch.manual_seed()` for reproducible randomness;
- `torch.distributions` for sampling and density evaluation;
- `torch.cov()` for covariance estimation.

The key engineering insight is that computations are performed in log space to preserve numerical precision.


## Next Lesson

**Lesson 10 — Common Probability Distributions in PyTorch**

The *Deep Learning* book surveys the distributions most commonly used in machine learning.

We'll study each one from the implementation perspective, focusing on how it is used in model design, loss functions, and output layer choices.
