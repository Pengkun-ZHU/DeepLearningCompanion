# Lesson 06 — Probability, Information Theory, and Numerical Stability


|Book|Chapter|Section|
|---|---|---|
| Deep Learning | 3 — Probability and Information Theory; 4 — Numerical Computation  | Sections 3.1–3.6, 3.9, 3.13–3.14; Sections 4.1–4.2 |

This lesson connects probability foundations, information theory, and numerical computation into a single narrative: the language deep learning uses to express learning, the loss functions that result, and the engineering tricks required to evaluate them reliably.


## Objectives

By the end of this lesson you should be able to:

- draw samples and evaluate log-probabilities using `torch.distributions`;
- explain why choosing a loss function is equivalent to choosing an output distribution;
- compute entropy, cross-entropy, and KL divergence from first principles;
- explain the equivalence between cross-entropy loss, maximum likelihood, and KL divergence minimisation;
- identify overflow, underflow, and NaN propagation in naive implementations;
- explain why `CrossEntropyLoss` takes raw logits and uses `log_softmax` internally.


## Motivation

Deep learning models are fundamentally probabilistic.

A classifier does not predict "this is a cat" — it predicts a probability distribution over classes. A regression model does not predict a single value — it implicitly assumes a Gaussian distribution around its prediction.

This means the objectives we optimise, the losses we compute, and the regularisers we add are all written in the language of probability and information theory.

And because these computations are performed in finite-precision arithmetic, the naive mathematical formula is often numerically dangerous.

This lesson traces the full chain.


## Probability Foundations in PyTorch

### Seeding and Reproducibility

```python
import torch

torch.manual_seed(42)
x = torch.rand(5)
print(x)

# Verify reproducibility
torch.manual_seed(42)
print(torch.rand(5))   # Identical
```

Seeding matters for reproducible experiments, debugging, and consistent weight initialisation.


### Discrete Random Variables

A discrete random variable takes values from a finite or countable set.

```python
probs = torch.tensor([0.5, 0.5])
dist = torch.distributions.Categorical(probs=probs)

samples = dist.sample((1000,))
print(f"P(class 0): {(samples == 0).float().mean():.3f}")
print(f"P(class 1): {(samples == 1).float().mean():.3f}")
```

As the sample count grows, empirical frequencies converge to the true probabilities — the **law of large numbers** in action.


### Continuous Random Variables

A continuous random variable takes values from an uncountable set.

PyTorch represents continuous distributions through probability density functions (PDFs), not probability mass functions (PMFs).

```python
dist = torch.distributions.Normal(loc=0.0, scale=1.0)
samples = dist.sample((10000,))

print(f"Sample mean: {samples.mean():.4f}")   # ≈ 0.0
print(f"Sample std : {samples.std():.4f}")    # ≈ 1.0
```

The probability of any single value is zero; only intervals have non-zero probability.


### Expectation and Variance

Expectation is the average value of a random variable. Variance measures how spread out it is.

```python
dist = torch.distributions.Uniform(0.0, 1.0)
samples = dist.sample((100000,))

print(f"Empirical mean : {samples.mean():.4f}")   # ≈ 0.5
print(f"Empirical var  : {samples.var():.4f}")    # ≈ 1/12 ≈ 0.083

# E[aX + b] = a·E[X] + b
a, b = 3.0, 5.0
print(f"E[3X + 5] ≈ {(a * samples + b).mean():.4f}")
print(f"Exact       : {a * 0.5 + b:.4f}")
```


## Common Distributions as Model Design Choices

The *Deep Learning* book surveys the distributions most commonly used in machine learning. The key insight is that **choosing a loss function and choosing an output distribution are the same decision**, expressed in different languages.

| Task | Distribution | Loss |
|---|---|---|
| Binary classification | Bernoulli | Binary cross-entropy |
| Multi-class classification | Categorical | Cross-entropy |
| Regression | Gaussian | Mean squared error |
| Robust regression | Laplace | Mean absolute error |


### Bernoulli — Binary Classification

```python
dist = torch.distributions.Bernoulli(probs=0.7)
samples = dist.sample((10000,))
print(f"Empirical P(y=1): {samples.mean():.4f}")

# In practice, the probability comes from a sigmoid output
logit = torch.tensor(0.847)
prob = torch.sigmoid(logit)
dist = torch.distributions.Bernoulli(probs=prob)
print(f"P(y=1) = {prob:.4f}")
```


### Categorical — Multi-class Classification

```python
logits = torch.tensor([1.0, 3.0, 2.0])   # raw network output

dist = torch.distributions.Categorical(logits=logits)
# Internally applies softmax
print(dist.probs)
```

Neural networks produce unnormalised scores called **logits**. Passing logits directly lets PyTorch handle the normalisation in a numerically stable way.


### Gaussian — Regression and MSE

```python
mu    = torch.tensor(3.0)
sigma = torch.tensor(1.5)

dist = torch.distributions.Normal(mu, sigma)
x = torch.tensor(3.5)
print(f"log p(x=3.5) = {dist.log_prob(x):.4f}")
```

Using a Gaussian output is equivalent to minimising mean squared error. Maximising $\log p(y \mid x)$ for a Gaussian reduces to minimising $(y - \mu)^2 / 2\sigma^2$.


### Laplace — Robust Regression

```python
laplace = torch.distributions.Laplace(loc=0.0, scale=1.0)
normal  = torch.distributions.Normal(loc=0.0, scale=1.0)

x = torch.tensor([0.0, 1.0, 2.0, 3.0])
print("x     | Laplace log p | Normal log p")
for xi in x:
    print(f"{xi.item():.1f}   | {laplace.log_prob(xi).item():12.4f}  | {normal.log_prob(xi).item():12.4f}")
```

The Laplace distribution assigns higher probability to values far from the mean, making it more robust to outliers. This is equivalent to minimising mean absolute error.


### Working in Log Space

Every distribution in `torch.distributions` exposes `log_prob()` as the primary interface, not `prob()`.

```python
# Naive: multiplying small numbers underflows
prob = 0.01 * 0.02 * 0.005   # → 1e-6, fine here but catastrophic at scale

# Correct: sum log-probabilities
log_prob = torch.log(torch.tensor(0.01)) + torch.log(torch.tensor(0.02)) + torch.log(torch.tensor(0.005))
```

This avoids multiplying many small numbers, which quickly underflows to zero in floating-point arithmetic. This principle will become critical in the next section.


## Information Theory: Entropy, Cross-Entropy, and KL Divergence

### Entropy

Shannon entropy measures the expected amount of surprise in a probability distribution.

$$
H(P) = -\sum P(x) \log P(x)
$$

A uniform distribution has high entropy — outcomes are maximally uncertain. A peaked distribution has low entropy — outcomes are predictable.

```python
def entropy(probs):
    log_probs = torch.where(probs > 0,
                            torch.log(probs),
                            torch.zeros_like(probs))
    return -(probs * log_probs).sum()

uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
print(f"Entropy (uniform): {entropy(uniform):.4f}")   # log(4) ≈ 1.386

certain = torch.tensor([1.0, 0.0, 0.0, 0.0])
print(f"Entropy (certain): {entropy(certain):.4f}")   # 0.0
```

`torch.where` acts as a safeguard against a fatal mathematical edge case $\log(0) = -\infty$: it intercepts zeros and substitutes them before the log is computed.

But the pattern is not safe under autograd — the backward pass can produce `NaN` gradients even when the forward output looks correct. This hazard is discussed in detail in the Numerical Stability section.

`torch.distributions` provides entropy directly:

```python
dist = torch.distributions.Categorical(probs=torch.tensor([0.1, 0.4, 0.5]))
print(f"Entropy: {dist.entropy():.4f}")
```


### Cross-Entropy

Cross-entropy measures the expected cost of encoding outcomes from distribution $P$ using a code optimised for distribution $Q$:

$$
H(P, Q) = -\sum P(x) \log Q(x)
$$

When $Q$ is the model's prediction and $P$ is the true distribution (usually one-hot), this becomes the standard classification loss.

```python
y_true = torch.tensor([0.0, 0.0, 1.0])            # true label: class 2
y_pred_good = torch.tensor([0.05, 0.05, 0.90])   # confident and correct
y_pred_bad  = torch.tensor([0.33, 0.33, 0.34])   # uncertain

def cross_entropy(p, q):
    return -(p * torch.log(q)).sum()

print(f"Good prediction CE: {cross_entropy(y_true, y_pred_good):.4f}")   # low
print(f"Bad  prediction CE: {cross_entropy(y_true, y_pred_bad):.4f}")    # high
```

Because the true label is one-hot, cross-entropy simplifies to $-\log Q(\text{correct class})$ — exactly what `torch.nn.CrossEntropyLoss` computes.


### Cross-Entropy in PyTorch

```python
logits = torch.tensor([[2.0, 1.0, 4.0]])   # shape (batch, classes)
labels = torch.tensor([2])                  # correct class index

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)

# Manual verification
probs = torch.softmax(logits, dim=1)
manual_ce = -torch.log(probs[0, 2])
print(f"CrossEntropyLoss: {loss:.4f}")
print(f"Manual CE       : {manual_ce:.4f}")
```

`CrossEntropyLoss` applies `log_softmax` internally for numerical stability. **Never** apply `softmax` yourself before passing logits to it.


### KL Divergence

KL divergence measures how different distribution $Q$ is from distribution $P$:

$$
\text{KL}(P \| Q) = \sum P(x) \log \frac{P(x)}{Q(x)}
$$

It is always non-negative, zero if and only if $P = Q$, and not symmetric.

```python
def kl_divergence(p, q):
    return (p * (torch.log(p) - torch.log(q))).sum()

p = torch.tensor([0.4, 0.4, 0.2])
q = torch.tensor([0.3, 0.3, 0.4])

print(f"KL(P||Q) = {kl_divergence(p, q):.4f}")
print(f"KL(Q||P) = {kl_divergence(q, p):.4f}")   # different!
print(f"KL(P||P) = {kl_divergence(p, p):.4f}")   # 0.0
```

The relationship to cross-entropy is:

$$
H(P, Q) = H(P) + \text{KL}(P \| Q)
$$


### The Equivalence Chain

Minimising cross-entropy loss is equivalent to:

1. **Maximising the log-likelihood** of the training data.
2. **Minimising the KL divergence** from the model distribution to the empirical data distribution.

This is why cross-entropy is not a convenient heuristic — it is the principled loss function for probabilistic classification, derived directly from information theory.

In variational autoencoders and other latent variable models, KL divergence also appears explicitly as a regulariser that keeps the learned latent distribution close to a prior.


## Numerical Stability

The mathematical formulas above are clean. Their floating-point implementations are not.

### Overflow and Underflow

```python
x = torch.tensor(89.0)
print(f"exp(89)  = {x.exp():.4e}")    # large but representable

x = torch.tensor(90.0)
print(f"exp(90)  = {x.exp():.4e}")    # overflows to inf in float32

x = torch.tensor(-90.0)
print(f"exp(-90) = {x.exp():.4e}")    # underflows toward 0
```

These limits matter because softmax computes `exp()` on raw logits.


### Naive vs Stable Softmax

```python
def naive_softmax(x):
    return x.exp() / x.exp().sum()

logits = torch.tensor([1000.0, 1001.0, 1002.0])
print(naive_softmax(logits))   # all inf — overflow
```

The fix is to subtract the maximum before exponentiating:

```python
def stable_softmax(x):
    shifted = x - x.max()
    return shifted.exp() / shifted.exp().sum()

print(stable_softmax(logits))  # correct
```

Subtracting a constant does not change the softmax value mathematically — it cancels in numerator and denominator — but it prevents the intermediate exponentials from overflowing.


### Log-Sum-Exp

The pattern above is so common that PyTorch provides it as a primitive:

```python
logits = torch.tensor([1000.0, 1001.0, 1002.0])

# log(sum(exp(x))) — numerically stable
lse = torch.logsumexp(logits, dim=0)
print(f"logsumexp: {lse:.4f}")

# log-softmax = x - logsumexp(x)
log_sm = logits - torch.logsumexp(logits, dim=0)
print(log_sm)
print(torch.log_softmax(logits, dim=0))   # same result
```

`torch.logsumexp` is used internally in `CrossEntropyLoss`, which computes:

$$
\text{loss} = -\text{logits}[y] + \text{logsumexp}(\text{logits})
$$

rather than the naive $-\log(\text{softmax}(\text{logits})[y])$.


### NaN Propagation

NaN propagates silently through arithmetic:

```python
x = torch.tensor(0.0)
y = x / x              # NaN
print(y + 1.0)         # tensor(nan) — NaN is contagious
```

A loss that becomes `NaN` at step 1000 may have been accumulating errors since step 1. Detect early:

```python
def check_finite(t, name="tensor"):
    if not torch.isfinite(t).all():
        print(f"WARNING: {name} contains NaN or Inf")

check_finite(torch.tensor([1.0, float('nan')]), "x")
```

PyTorch also provides `torch.autograd.set_detect_anomaly(True)` for detailed NaN gradient diagnostics.


### `torch.where` Is Not Autograd-Safe

The `torch.where` guard used in the Entropy section — `torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs))` — looks safe because the forward pass correctly discards `-inf`. However, the backward pass can still produce `NaN` gradients. (Full details of forward and backward propagation are covered in later lessons; this is a brief heads-up.)

#### Why the Backward Pass Breaks

PyTorch's autograd engine records every operation into a **computational graph** during the forward pass so it can apply the chain rule in reverse:

```
        probs = 0.0
           /       \
     [torch.log]   [zeros_like]
          |             |
        -inf           0.0
           \           /
        [torch.where]   ← selects 0.0 for forward output
              |
            output
```

Notice that `torch.log` is still a node in that graph, even though `torch.where` discarded its output. When `.backward()` is called, autograd traverses the graph in reverse and differentiates every executed operation — including `torch.log`.

The derivative of $\log(x)$ is $\frac{1}{x}$. At $x = 0$:

$$
\text{grad\_x} = \text{grad\_output} \times \frac{1}{0} = \text{NaN}
$$

Because `probs` feeds into *both* the condition and `torch.log`, its total gradient is the sum of gradients from all paths:

$$
\text{grad\_probs} = \underbrace{0.0}_{\text{from where (discarded)}} + \underbrace{\text{NaN}}_{\text{from log}} = \text{NaN}
$$

In IEEE floating-point, anything plus `NaN` is `NaN`. The poisoned gradient silently corrupts `probs.grad` and propagates upstream.

#### Proof

```python
probs = torch.tensor([0.5, 0.0], requires_grad=True)

# Forward pass
log_probs = torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs))
loss = log_probs.sum()

print("Forward output:", log_probs.detach())
# tensor([-0.6931,  0.0000])  — looks clean

# Backward pass
loss.backward()
print("Gradients:", probs.grad)
# tensor([2.0000, nan])  — NaN leaked!
```

#### The Fix: Clamp Before Log

Prevent `probs` from ever reaching absolute zero inside `log`:

```python
# Safe both forward and backward
log_probs = torch.log(torch.clamp(probs, min=1e-12))
```

With `clamp`, `torch.log` never sees a true zero. The forward value is a large-but-finite negative number, and the gradient is a large-but-finite value — no `NaN`, no silent corruption.


## Companion Insight

There is a single principle connecting everything in this lesson:

> **Avoid computing very large or very small intermediate values.**

This is why:

- distributions expose `log_prob()` rather than `prob()` — stay in log space;
- `CrossEntropyLoss` uses `log_softmax` internally rather than `softmax` followed by `log`;
- stable softmax subtracts the maximum before exponentiating;
- `torch.logsumexp` exists as a separate primitive;
- `torch.clamp` prevents `log(0)` from producing `-inf` in the forward pass and `NaN` in the backward pass.

Each of these is the same numerical stability technique, applied at a different layer of the stack.


## Source Reading

Open:

```
torch/distributions/
```

Each distribution inherits from `Distribution` and provides `sample()`, `log_prob()`, and `entropy()`. Open `normal.py` and verify that `log_prob()` is computed directly as a formula, not by computing the probability and then taking the log.

Then open:

```
torch/nn/functional.py
```

Find `cross_entropy`. Trace how it delegates to `nll_loss` with `log_softmax`. Compare the stable formula:

```
-logits[y] + logsumexp(logits)
```

to the naive:

```
-log(softmax(logits)[y])
```

Verify mathematically that they are identical.


## Exercises

### 1

Draw 10,000 samples from `Normal(mean=5.0, std=2.0)`. Verify that the sample mean and standard deviation match the distribution parameters. How many samples are needed before they agree within 1%?

### 2

For each scenario, choose the appropriate output distribution and explain why:

- Predicting house prices.
- Predicting whether an email is spam.
- Predicting which digit (0–9) an image contains.
- Predicting tomorrow's stock return when large outliers are common.

### 3

Verify numerically that $H(P, Q) = H(P) + \text{KL}(P \| Q)$ for two arbitrary distributions.

### 4

Show experimentally that minimising mean squared error is equivalent to maximising the log-likelihood of a Gaussian model. Design an experiment where Laplace loss outperforms Gaussian loss in the presence of outliers.

### 5

Verify that `softmax(x - max(x))` is mathematically identical to `softmax(x)`. Prove it algebraically and verify numerically with logits `[1000, 1001, 1002]`.

### 6

Explain why `CrossEntropyLoss` does not accept probabilities as input. What would go wrong numerically if you passed softmax outputs to it?


## Summary

Probability, information theory, and numerical computation form a single narrative:

- **Probability** provides the language — distributions, log-probabilities, sampling.
- **Distribution choice** determines the loss function — MSE implies Gaussian, cross-entropy implies Categorical.
- **Information theory** justifies the loss — cross-entropy equals maximum likelihood equals KL divergence minimisation.
- **Numerical stability** makes the loss computable — log space, stable softmax, logsumexp, and safe gradient patterns like clamping.

The central engineering insight is that all numerical stability techniques in deep learning are manifestations of the same principle: stay in log space and avoid extreme intermediate values.


## Next Lesson

**Lesson 07 — Gradient-Based Optimisation Foundations**

Chapter 4 continues with optimisation. We'll study the mathematical foundation of gradient descent, connect it to PyTorch's optimiser API, and investigate the relationship between the loss surface and optimisation dynamics.
