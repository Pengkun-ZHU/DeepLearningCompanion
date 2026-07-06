# Lesson 16 — Numerical Precision and Stability

| | |
|---|---|
| **Chapter** | 4 — Numerical Computation |
| **Reading Assignment** | Sections 4.1–4.2 (Overflow and Underflow, Poor Conditioning) |

This lesson investigates how floating-point arithmetic behaves in PyTorch and how framework implementations are designed to remain numerically stable.

## Objectives

By the end of this lesson you should be able to:

- explain overflow and underflow in floating-point arithmetic;
- identify common sources of numerical instability in deep learning;
- understand why `log_softmax` exists as a separate operation;
- use `torch.finfo` to inspect floating-point precision limits;
- recognize numerically stable implementations and explain why they differ from the naive formula.


## Motivation

The *Deep Learning* book opens Chapter 4 with a simple observation:

> Digital computers can represent only a finite subset of real numbers.

This limitation is not just theoretical.

Numerical issues cause:

- training divergence when losses become `nan` or `inf`;
- incorrect gradients from numerically unstable operations;
- silently wrong results that are difficult to debug.

Understanding numerical precision is part of engineering robust deep learning systems.


## Floating-Point Basics

PyTorch uses IEEE 754 floating-point arithmetic.

```python
import torch

# Inspect float32 limits
info = torch.finfo(torch.float32)
print(f"float32 max    : {info.max}")
print(f"float32 min    : {info.min}")
print(f"float32 eps    : {info.eps}")
print(f"float32 tiny   : {info.tiny}")
```

`eps` is the smallest number such that `1.0 + eps ≠ 1.0`.

`tiny` is the smallest positive normalized number.


## Investigation 1 — Overflow and Underflow

```python
# Overflow: a value too large to represent
x = torch.tensor(89.0)
print(f"exp(89)  = {x.exp():.4e}")    # large but representable

x = torch.tensor(90.0)
print(f"exp(90)  = {x.exp():.4e}")    # overflows to inf for float32

# Underflow: a value too small to represent
x = torch.tensor(-90.0)
print(f"exp(-90) = {x.exp():.4e}")    # underflows toward 0
```

```python
# Softmax naively
def naive_softmax(x):
    return x.exp() / x.exp().sum()

logits = torch.tensor([1000.0, 1001.0, 1002.0])
print(naive_softmax(logits))   # all inf — overflow
```

The numerically stable version subtracts the maximum:

```python
def stable_softmax(x):
    c = x.max()
    shifted = x - c
    return shifted.exp() / shifted.exp().sum()

print(stable_softmax(logits))  # correct
```

Subtracting `c` does not change the softmax value mathematically — it cancels in numerator and denominator — but it prevents overflow.


## Investigation 2 — Log-Sum-Exp

The pattern above is so common that PyTorch provides it as a primitive.

```python
logits = torch.tensor([1000.0, 1001.0, 1002.0])

# Numerically stable log(sum(exp(x)))
log_sum_exp = torch.logsumexp(logits, dim=0)

print(f"logsumexp    : {log_sum_exp:.4f}")
print(f"manual check : {stable_softmax(logits).log() + log_sum_exp - logits}")
```

`torch.logsumexp()` is used internally in many PyTorch operations.

Log-softmax is:

```python
log_softmax = logits - torch.logsumexp(logits, dim=0)

print(log_softmax)
print(torch.log_softmax(logits, dim=0))   # same result
```


## Investigation 3 — NaN Propagation

NaN (Not a Number) propagates silently through arithmetic.

```python
x = torch.tensor(0.0)

y = x / x              # NaN
print(y)               # tensor(nan)
print(y + 1.0)         # tensor(nan) — NaN is contagious
print(y > 0)           # tensor(False) — comparisons fail silently
```

This is why a loss that becomes NaN at step 1000 may have been accumulating numerical errors since step 1.

```python
# Detect NaN/Inf in a tensor
def check_finite(t, name="tensor"):
    if not torch.isfinite(t).all():
        print(f"WARNING: {name} contains NaN or Inf")
    else:
        print(f"{name} is finite")

check_finite(torch.tensor([1.0, float('nan')]), "x")
check_finite(torch.tensor([1.0, 2.0, 3.0]), "y")
```

PyTorch also provides:

```python
torch.autograd.set_detect_anomaly(True)
```

This enables detailed error messages when NaN gradients are produced.


## Investigation 4 — dtype and Precision

PyTorch supports multiple floating-point types.

```python
x32 = torch.tensor(1.0, dtype=torch.float32)
x64 = torch.tensor(1.0, dtype=torch.float64)
x16 = torch.tensor(1.0, dtype=torch.float16)

print(torch.finfo(torch.float32).eps)   # ~1.2e-7
print(torch.finfo(torch.float64).eps)   # ~2.2e-16
print(torch.finfo(torch.float16).eps)   # ~9.8e-4
```

float16 is commonly used for training speed but has very limited range.

```python
x = torch.tensor(65504.0, dtype=torch.float16)
print(x)               # maximum float16 value
print(x + x)           # overflow to inf
```

Modern training often uses **mixed precision**: forward pass in float16, gradient accumulation in float32.


## Companion Insight

Numerically stable implementations and naive implementations compute mathematically identical functions.

The difference is entirely in **how the computation is ordered**.

The key pattern is:

> Avoid computing very large or very small intermediate values.

This is why:

- `CrossEntropyLoss` uses `log_softmax` internally rather than `softmax` followed by `log`;
- gradient clipping prevents parameter updates from becoming too large;
- batch normalization keeps activations in a reasonable range.

Each of these is a numerical stability technique dressed in different terminology.


## Source Reading

Open:

```
torch/nn/functional.py
```

Find `cross_entropy`.

Trace how it delegates to `nll_loss` with `log_softmax`.

Compare the stable implementation to the naive formula:

```
-log(softmax(logits)[y])
```

versus the combined:

```
-logits[y] + logsumexp(logits)
```

Verify mathematically that they are identical.


## Exercises

### 1

Verify that `softmax(x - max(x))` is mathematically identical to `softmax(x)`.

Prove it algebraically and verify numerically.

### 2

Show that computing softmax of `[1000, 1001, 1002]` overflows with `float32` but not `float64`.

What does this suggest about using `float64` for debugging?

### 3

Find a computation in PyTorch that silently produces NaN.

Write a check that would detect it.

### 4

Explain why `CrossEntropyLoss` does not accept probabilities as input.

What would go wrong numerically if you passed softmax outputs to it?

### 5

When would you use `float64` instead of `float32`?

What is the cost?


## Summary

Floating-point arithmetic imposes practical constraints on deep learning implementations.

The framework's responsibility is to perform mathematically correct computations in a numerically stable way.

Key principles:

- subtract the maximum before computing softmax or log-sum-exp;
- work in log space whenever possible;
- use `torch.logsumexp` rather than computing `log(sum(exp()))` directly;
- detect NaN and Inf early rather than letting them propagate.


## Next Lesson

**Lesson 17 — Gradient-Based Optimization Foundations**

Chapter 4 continues with optimization.

We'll study the mathematical foundation of gradient descent, connect it to PyTorch's optimizer API, and investigate the relationship between the loss surface and optimization dynamics.
