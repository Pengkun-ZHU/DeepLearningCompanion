# Lesson 05 — Norms, Special Matrices, and Decompositions

|Book|Chapter|Section|
|---|---|---|
| Deep Learning | 2 — Linear Algebra  | Sections 2.5–2.12 (Norms, Special Matrices, Eigendecomposition, SVD, PCA) |


This lesson covers the remaining sections of Chapter 2 and their PyTorch implementations.

## Objectives

By the end of this lesson you should be able to:

- compute vector and matrix norms in PyTorch;
- identify and create special matrix types (diagonal, symmetric, orthogonal);
- compute eigendecompositions and SVD in PyTorch;
- understand when and why these decompositions are used in deep learning;
- recognize how the Moore-Penrose pseudoinverse relates to SVD.


## Motivation

The *Deep Learning* book covers these topics to build the mathematical vocabulary needed for later chapters.

From an implementation perspective, these operations share an important characteristic:

They produce new matrices that are computationally derived from the input — there is no metadata shortcut.

Understanding their computational cost helps you reason about where they should and should not appear in a training loop. As you work through this chapter, it is worth asking yourself — and verifying experimentally — what each operation actually costs.


## Norms

A norm is a function that measures the size of a vector.

The most common norm in deep learning is the L2 norm:

$$
\|x\|_2 = \sqrt{\sum x_i^2}
$$

PyTorch provides `torch.linalg.norm()`.


## Investigation 1 — Vector Norms

```python
import torch

x = torch.tensor([3.0, 4.0])

# L2 norm
print(torch.linalg.norm(x))          # 5.0

# L1 norm
print(torch.linalg.norm(x, ord=1))   # 7.0

# Max norm (L-infinity)
print(torch.linalg.norm(x, ord=float('inf')))  # 4.0
```


## Investigation 2 — Matrix Norms

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

# Frobenius norm — square root of sum of squared entries
print(torch.linalg.norm(A, ord='fro'))

# Spectral norm — largest singular value
print(torch.linalg.norm(A, ord=2))
```

The Frobenius norm is often used to measure weight matrix magnitude in regularization.

The spectral norm is often used to measure and bound a model's sensitivity to input perturbations, helping to improve adversarial robustness and analyze generalization.

Computing the spectral norm does **not** perform a full SVD. Optimised linear algebra backends typically compute only the largest singular value, using iterative methods such as power iteration or Lanczos bidiagonalisation when appropriate. This is significantly cheaper than a complete decomposition.



## Special Matrices

### Diagonal matrices

```python
d = torch.tensor([1.0, 2.0, 3.0])

D = torch.diag(d)

print(D)
```


Multiplying diagonal matrix by a vector, i.e., 1-D tensor, can be very cheap if you use Hadamard product `d * x`, you may try it by hand to verify it gives the same result as `D @ x`.

```python
x = torch.randn(3)

# Effectively equivalent to D @ x — but largely different performance-wise
print(d * x)
```

The drastic performance difffernce comes from two sources: the Hadamard path avoids allocating the $N \times N$ matrix entirely, and it reduces the computation from $O(N^2)$ to $O(N)$. Whenever a diagonal matrix appears in a derivation, it is worth checking whether the full matrix can be replaced by its 1-D vector of diagonal entries.

### Symmetric matrices

```python
A = torch.tensor([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 5.0],
                  [3.0, 5.0, 6.0]])

print(torch.allclose(A, A.T))  # True
```

Symmetric matrices arise naturally when computing covariance matrices, Gram matrices, and Hessians.

### Orthogonal matrices

```python
Q, _ = torch.linalg.qr(torch.randn(4, 4))

print(torch.allclose(Q @ Q.T, torch.eye(4), atol=1e-5))   # True
print(torch.allclose(Q.T @ Q, torch.eye(4), atol=1e-5))   # True
```

> "In linear algebra, a QR decomposition, also known as a QR factorization or QU factorization, is a decomposition of a matrix A into a product A = QR of an orthonormal matrix Q and an upper triangular matrix R. QR decomposition is often used to solve the linear least squares (LLS) problem and is the basis for a particular eigenvalue algorithm, the QR algorithm." — Wikipedia

Orthogonal matrices preserve norms.

```python
x = torch.randn(4)
print(torch.linalg.norm(Q @ x))    # ≈ torch.linalg.norm(x)
```


## Investigation 3 — Eigendecomposition

A matrix `A` can be decomposed as:

$$
A = V \ \text{diag}(\lambda) \ V^{-1}
$$

where $\lambda$ are eigenvalues and the columns of $V$ are eigenvectors.

```python
A = torch.tensor([[4.0, 1.0],
                  [2.0, 3.0]])

eigenvalues, eigenvectors = torch.linalg.eig(A)

print("Eigenvalues :", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

Verify that for each eigenvalue–eigenvector pair:

```python
# A @ v = λ * v
v = eigenvectors[:, 0].real
lam = eigenvalues[0].real

print(torch.allclose(A @ v, lam * v, atol=1e-4))
```

Symmetric matrices have real eigenvalues and orthogonal eigenvectors.

```python
S = A + A.T   # make symmetric

eigenvalues, eigenvectors = torch.linalg.eigh(S)

print(torch.allclose(eigenvectors @ eigenvectors.T, torch.eye(2), atol=1e-5))
```


## Investigation 4 — Singular Value Decomposition

SVD generalizes eigendecomposition to non-square matrices:

$$
A = U S V^T
$$

where $U$ and $V$ are orthogonal and $S$ contains the singular values.

```python
A = torch.randn(3, 4)

U, S, Vh = torch.linalg.svd(A, full_matrices=False)

print("U :", U.shape)
print("S :", S.shape)
print("Vh:", Vh.shape)

# Reconstruct A
A_reconstructed = U @ torch.diag(S) @ Vh

print(torch.allclose(A, A_reconstructed, atol=1e-5))
```


## Low-Rank Approximation

SVD enables compact matrix approximations by retaining only the `k` largest singular values.

```python
k = 2

A_approx = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]

error = torch.linalg.norm(A - A_approx, ord='fro')

print(f"Frobenius error with rank-{k} approximation: {error:.4f}")
```

This principle underlies techniques such as:

- principal component analysis (PCA);
- low-rank weight factorization;
- dimensionality reduction.


## Moore-Penrose Pseudoinverse

For non-square or singular matrices, the standard inverse does not exist.

The pseudoinverse is the least-squares generalization:

```python
A = torch.randn(5, 3)

A_pinv = torch.linalg.pinv(A)

print(A_pinv.shape)   # (3, 5)

# A_pinv is computed using SVD internally
```

The relationship is:

$$
A^+ = V S^+ U^T
$$

where $S^+$ replaces each nonzero singular value with its reciprocal.


## Companion Insight

The operations in this lesson share a common property:

> They require genuine computation.

Unlike `transpose()` or `expand()`, which manipulate metadata, eigendecomposition and SVD perform substantial arithmetic.

SVD in particular scales as $O(\min(m,n) \cdot m \cdot n)$ for an $m \times n$ matrix.

This is why you should never compute SVD or eigendecomposition inside a tight training loop unless specifically required.

In practice, these operations appear in:

- model initialization;
- analysis and debugging;
- specialized layers such as spectral normalization.


## Source Reading

Open the PyTorch documentation for

```
torch.linalg
```

Locate the functions:

```
torch.linalg.eig
torch.linalg.svd
torch.linalg.norm
torch.linalg.pinv
```

Each of these dispatches to LAPACK (on CPU) or cuSOLVER (on CUDA).

This delegation pattern is not limited to matrix multiplication. Norms, decompositions, solves, and other structured linear algebra operations are all handed off to specialised third-party libraries — LAPACK, cuSOLVER, and MAGMA among them. PyTorch's role is to provide a unified tensor API; the heavy numerical lifting is done by battle-tested Fortran and CUDA code that has been optimised for decades.


## Exercises

### 1

Compute the L1, L2, and Frobenius norms of the matrix

```python
A = torch.tensor([[1.0, -2.0],
                  [3.0,  4.0]])
```

### 2

Verify that multiplying a vector by an orthogonal matrix preserves its L2 norm.

### 3

Compute the SVD of a 6×4 matrix.

Reconstruct the original matrix using all singular values.

Then reconstruct it using only the top 2 singular values.

Compare the reconstruction errors.

### 4

Explain in one paragraph why SVD is too expensive to compute in a training loop.

### 5

Look up spectral normalization for neural networks.

How does it use SVD, and how does it avoid computing a full SVD at every step?


## Summary

Chapter 2 of the *Deep Learning* book establishes the complete vocabulary of linear algebra used throughout the rest of the text.

The PyTorch `torch.linalg` module provides all of these operations.

The key engineering insight is that operations which require genuine computation — norms, decompositions, factorizations — have costs proportional to matrix size, and should not be placed carelessly inside training loops.


## Next Lesson

**Lesson 06 — Probability, Information Theory, and Numerical Stability**

Chapter 3 of the *Deep Learning* book introduces probability theory and information theory, and chapter 4 introduces numerical computation.

The two chapters are combined into a single lesson, connecting the mathematical definitions to PyTorch's random number and distribution infrastructure, and exploring what it means to model uncertainty computationally.
