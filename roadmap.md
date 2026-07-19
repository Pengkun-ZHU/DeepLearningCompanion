# Learning Roadmap

This roadmap tracks the planned lessons and long-term goals for **DeepLearningCompanion**.


## Lesson Progress

| # | Topic | Status |
|---|-------|--------|
| 01 | Tensor Internals — Storage, Strides, Views, Contiguous | ✅ Published |
| 02 | Indexing and Broadcasting | ✅ Published |
| 03 | Matrix Multiplication — Foundations, Performance, and Backends | ✅ Published |
| 04 | CUDA GEMM — From Naive to Tiled | ✅ Published |
| 05 | Norms, Special Matrices, and Decompositions | ✅ Published |
| 06 | Probability, Information Theory, and Numerical Stability | ✅ Published |
| 07 | Gradient-Based Optimisation Foundations | 🔲 Planned |
| 08 | Computational Graphs and Autograd | 🔲 Planned |
| 09 | Learning Algorithms and the ML Pipeline | 🔲 Planned |
| | ⋯ | |
| | **To Be Decided** | |
| | Additional topics covering regularisation, maximum likelihood estimation, Bayesian inference, advanced optimisers, and more — to be announced as the curriculum develops. | |

Lessons are designed to be completed sequentially, following the structure of the **Deep Learning** book.


## Planned Feature: Interactive Web Platform

### The Problem

Getting started with the lessons currently requires a non-trivial local setup:

- Installing a compatible Python environment and PyTorch version
- Installing the CUDA toolkit and a working GPU driver
- Dealing with version mismatches between PyTorch, CUDA, and OS
- Finding GPU access (local hardware, cloud instance, or Colab)

These friction points distract from what the lessons are actually about — **understanding how deep learning systems work**.

### The Plan

Build an **interactive web-based code editor** that accompanies the lessons, where users can:

- **Run Python and CUDA code directly in the browser** — no local GPU or installation required.
- **Follow along with each lesson** — code snippets from the lessons will be pre-loaded and executable with a single click.
- **Experiment freely** — modify kernel parameters, tensor shapes, and benchmark configurations to build intuition.
- **See real GPU output** — CUDA kernels will be compiled and executed on managed GPU infrastructure, with results streamed back to the browser.

The goal is to let learners focus entirely on the concepts and code, without ever fighting their environment.

### Scope (Tentative)

| Component | Description |
|-----------|-------------|
| Browser-based editor | Syntax-highlighted code editor (Python + CUDA C) embedded in each lesson page |
| Managed GPU backend | Remote execution environment with pre-installed PyTorch and CUDA toolchain |
| Lesson integration | Each lesson's code blocks become runnable, editable cells |
| Output panel | Real-time stdout, tensor prints, and performance metrics displayed inline |
| Sandboxing | Each user session gets an isolated environment to prevent interference |

### Status

This feature is in the **planning stage**. The immediate focus is on completing the lesson content. The web platform will be developed once the core curriculum is stable.

