# DeepLearningCompanion

> **A companion repository for *Deep Learning* (Goodfellow, Bengio & Courville), connecting mathematical theory with PyTorch implementation, CUDA optimization, and framework internals.**



The book **Deep Learning** is one of the best references for understanding the theory behind modern machine learning.

However, like most textbooks, it deliberately focuses on **what** the algorithms are and **why** they work. It does not attempt to explain in detail:

* How PyTorch implements these ideas.
* How tensors are represented internally.
* How automatic differentiation works in practice.
* Why certain operations are fast while others are slow.
* How CUDA accelerates neural network workloads.
* How modern deep learning frameworks are engineered.

This repository is designed to answer those questions.

Rather than replacing the book, **DeepLearningCompanion** is intended to be read **alongside** it.



# Learning Philosophy

Every lesson begins with a reading assignment from the **Deep Learning** book.

After reading the assigned sections, we'll explore the same concepts from four additional perspectives:

```text
               Deep Learning (Book)
                        │
                        ▼
              Mathematics & Intuition
                        │
                        ▼
               PyTorch Programming
                        │
                        ▼
            PyTorch Source Code Reading
                        │
                        ▼
          CUDA & Performance (when applicable)
```

The objective is not simply to learn deep learning.

The objective is to understand **how modern deep learning systems are built.**


# Target Audience

This repository assumes the reader:

* is already comfortable with programming;
* has extended C++ and Python knowledge;
* is willing to read large code bases;
* wants to understand implementation details instead of only using APIs.

This is **not** an introductory programming course.


# Repository Structure

```text
DeepLearningCompanion/

README.md
roadmap.md

lessons/
    Companion lessons following the Deep Learning book.

minitorch/
    A minimal deep learning framework implemented incrementally.

```


# Lesson Structure

Every lesson follows the same format.

```text
Reading Assignment
        │
        ▼
Learning Objectives
        │
        ▼
Companion Notes
        │
        ▼
PyTorch Investigation
        │
        ▼
Source Reading
        │
        ▼
CUDA Insight (when applicable)
        │
        ▼
Programming Experiments
        │
        ▼
Exercises
        │
        ▼
What's Next
```

The **Deep Learning** book remains the authoritative source for theory.

This repository focuses on implementation, engineering, and practical investigation.


# Guiding Principles

### 1. The Book Comes First

Every lesson explicitly references the corresponding chapter and section(s) of **Deep Learning**.

Reading the assigned pages is considered a prerequisite.


### 2. Observe Before Explaining

Whenever possible, we start with an experiment.

Instead of immediately explaining a concept, we first observe how PyTorch behaves, form hypotheses, and then connect those observations to the mathematical theory and implementation.


### 3. Read Real Source Code

Many excellent tutorials stop at the API.

This repository regularly points to the relevant files in the PyTorch source tree and explains why they matter.

The goal is not to understand every line immediately, but to gradually become comfortable navigating a large production code base.


### 4. Learn Performance as Part of the Concept

CUDA is introduced only when it helps explain the implementation or performance characteristics of a topic.

Rather than treating CUDA as a separate subject, we study it exactly where it becomes relevant.


### 5. Build Understanding Incrementally

Concepts introduced in one lesson reappear throughout the repository from different perspectives:

* mathematics;
* PyTorch usage;
* source code;
* CUDA;
* experiments.

Each revisit adds another layer of understanding.


# Experiments

Most lessons include one or more small investigations.

These are intentionally designed to answer questions such as:

* Why is `transpose()` O(1)?
* Why can two tensors share the same storage?
* Why does `view()` sometimes fail?
* What makes a tensor contiguous?
* Why are matrix multiplications so fast?
* When does CUDA actually provide a speedup?

The experiments are intended to encourage exploration rather than simply verify expected results.


# Long-Term Goals

By the end of this repository, you should be able to:

* understand the mathematical foundations of deep learning;
* write idiomatic PyTorch programs;
* navigate relevant parts of the PyTorch source tree;
* explain how tensors are represented internally;
* understand the principles behind automatic differentiation;
* reason about tensor memory layout and performance;
* understand where CUDA fits into modern deep learning systems;
* read implementation code with confidence.


# Planned: Interactive Web Platform

A major planned feature is an **interactive web-based code editor** that will let you run the Python and CUDA code from every lesson directly in your browser — no local GPU, no PyTorch installation, no CUDA toolkit, no version mismatches.

The goal is to remove all environment friction so you can focus entirely on learning.

See **roadmap.md** for the full plan and status of this feature.


# Progress

See **roadmap.md** for the complete learning roadmap.

Lessons are designed to be completed sequentially, following the order of the **Deep Learning** book.


# Contributions Welcome

This repository is the result of my journey from an experienced software engineer into the world of deep learning systems. I am still actively learning these subjects, and these notes are part of that process—not the final word on them.

That means mistakes are possible: in the explanations, the code, the CUDA details, or the connections drawn between theory and implementation. If you notice something incorrect, misleading, or that could be explained more clearly, I would genuinely appreciate your feedback.

Corrections, suggestions, pull requests, and even a simple issue pointing out a mistake are all warmly welcome. Improving the repository together is one of the best contributions you can make.
