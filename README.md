# Bayesian Black-Box Optimisation — Capstone Project

**Imperial College London · Executive Master in Machine Learning \& Artificial Intelligence**  
**Gian Franco Cattaneo**

\---

## Non-Technical Summary

This project applies Bayesian optimisation to eight unknown mathematical functions, each accepting between two and eight numerical inputs and returning a single score. The goal is to find the input combination that produces the highest score, using as few experiments as possible — much like tuning a complex industrial process without being able to see inside it. A Gaussian Process surrogate model learns from each result to predict where the next best experiment should be run. The project demonstrates how data-efficient optimisation can replace exhaustive trial-and-error in real-world engineering and business contexts.

\---

## Project Structure

```
bbo-capstone/
├── README.md                        ← this file
├── DATASHEET.md                     ← dataset description, provenance, limitations
├── MODEL\_CARD.md                    ← model behaviour, assumptions, interpretability
├── SUBMISSIONS\_LOG.md               ← portal submission strings, round-by-round record
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── notebooks/
│   ├── BBO\_Round1\_W12.ipynb         ← Submission 1: initial GP-EI, d=2–8D
│   ├── BBO\_Round2\_W13.ipynb         ← Submission 2: updated surrogate, boundary analysis
│   ├── BBO\_Round3\_W14.ipynb         ← Submission 3: SVM classification framing
│   ├── BBO\_Round4\_W15.ipynb         ← Submission 4: NN surrogate + gradient analysis
│   └── BBO\_Round5\_W16.ipynb         ← Submission 5: corrected maximisation EI, R1–R4 data
│
├── data/
│   └── DATA\_SOURCES.md              ← external data description (no large files on GitHub)
│
├── plots/
│   ├── convergence/                 ← best Y per round per function
│   ├── gp\_posteriors/               ← GP posterior slice plots per round
│   └── gradient\_heatmaps/           ← NN gradient magnitude charts
│
└── submissions/
    └── SUBMISSIONS\_LOG.md           ← copy-paste portal strings, all rounds
```

\---

## Problem Statement

Eight synthetic black-box functions `f₁(x)` … `f₈(x)` are optimised sequentially under a strict query budget. Each function:

|Function|Dimensionality|Search Space||
|-|-|-|-|
|f1|2D|\[0, 1)²||
|f2|2D|\[0, 1)²||
|f3|3D|\[0, 1)³||
|f4|4D|\[0, 1)⁴||
|f5|4D|\[0, 1)⁴||
|f6|5D|\[0, 1)⁵||
|f7|6D|\[0, 1)⁶||
|f8|8D|\[0, 1)⁸||

**Objective**: maximisation. All functions are framed as higher-is-better (minimisation problems are negated).

\---

## Methodology

### Surrogate Model

* **Gaussian Process** with Matérn-5/2 ARD kernel + WhiteKernel
* `StandardScaler` applied to X before GP fitting
* `n\_restarts\_optimizer=10`, `normalize\_y=True`
* Kernel: `ConstantKernel(1.0) × Matern(ν=2.5, ARD) + WhiteKernel`

### Acquisition Function

* **Expected Improvement (maximisation)**:

$$\\text{EI}(x) = (\\mu(x) - y^\* - \\xi),\\Phi(Z) + \\sigma(x),\\phi(Z), \\quad Z = \\frac{\\mu(x) - y^\* - \\xi}{\\sigma(x)}$$

where $y^\* = \\max\_i, y\_i$ (observed maximum), $\\xi$ is the exploration–exploitation trade-off parameter.

* **Upper Confidence Bound (UCB)** used for f6 where all outputs negative and exploration is prioritised.

### NN Surrogate (supplementary)

* MLPRegressor: 2 × 32 hidden layers, ReLU, L-BFGS-B solver, α=1e-3
* Used for finite-difference gradient analysis at the best-known point
* Identifies dominant input dimensions to inform acquisition function bounds

### Optimisation Loop

* Multi-start L-BFGS-B (35 restarts) to maximise EI
* Per-function bounds constraints derived from landscape analysis
* Manual override when GP sigma > 5× best\_Y range (GP posterior unreliable at N=4)

\---

\---

## Why Bayesian Optimisation

|Requirement|BO Solution|
|-|-|
|Expensive evaluations (1 query/round)|GP surrogate amortises each query across the whole input space|
|Non-linear, non-convex landscapes|Matérn-5/2 kernel handles discontinuities and rough surfaces|
|Unknown noise level|WhiteKernel fits observation noise automatically|
|Exploration vs exploitation|EI/UCB acquisition functions balance both principally|
|High-dimensional (up to 8D)|ARD length-scales identify inactive dimensions|

\---

## Data

Initial `.npy` datasets provided by the Imperial College London BBO portal at project start. Cumulative datasets are updated after each portal submission. **Large arrays are not stored in this repository** — see `data/DATA\_SOURCES.md`.

\---

## Dependencies

See `requirements.txt`. Core: `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `pandas`.

\---

## Reproducibility

All notebooks use `np.random.seed(42)` and `random\_state=42` throughout. GP kernel initialisations are deterministic given the seed.

\---

*Repository maintained as a living document. Final submission: Module 25, Imperial College London.*

