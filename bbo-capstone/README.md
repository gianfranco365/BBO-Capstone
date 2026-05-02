# Bayesian Black-Box Optimisation — Capstone Project

**Imperial College London · Executive Master in Machine Learning & Artificial Intelligence**  
**Gian Franco Cattaneo · COO, SALOV S.p.A.**

---

## Non-Technical Summary (~100 words)

This project applies Bayesian optimisation to eight unknown mathematical functions, each accepting between two and eight numerical inputs and returning a single score. The goal is to find the input combination that produces the highest score, using as few experiments as possible — much like tuning a complex industrial process without being able to see inside it. A Gaussian Process surrogate model learns from each result to predict where the next best experiment should be run. Across five submission rounds, the algorithm progressively refined its search, achieving consistent improvements on six of eight functions. The project demonstrates how data-efficient optimisation can replace exhaustive trial-and-error in real-world engineering and business contexts.

---

## Project Structure

```
bbo-capstone/
├── README.md                        ← this file
├── DATASHEET.md                     ← dataset description, provenance, limitations
├── MODEL_CARD.md                    ← model behaviour, assumptions, interpretability
├── SUBMISSIONS_LOG.md               ← portal submission strings, round-by-round record
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── notebooks/
│   ├── BBO_Round1_W12.ipynb         ← Submission 1: initial GP-EI, d=2–8D
│   ├── BBO_Round2_W13.ipynb         ← Submission 2: updated surrogate, boundary analysis
│   ├── BBO_Round3_W14.ipynb         ← Submission 3: SVM classification framing
│   ├── BBO_Round4_W15.ipynb         ← Submission 4: NN surrogate + gradient analysis
│   └── BBO_Round5_W16.ipynb         ← Submission 5: corrected maximisation EI, R1–R4 data
│
├── data/
│   └── DATA_SOURCES.md              ← external data description (no large files on GitHub)
│
├── plots/
│   ├── convergence/                 ← best Y per round per function
│   ├── gp_posteriors/               ← GP posterior slice plots per round
│   └── gradient_heatmaps/           ← NN gradient magnitude charts
│
└── submissions/
    └── SUBMISSIONS_LOG.md           ← copy-paste portal strings, all rounds
```

---

## Problem Statement

Eight synthetic black-box functions `f₁(x)` … `f₈(x)` are optimised sequentially under a strict query budget. Each function:

| Function | Dimensionality | Search Space | Notes |
|----------|---------------|-------------|-------|
| f1 | 2D | [0, 1)² | Near-flat landscape |
| f2 | 2D | [0, 1)² | Sharp local peak near x1≈0.70 |
| f3 | 3D | [0, 1)³ | All-negative outputs; interior optimum |
| f4 | 4D | [0, 1)⁴ | Bimodal: interior good, boundary catastrophic |
| f5 | 4D | [0, 1)⁴ | High dynamic range; x2-dominant (NN gradient=4751) |
| f6 | 5D | [0, 1)⁵ | All-negative; x4 high + x5 low structural pattern |
| f7 | 6D | [0, 1)⁶ | Sparse: x1=0 locked; confirmed optimum at R1/R3 |
| f8 | 8D | [0, 1)⁸ | x5 boundary-locked; tight optimum at R1/R3 |

**Objective**: maximisation. All functions are framed as higher-is-better (minimisation problems are negated).

---

## Methodology

### Surrogate Model
- **Gaussian Process** with Matérn-5/2 ARD kernel + WhiteKernel
- `StandardScaler` applied to X before GP fitting
- `n_restarts_optimizer=10`, `normalize_y=True`
- Kernel: `ConstantKernel(1.0) × Matern(ν=2.5, ARD) + WhiteKernel`

### Acquisition Function
- **Expected Improvement (maximisation)**:

$$\text{EI}(x) = (\mu(x) - y^* - \xi)\,\Phi(Z) + \sigma(x)\,\phi(Z), \quad Z = \frac{\mu(x) - y^* - \xi}{\sigma(x)}$$

where $y^* = \max_i\, y_i$ (observed maximum), $\xi$ is the exploration–exploitation trade-off parameter.

- **Upper Confidence Bound (UCB)** used for f6 where all outputs negative and exploration is prioritised.

### NN Surrogate (supplementary)
- MLPRegressor: 2 × 32 hidden layers, ReLU, L-BFGS-B solver, α=1e-3
- Used for finite-difference gradient analysis at the best-known point
- Identifies dominant input dimensions to inform acquisition function bounds

### Optimisation Loop
- Multi-start L-BFGS-B (35 restarts) to maximise EI
- Per-function bounds constraints derived from landscape analysis
- Manual override when GP sigma > 5× best_Y range (GP posterior unreliable at N=4)

---

## Results Summary (Rounds 1–5)

| Function | d | Best Y | Best Round | Key Insight |
|----------|---|--------|------------|-------------|
| f1 | 2 | ~2.7e-9 | R4 | Landscape essentially flat; maximisation near centre |
| f2 | 2 | 0.7237 | R1 | Sharp peak at (0.695, 0.396); high-confidence exploitation |
| f3 | 3 | −0.0891 | R1 | All-negative; GP exploiting interior |
| f4 | 4 | 0.2748 | R3 | Boundary collapse (−30) confirmed; interior safe zone |
| f5 | 4 | 2932.695 | R3 | x1=0 locked; x2 dominant (NN gradient); push x2→1 |
| f6 | 5 | −0.5508 | R1 | x4→1, x5→0 structural pattern; all-negative landscape |
| f7 | 6 | 2.2073 | R1 | x1=0 locked; confirmed optimum stable R1/R3 |
| f8 | 8 | 9.8595 | R1 | x5 boundary; R1/R3 near-identical; tight exploitation |

*Results updated through Round 4 portal feedback. Round 5 pending.*

---

## Key Insights

1. **Maximisation convention is critical**: Round 4 incorrectly used `y_best = np.min()` (minimisation EI), causing four function deteriorations. Corrected in Round 5.
2. **ARD length-scales reveal structure**: short length-scales flag high-sensitivity dimensions (f5: x2, x3; f8: x4, x5, x7) without requiring domain knowledge.
3. **Boundary collapse in f4**: x1→1 triggers catastrophic output (−27 to −30). Interior constraint (0.25–0.65)⁴ required from Round 5 onward.
4. **NN gradients complement GP**: at N=4, GP posteriors are unreliable in high dimensions. NN finite-difference gradients provide directional signals that GP acquisition cannot.
5. **Sparse structure in f7**: x1=x3=x5=0 is structurally required. Violations (R2, R4) consistently collapse output from 2.207 to ~0.02–0.05.

---

## Why Bayesian Optimisation

| Requirement | BO Solution |
|-------------|-------------|
| Expensive evaluations (1 query/round) | GP surrogate amortises each query across the whole input space |
| Non-linear, non-convex landscapes | Matérn-5/2 kernel handles discontinuities and rough surfaces |
| Unknown noise level | WhiteKernel fits observation noise automatically |
| Exploration vs exploitation | EI/UCB acquisition functions balance both principally |
| High-dimensional (up to 8D) | ARD length-scales identify inactive dimensions |

---

## Data

Initial `.npy` datasets provided by the Imperial College London BBO portal at project start. Cumulative datasets are updated after each portal submission. **Large arrays are not stored in this repository** — see `data/DATA_SOURCES.md`.

---

## Dependencies

See `requirements.txt`. Core: `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `pandas`.

---

## Reproducibility

All notebooks use `np.random.seed(42)` and `random_state=42` throughout. GP kernel initialisations are deterministic given the seed.

---

*Repository maintained as a living document. Final submission: Module 25, Imperial College London.*
