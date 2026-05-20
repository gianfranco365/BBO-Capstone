# Technical Foundations of the BBO Capstone Strategy

**Author:** Gian Franco Cattaneo  
**Programme:** Executive Master in ML/AI — Imperial Business School  
**Repository:** BBO Capstone | Rounds 1–8 | Maximisation objective over [0, 0.999999]ⁿ  
**Last updated:** May 2026 (W19)

---

## 1. Core Technical Justification

The strategy is built on **Bayesian Optimisation (BO)** with a **Gaussian Process (GP) surrogate
and Expected Improvement (EI) acquisition function**, configured for a maximisation objective.
This combination is the canonical approach for sample-efficient black-box optimisation of
expensive, gradient-free functions — precisely the capstone setting where each query has a
fixed cost and no analytical gradient is available.

The foundational justification derives from **Jones et al. (1998)** *"Efficient Global
Optimization of Expensive Black-Box Functions"* (Journal of Global Optimization, 13, 455–492),
which introduced the **EGO algorithm** — the direct predecessor of the GP-EI pipeline
implemented here. EGO demonstrated that a GP posterior provides both a predictive mean
(exploitation signal) and a predictive variance (exploration signal), and that EI naturally
balances these without requiring manual tuning of a separate exploration coefficient.

The choice of the **Matérn-5/2 kernel** over the standard squared-exponential (RBF) kernel is
justified by **Stein (1999)** and later operationalised by **Rasmussen & Williams (2006)**:
Matérn-5/2 assumes twice-differentiable functions — a weaker and empirically more robust
smoothness assumption than the infinitely differentiable RBF, which over-smooths irregular
surfaces and can mislead the posterior in low-data regimes. With 7–8 observations per
function through Round 8, this distinction remains material.

**Automatic Relevance Determination (ARD)** — a separate length-scale θⱼ per input
dimension — is justified by **Neal (1996)** and is particularly critical for higher-dimensional
functions (f7: 6D, f8: 8D) where active and inactive dimensions must be distinguished
automatically from data.

---

## 2. Key Academic References

| Paper | Relevance to this project |
|---|---|
| Jones, Schonlau & Welch (1998). *Efficient Global Optimization of Expensive Black-Box Functions*. J. Global Optim. | Foundational EGO algorithm; EI acquisition function derivation |
| Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press. | GP theory, Matérn kernels, ARD, marginal likelihood optimisation |
| Snoek, Larochelle & Adams (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. NeurIPS. | Practical BO implementation; L-BFGS-B for EI maximisation; ARD priors |
| Brochu, Cora & de Freitas (2010). *A Tutorial on Bayesian Optimization*. arXiv:1012.2599. | EI vs UCB vs PI acquisition; ξ parameter sensitivity |
| Stein (1999). *Interpolation of Spatial Data*. Springer. | Theoretical basis for Matérn kernel preference over RBF |
| Bull (2011). *Convergence Rates of Efficient Global Optimization Algorithms*. JMLR 12. | Convergence guarantees for GP-EI under Matérn smoothness assumptions |
| Bergstra & Bengio (2012). *Random Search for Hyper-Parameter Optimization*. JMLR 13. | Benchmarks confirming GP-BO superiority over random/grid search |
| Cowen-Rivers et al. (2022). *HEBO: Pushing the Limits of Sample-Efficient HPO*. JAIR. | NeurIPS 2020 BBO challenge winner; MACE ensemble and heteroscedastic GP — applied in W18/W19 pipeline mapping |

**Most directly applied ideas:**

- **EI for maximisation** (Jones et al., Brochu et al.): `EI(x) = (μ(x) − y* − ξ)·Φ(Z) + σ(x)·φ(Z)` where `y* = max(y_observed)` — sign convention confirmed explicitly from Round 5 onward.
- **Multi-start L-BFGS-B with warm-start** (Snoek et al., upgraded W19): top-35 of 5,000 random EI evaluations seed L-BFGS-B, replacing pure random starts. This concentrates gradient descent in the high-EI region — critical in late exploitation rounds where the acquisition landscape is sharply concentrated near the current best.
- **ConstantKernel amplitude decoupling** (W19 addition): separates signal scale from Matérn length-scales, restoring hyperparameter identifiability in tight exploit clusters (Δ‖x‖ < 0.02).
- **Log-space GP for f1** (candidate — not yet applied): outputs spanning ~270 orders of magnitude violate GP stationarity. R6's negative value (−5.32e-7) and R7's near-zero best (1.31e-7) confirm the signal-to-noise ratio is too low for the GP to provide meaningful guidance.
- **HEBO design principles** (Cowen-Rivers et al.): WhiteKernel noise floor, ARD length-scale regularisation, and per-function exploit/explore blending — all operationalised in `build_kernel()` and `optimise_acquisition()` (see §4).

---

## 3. Third-Party Libraries

| Library | Version | Role | Justification vs alternatives |
|---|---|---|---|
| `scikit-learn` | ≥1.3 | `GaussianProcessRegressor`, `ConstantKernel`, `Matern`, `WhiteKernel`, `StandardScaler` | Production-grade GP with ARD support, marginal likelihood optimisation, and stable API. `ConstantKernel` added in W19 to decouple amplitude. GPyTorch offers deep kernels but requires GPU infrastructure not available here. GPflow adds unnecessary dependency weight for 8-function, 7-point datasets. |
| `scipy` | ≥1.11 | `minimize` (L-BFGS-B, `maxiter=500, ftol=1e-12`), `norm` (EI computation) | Standard choice for constrained EI maximisation (Snoek et al., 2012). W19 adds tighter convergence tolerance (`ftol=1e-12`) to improve acquisition precision in late-exploitation rounds. |
| `numpy` | ≥1.24 | Array operations, `np.maximum.accumulate` for best-so-far tracking, warm-start candidate generation | Foundational. `np.random.seed(42)` ensures reproducibility across the full 5,000-candidate warm-start and 35-restart optimisation pipeline. |
| `matplotlib` | ≥3.7 | GP posterior visualisation (convergence trajectories, posterior diagnostics, R8 query comparison) | Chosen over Plotly/Seaborn for static reproducible figures in a Jupyter/GitHub context. |

**Design choice: no dedicated BO library (BoTorch, Ax, Optuna)**  
Deliberate. Using `scikit-learn` primitives directly exposes the full GP-EI pipeline — kernel
construction, marginal likelihood optimisation, EI evaluation, multi-start maximisation — making
every architectural decision explicit and auditable. A black-box BO library would obscure the
choices that are the learning objective of this capstone.

---

## 4. HEBO Connection — W18/W19 Module Alignment

Rounds 7–8 explicitly map the BBO pipeline to HEBO design principles, establishing the bridge
between this capstone and the W18/W19 hyperparameter tuning modules:

| HEBO Component | This Project Analogue | Implementation | W19 Status |
|---|---|---|---|
| Matérn-5/2 ARD kernel | Per-dimension length-scale | `Matern(nu=2.5, length_scale=np.ones(d))` | Retained |
| ConstantKernel amplitude | Explicit signal scale | `ConstantKernel(1.0, (1e-3, 1e3))` | **Added W19** |
| WhiteKernel noise floor | Numerical stability | `noise_level_bounds=(1e-8, 1e-1)` | Bounds updated W19 |
| Maximisation EI (ξ=0.01) | `expected_improvement()` | `y_best = np.max(y)` — corrected from R4 | Confirmed |
| Multi-restart L-BFGS-B | `optimise_acquisition()` | 35 starts, `maxiter=500`, `ftol=1e-12` | **Warm-start upgraded W19** |
| MACE Pareto ensemble (implicit) | Per-function strategy blend | EI + directional + hold (per-function) | R8 applied |
| Input warping | Not applied | — | Candidate for f1 if further rounds |

**SALOV Industrial Mapping (R8 Reflection):**  
The 8-function portfolio continues to mirror multi-KPI OEE optimisation on the 12,000 bph PET bottling line, with sharper convergence diagnostics available at Round 8:

- **f5** (global max confirmed, 3-round plateau) ↔ a process parameter already operating at its physical optimum — further trials consume budget with zero expected gain, analogous to running trials on a bottling parameter confirmed at its engineering limit.
- **f4** (R7 regression after R6 peak) ↔ fill-volume overshoot: crossing the optimal setpoint triggers a non-conformance cascade. The R3→R6 improvement followed by R7 regression is precisely the OEE signature of overtuning a closed-loop controller.
- **f3, f7** (monotone directional improvement, Δ≈0.018 and Δ≈0.012 per round) ↔ active PID loop tuning: a stable, decelerating improvement trajectory is the expected response when nudging a tuning parameter toward its optimum under process inertia.
- **f1** (near-zero everywhere) ↔ a conveyor timing parameter operating in a dead band: the signal is so weak that measurement noise dominates, and additional trials without a structural change to the experiment design (e.g., wider exploration or log-scale analysis) yield no actionable gradient.
- **f8** (x3 as the sole active gradient) ↔ a high-dimensional process where only one parameter drives the KPI — analogous to identifying that only the sealing temperature (x3) affects seal integrity across the 8-parameter bottling configuration.

---

## 5. GitHub Documentation Plan

The repository is structured to make all technical choices traceable:

```
bbo-capstone/
├── README.md                             # Project overview, objective, submission format
├── TECHNICAL_FOUNDATIONS.md             # This document — academic and software justifications
├── DATASHEET.md                         # Dataset provenance: portal inputs/outputs per round
├── MODEL_CARD.md                        # GP surrogate specification: kernel, scaler, EI formula
├── SUBMISSIONS_LOG.md                   # All 8 rounds: query strings, best-observed, strategy notes
├── notebooks/
│   ├── BBO_Round8_Submission.ipynb      # W19 — ConstantKernel + warm-start, fully executable
│   ├── BBO_Round7_GianFranco_Cattaneo.ipynb   # W18
│   ├── BBO_Round6_GianFrancoCattaneo.ipynb    # W17
│   └── ...
└── reflections/
    ├── W19_reflection.md                # Module 19.1: refining strategies; LLM-BBO analogy
    └── W18_reflection.md                # HEBO connection + industrial mapping
```

**Visibility conventions applied throughout:**

- Every kernel parameter in `build_kernel()` has an inline comment citing source paper and change rationale.
- The EI formula in `expected_improvement()` includes the maximisation sign convention and the Round 5 bug-fix note.
- `SUBMISSIONS_LOG.md` records strategy rationale, GP posterior diagnostics (μ, σ, EI at query point), and `‖Δx‖` per function per round.
- `MODEL_CARD.md` states known limitations, convergence status, and GP posterior predictions at the next query point — queryable by a peer reviewer without running any code.
- The GP-EI suggestion vs strategic final comparison table (Cell 6 in R8 notebook) documents every expert override decision with quantitative justification.

---

## 6. Forward-Looking Sources

| Source | Purpose |
|---|---|
| **BoTorch** (Balandat et al., 2020, NeurIPS) | Benchmark the current pipeline against GPU-accelerated batch-parallel BO — particularly relevant for f8 (8D search space) and as a final-round alternative if further queries are possible. |
| **HEBO** (Cowen-Rivers et al., 2022, JAIR) | Winning NeurIPS 2020 BBO submission; heteroscedastic GP + evolutionary acquisition — directly applicable to f4's bimodal surface. Applied as the W18/W19 module connection in §4. |
| **Nevergrad** (Rapin & Teytaud, 2018, Facebook AI) | Gradient-free benchmark suite; quantitative comparison of GP-EI against CMA-ES and differential evolution on functions with the structural signatures observed (boundary optima for f5, sharp ridges for f7, near-zero surfaces for f1). |
| **BBOB benchmark suite** (Hansen et al., 2009) | Standard 24-function testbed; comparing f1–f8 structural signatures against BBOB classes would allow literature-grounded convergence rate expectations. |
| **Frazier (2018)**. *A Tutorial on Bayesian Optimization*. arXiv:1807.02811. | Most current accessible survey; covers batch BO, noisy observations, constrained BO. |
| **Input warping via Kumaraswamy CDF** (Snoek et al., 2014, ICML) | Addresses the f1 near-zero surface: non-linear input transformation improves GP posterior quality when outputs concentrate near machine-precision boundaries. The R7 best of 1.31e-7 confirms the problem persists; this is the most actionable structural improvement available if additional rounds were permitted. |
| **Thompson Sampling for BO** (Chapelle & Li, 2011, NeurIPS) | Alternative acquisition function to EI; draws a GP sample and maximises it, providing natural exploration-exploitation balance without the ξ parameter. Relevant for f2 (sharp, stochastic-appearing landscape) where ξ tuning is difficult. |
