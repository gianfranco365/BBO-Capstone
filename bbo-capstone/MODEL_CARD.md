# Model Card — BBO Capstone Surrogate Models

*Following the Model Cards for Model Reporting framework (Mitchell et al., 2019)*

---

## Model Details

| Attribute | Value |
|-----------|-------|
| **Primary surrogate** | Gaussian Process Regressor (GPR) |
| **Kernel** | `ConstantKernel(1.0) × Matérn(ν=2.5, ARD) + WhiteKernel` |
| **Input scaling** | `StandardScaler` (zero mean, unit variance) |
| **Output normalisation** | `normalize_y=True` (internal to GPR) |
| **Hyperparameter optimisation** | `n_restarts_optimizer=10`, marginal likelihood maximisation |
| **Acquisition function** | Expected Improvement (EI) — maximisation convention |
| **Secondary surrogate** | MLPRegressor (2×32 ReLU, L-BFGS-B, α=1e-3) |
| **Framework** | scikit-learn 1.x, scipy, numpy |
| **Random seed** | 42 throughout |
| **Developed by** | Gian Franco Cattaneo |
| **Date** | March–April 2026 |
| **Version** | Round 5 (W16) |

---

## Intended Use

### Primary use
Surrogate-based Bayesian optimisation of eight synthetic black-box functions (d=2–8D) within the Imperial College London BBO Capstone project. The GP surrogate approximates each unknown function from limited observations, and the EI acquisition function guides selection of the next query point per round.

### Secondary use
NN surrogate (MLPRegressor) is used exclusively for finite-difference gradient analysis at the best-known point — identifying dominant input dimensions to inform per-function acquisition bounds. It is **not** used for query generation.

### Out-of-scope uses
- Real-world engineering or industrial deployment without revalidation
- Functions with discontinuous or non-stationary landscapes (Matérn-5/2 assumes local stationarity)
- Problems requiring batch queries (this model generates one point per function per round)

---

## Training Data

See `DATASHEET.md` for full data documentation.

**Summary**: Each function's GP is fitted on N=4 submitted query points (Rounds 1–4) after Round 5 computation. The initial portal `.npy` files (N₀=10–40 depending on dimensionality) were not available during Rounds 1–4 and have not been incorporated into the submitted queries. Incorporating these initial arrays would substantially improve surrogate quality and is recommended for subsequent rounds.

---

## Model Architecture

### Gaussian Process Surrogate

```
Input x ∈ [0,1)^d
    ↓  StandardScaler  →  x_scaled ∈ ℝ^d
    ↓  GP posterior    →  μ(x), σ²(x)
    ↓  EI acquisition  →  EI(x) = (μ(x) − y* − ξ)·Φ(Z) + σ(x)·φ(Z)
    ↓  L-BFGS-B        →  x_next = argmax EI(x)
```

**Kernel decomposition**:
- `ConstantKernel(1.0)`: overall output scale
- `Matérn(ν=2.5, ARD)`: spatial correlation with per-dimension length-scales  
  — ARD (Automatic Relevance Determination) sets independent length-scale per input dimension, effectively identifying inactive or low-sensitivity dimensions
- `WhiteKernel`: homoscedastic observation noise fitting

**EI formulation (maximisation)**:

$$\text{EI}(x) = \begin{cases}
(\mu(x) - y^* - \xi)\,\Phi(Z) + \sigma(x)\,\phi(Z) & \text{if } \sigma(x) > 0 \\
0 & \text{otherwise}
\end{cases}$$

$$Z = \frac{\mu(x) - y^* - \xi}{\sigma(x)}, \quad y^* = \max_{i=1}^{N} y_i$$

Where $\Phi$ is the standard normal CDF and $\phi$ is the standard normal PDF.

### NN Surrogate (gradient analysis only)

```
Architecture:  input(d) → Dense(32, ReLU) → Dense(32, ReLU) → output(1)
Solver:        L-BFGS-B
Regularisation: L2, α=1e-3
Gradient:      finite-difference central scheme, ε=1e-4
```

---

## Evaluation

### GP Posterior Quality Indicators

| Metric | Assessment method |
|--------|------------------|
| Log marginal likelihood | Reported per function per round |
| ARD length-scales | Short scale → high sensitivity; long scale → low sensitivity |
| Posterior σ at query point | Low σ → confident prediction; high σ → uncertain (explore) |
| LOO-MSE (normalised) | Reported in R4 notebook for GP, SVR, NN comparison |

### Round 4 LOO-MSE Comparison (R1–R3 data, normalised Y)

| Function | LR | SVR | NN | Winner |
|----------|-----|-----|-----|--------|
| f1 | 2.047 | 2.301 | 2.841 | LR |
| f2 | 488.6 | 3.241 | 54.91 | SVR |
| f3 | 2.122 | 2.158 | 1.451 | NN |
| f4 | 0.889 | 1.507 | 0.386 | NN |
| f5 | 9.279 | 2.251 | 2.493 | SVR |
| f6 | 2.177 | 2.257 | 1.839 | NN |
| f7 | 1.469 | 1.507 | 0.608 | NN |
| f8 | 1.485 | 1.508 | 0.766 | NN |

*Note: with N=3–4, LOO-MSE is indicative only. GP remains primary surrogate for all functions.*

---

## Per-Function Model Behaviour (Round 5)

| f | GP kernel (fitted) | ARD top dim | σ at query | Reliability |
|---|-------------------|-------------|------------|-------------|
| f1 | ConstK × Matérn([10, 0.38]) + WK(1e-6) | x2 | ~0 | Low (flat landscape) |
| f2 | ConstK × Matérn([0.01, 0.01]) + WK(0.096) | both | 0.062 | Moderate |
| f3 | ConstK × Matérn([0.18, 10, 10]) + WK(1e-4) | x1 | 0.113 | Moderate |
| f4 | ConstK × Matérn([1.40, 10, 10, 10]) + WK(1e-8) | x1 | 6.17 | **Low** — manual override |
| f5 | ConstK × Matérn([0.04, 0.74, 1.02, 6.30]) + WK(0.10) | x2 | 700 | Moderate (high range) |
| f6 | ConstK × Matérn([3.40, 10, 10, 10, 1.65]) + WK(2e-8) | x1 | 0.182 | Moderate |
| f7 | ConstK × Matérn([7.03, 10, 10, 10, 0.99, 10]) + WK(1e-8) | x5 | 0.348 | Moderate |
| f8 | ConstK × Matérn([10, 10, 10, 0.07, 0.43, 10, 0.86, 10]) + WK(1e-8) | x4 | 0.406 | Good |

*WK = WhiteKernel. ARD length-scale ≤ 0.5 → high sensitivity; ≥ 5.0 → low sensitivity.*

---

## Known Limitations

### 1. Small sample size
With N=4 per function, GP hyperparameter estimation is unreliable. Length-scale estimates can be degenerate (hitting bounds at 0.01 or 10.0). Results should be interpreted with wide confidence intervals.

### 2. Stationarity assumption
Matérn-5/2 assumes the function's smoothness is uniform across the input space. Functions with regime changes (e.g., f4's boundary collapse from +0.27 to −30) violate this assumption locally.

### 3. Single-point acquisition
One query per round per function means slow convergence. Batch EI or Thompson sampling would accelerate optimisation but require modification to the acquisition pipeline.

### 4. NN gradient reliability
The MLPRegressor is trained on N=4 points. Finite-difference gradients at the best-known point are sensitive to overfitting and may not represent true function gradients. They are used as directional indicators, not precise descent vectors.

### 5. Round 4 maximisation error
Round 4 used minimisation EI (`y_best = min(Y)`), causing four functions to deteriorate. This is corrected in Round 5. Historical best values for f4, f5, f7, f8 from Round 4 should be disregarded as exploration results.

---

## Assumptions

1. All eight functions return noise-free or low-noise scalar outputs for a given input.
2. The portal applies the maximisation transformation correctly (minimisation functions are negated).
3. The search space is [0, 1)^d per the project specification; boundary value 0.999999 is used in place of 1.0.
4. The GP kernel hyperparameters learned on N=4 points are sufficient for local landscape characterisation near the observed best point.

---

## Interpretability

### What the GP posterior tells us
- **μ(x)**: best estimate of function value at unobserved point x
- **σ(x)**: uncertainty — high where no data exists; low near observed points
- **ARD length-scales**: proxy for input dimension sensitivity. Short length-scale = the function changes rapidly in that direction = high sensitivity.

### What the NN gradient tells us
- **∂ŷ/∂xᵢ at best_x**: estimated rate of change of the function output with respect to each input, at the current best point
- Positive gradient → increasing that input should improve output (at that point)
- Large magnitude → that dimension has high local sensitivity

### What EI tells us
- High EI → either the GP predicts a better output than current best (exploitation), or uncertainty is high enough to warrant exploration
- ξ parameter controls the balance: small ξ → exploit; large ξ → explore

---

## Ethical Considerations

This model is used exclusively for academic purposes within a controlled educational exercise. The synthetic functions have no real-world consequences. No personal data, proprietary information, or sensitive materials are involved.

---
--------------------------
--------------------------------------------
# Model Card — GP-EI BBO Surrogate

**Author:** Gian Franco Cattaneo | Imperial Business School Executive Master ML/AI  
**Last updated:** May 2026 | Round 8 submitted (W19) — 56 observations total (7 rounds × 8 functions)

---

## Model Summary

| Attribute | Value |
|---|---|
| Surrogate type | Gaussian Process Regressor |
| Kernel | ConstantKernel (amplitude) × Matérn-5/2 ARD + WhiteKernel |
| Input scaling | StandardScaler (zero mean, unit variance per dimension) |
| Y scaling | `normalize_y=True` in GPR (internal standardisation) |
| Acquisition function | Expected Improvement — **maximisation** formulation, ξ = 0.01 |
| EI optimiser | L-BFGS-B, 35 restarts; warm-start from top-35 of 5,000 random candidates |
| GP hyperparameter fitting | Marginal log-likelihood maximisation, 10 L-BFGS-B restarts (`n_restarts_optimizer=10`) |
| Random seed | 42 (all stochastic components) |
| Implementation | scikit-learn ≥1.3, scipy ≥1.11, numpy ≥1.24 |

> **W19 update:** Kernel upgraded with an explicit `ConstantKernel` amplitude factor,
> decoupling signal scale from Matérn length-scales. Acquisition warm-start revised:
> 5,000 random candidates are evaluated first; the top-35 by EI value seed L-BFGS-B,
> replacing the prior pure random start approach.

---

## EI Formula (Maximisation)

```
EI(x) = (μ(x) − y* − ξ) · Φ(Z) + σ(x) · φ(Z)
where Z = (μ(x) − y* − ξ) / σ(x)
      y* = max(y_observed)   ← MAXIMISATION (not minimisation)
      ξ  = 0.01              ← exploration offset
```

> **Critical note:** Round 4 contained a minimisation bug (`y_best = np.min(y)`)
> corrected in Round 5. All Rounds 5–8 results use the maximisation formulation above.

---

## Kernel Specification

```python
kernel = (
    ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3)   # W19: amplitude factor added
    )
    * Matern(
        length_scale=np.ones(d),             # ARD: one θⱼ per dimension
        length_scale_bounds=(1e-3, 10.0),
        nu=2.5                               # twice-differentiable (Stein 1999)
    )
    + WhiteKernel(
        noise_level=1e-4,
        noise_level_bounds=(1e-8, 1e-1)     # W19: upper bound tightened from 1e1
    )
)
```

**W19 kernel change rationale:**
- *ConstantKernel addition:* decouples signal amplitude from the Matérn
  length-scales. In W18, amplitude was implicitly absorbed into length-scale
  hyperparameters, creating identifiability pressure in exploit clusters where
  Δ‖x‖ < 0.02. The explicit amplitude parameter restores orthogonality between
  scale and shape.
- *WhiteKernel upper bound tightened (1e1 → 1e-1):* with 7 rounds confirmed
  deterministic oracle, excessive noise capacity risks absorbing genuine signal
  near the f3/f7 converging gradients.

---

## Acquisition Optimisation — W19 Architecture

```
Warm-start protocol (R8):
1. Draw 5,000 uniform random candidates over [0, 0.999999]^d
2. Evaluate EI at all 5,000 points (GP forward pass, vectorised)
3. Select top-35 by EI value as L-BFGS-B starting points
4. Run 35 × L-BFGS-B (maxiter=500, ftol=1e-12)
5. Return argmax across all 35 local optima
```

Compared to the W18 protocol (35 pure random starts), this warm-start approach
concentrates gradient descent in the high-EI region of the acquisition landscape,
reducing wasted starts in the near-zero EI plateau that dominates most of the domain
in late exploitation rounds.

---

## Known Limitations

| Function | Limitation | Status after R7 | Mitigation in R8 |
|---|---|---|---|
| f1 (d=2) | Near-zero outputs across all 7 rounds; GP surface effectively flat | R7 = 1.31e-7 (new best; magnitude ≈ 0) | Manual exploit-nudge near R7 cluster |
| f2 (d=2) | Sharp local peak; near-identical R1/R3 inputs yield Δy = 0.20 | R1 still best (0.7237); R7 = 0.487 | Probe x2 = 0.394 (below R1's 0.396) |
| f4 (d=4) | Bimodal surface (interior ≈ +0.46 vs corner ≤ −30); R7 overshot | R6 confirmed local max (0.4636); R7 = 0.363 | Fine exploit between R3 and R6 |
| f5 (d=4) | Converged at corner [0,1,1,1]; R5=R6=R7=4440.48 | Global max confirmed; 3-round plateau | Hold position |
| f6 (d=5) | R1 still best (−0.5508); all subsequent probes inferior | R7 = −0.636 — worst so far in active region | Fine probe x1 ↓ from R1; x4=1, x5=0 fixed |
| f8 (d=8) | Near-plateau; 9 ARD hyperparameters from 7 points — underdetermined | R7 = 9.8596 via x3 nudge (0.123→0.124) | Increment x3: 0.124→0.126 |
| All | Oracle deterministic; WhiteKernel models numerical precision only | — | `noise_level_bounds=(1e-8, 1e-1)` |

---

## HEBO Design Principles — Mapping to This Implementation

| HEBO Component | This Project Analogue | W19 Status |
|---|---|---|
| Matérn-5/2 ARD kernel | Per-dimension length-scale in `build_kernel()` | Retained |
| ConstantKernel amplitude | Explicit signal scale, decoupled from shape | **Added W19** |
| WhiteKernel noise floor | Explicit noise bounds | Updated bounds |
| Maximisation EI (ξ=0.01) | `expected_improvement()`: `y_best = np.max(y)` | Confirmed correct from R5 |
| Multi-restart L-BFGS-B | `optimise_acquisition()` with `n_restarts=35` | **Warm-start upgraded W19** |
| MACE Pareto ensemble (implicit) | Per-function strategy blend | EI + directional + hold |
| Input warping | Not applied | Under consideration for f1 |

> Reference: Cowen-Rivers et al. (2022). *HEBO: Pushing the Limits of Sample-Efficient
> Hyperparameter Optimisation*. JAIR. Winner of NeurIPS 2020 BBO challenge.

---

## Convergence Status (after Round 7)

| Function | Status | Evidence | R8 Strategy |
|---|---|---|---|
| F1 | Plateau — near-zero everywhere | R7 = 1.31e-7 (new best; effectively 0) | Exploit-nudge near R7 cluster |
| F2 | Plateau — sharp local peak | R7 = 0.487; R1 still best (0.724) | Probe x2 = 0.394 |
| F3 | **Active improvement** — monotone Δ≈+0.018/round | R5→R6→R7: −0.071→−0.053→−0.035 | Directional: x1↓ x3↑ |
| F4 | Local max confirmed at R6 | R7 regression: 0.363 < R6's 0.464 | Fine exploit between R3 and R6 |
| F5 | **Converged** — global max confirmed | R5=R6=R7=4440.48; 3-round plateau | Hold [0,1,1,1] corner |
| F6 | Stalled — R1 holds; probes consistently inferior | R7 = −0.636 (below R1's −0.551) | Fine probe x1 ↓ from R1 |
| F7 | **Active improvement** — uniform gradient | R1→R7: 2.207→2.250; x1=0 locked | Continue −0.003 step |
| F8 | Near-plateau — x3 the only active gradient | R7 = 9.8596 via x3 nudge | Increment x3: 0.124→0.126 |

---

## GP Posterior at Round 8 Query Points

| Function | y_best (obs) | μ (pred) | σ (pred) | EI | Mode |
|---|---|---|---|---|---|
| F1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | Exploit-nudge |
| F2 | 0.723740 | 0.528056 | 0.032649 | 0.000000 | Exploit-probe |
| F3 | −0.035298 | −0.019700 | 0.003575 | 0.005688 | Directional |
| F4 | 0.463617 | 0.466031 | 0.007710 | 0.000662 | Fine-exploit |
| F5 | 4440.482960 | 4440.480858 | 0.192970 | 0.071084 | Hold-max |
| F6 | −0.550775 | −0.591326 | 0.049662 | 0.003999 | Fine-exploit |
| F7 | 2.250193 | 2.260488 | 0.001319 | 0.000687 | Directional |
| F8 | 9.859577 | 9.859639 | 0.000208 | 0.000000 | x3 gradient |

> F3 (EI = 0.005688) and F7 (EI = 0.000687) are the only functions with
> meaningful predicted improvement — consistent with their confirmed directional
> gradients. F4 (μ = 0.466 > y_best = 0.464, σ = 0.008) shows a positive but
> tight exploitation signal.

---

## Reproducibility

All notebooks are fully executable with `jupyter nbconvert --execute`. Output figures and
submission strings are deterministic given `random_state=42` and `np.random.seed(42)`.


## Citation

Cattaneo, G.F. (2026). *Bayesian Black-Box Optimisation Capstone Project*. Executive Master in Machine Learning & Artificial Intelligence, Imperial College London. GitHub repository: [link to be added at Module 25 submission].
