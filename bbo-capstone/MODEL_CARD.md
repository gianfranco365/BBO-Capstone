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

## Citation

Cattaneo, G.F. (2026). *Bayesian Black-Box Optimisation Capstone Project*. Executive Master in Machine Learning & Artificial Intelligence, Imperial College London. GitHub repository: [link to be added at Module 25 submission].
