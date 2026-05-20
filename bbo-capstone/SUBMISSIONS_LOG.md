# BBO Capstone — Submissions Log

All portal submission strings, round-by-round. Format: `x1-x2-...-xn` (6 decimal places).

---

## Round 1 — W12 | Submitted: 26/03/2026

| Function | d | Query String | Y (portal feedback) |
|----------|---|-------------|-------------------|
| F1 | 2 | `0.034388-0.909319` | −2.4675e-270 |
| F2 | 2 | `0.695196-0.395970` | 0.7237 |
| F3 | 3 | `0.548145-0.174647-0.303245` | −0.0891 |
| F4 | 4 | `0.440429-0.425456-0.378357-0.397088` | 0.2596 |
| F5 | 4 | `0.000000-0.675974-0.999999-0.999999` | 2105.928 |
| F6 | 5 | `0.464677-0.242110-0.574863-0.999999-0.000000` | −0.5508 |
| F7 | 6 | `0.000000-0.241713-0.327655-0.218095-0.375335-0.747501` | 2.2073 |
| F8 | 8 | `0.064016-0.008062-0.123268-0.000000-0.999999-0.381742-0.031402-0.806010` | 9.8595 |

**Methodology:** GP surrogate (Matérn-5/2 ARD), EI acquisition (maximisation), 25 restarts L-BFGS-B.

---

## Round 2 — W13 | Submitted: 06/04/2026

| Function | d | Query String | Y (portal feedback) |
|----------|---|-------------|-------------------|
| F1 | 2 | `0.999999-0.999999` | 1.5176e-192 |
| F2 | 2 | `0.698486-0.000000` | 0.5298 |
| F3 | 3 | `0.850892-0.035316-0.936193` | −0.2398 |
| F4 | 4 | `0.999999-0.000000-0.000000-0.365908` | −27.8598 |
| F5 | 4 | `0.000000-0.000000-0.999999-0.999999` | 1616.626 |
| F6 | 5 | `0.142733-0.321812-0.416485-0.999999-0.304415` | −1.0045 |
| F7 | 6 | `0.000000-0.302741-0.000000-0.187177-0.000000-0.167182` | 0.0510 |
| F8 | 8 | `0.096074-0.000000-0.581701-0.000000-0.999999-0.383890-0.202189-0.999999` | 9.2934 |

**Methodology:** GP-EI. f4 confirmed boundary collapse (x1→1). f7: sparse structure identified (x1=x3=x5=0 required).

---

## Round 3 — W14 | Submitted: 08/04/2026

| Function | d | Query String | Y (portal feedback) |
|----------|---|-------------|-------------------|
| F1 | 2 | `0.250000-0.250000` | 9.7977e-42 |
| F2 | 2 | `0.695000-0.396000` | 0.5264 |
| F3 | 3 | `0.300000-0.500000-0.700000` | −0.1140 |
| F4 | 4 | `0.440000-0.425000-0.378000-0.397000` | 0.2748 |
| F5 | 4 | `0.000000-0.850000-0.999999-0.999999` | 2932.695 |
| F6 | 5 | `0.500000-0.500000-0.500000-0.500000-0.500000` | −1.0159 |
| F7 | 6 | `0.000000-0.242000-0.328000-0.218000-0.375000-0.748000` | 2.2072 |
| F8 | 8 | `0.064000-0.008000-0.120000-0.000000-0.999999-0.382000-0.031000-0.806000` | 9.8592 |

**Methodology:** GP-EI. f5: x2=0.85 improvement over R1 confirmed. f4: interior best confirmed.

---

## Round 4 — W15 | Submitted: 19/04/2026

| Function | d | Query String | Y (portal feedback) |
|----------|---|-------------|-------------------|
| F1 | 2 | `0.500000-0.500000` | 2.6753e-9 |
| F2 | 2 | `0.700000-0.200000` | 0.5814 |
| F3 | 3 | `0.950000-0.010000-0.990000` | −0.4594 |
| F4 | 4 | `0.999999-0.000000-0.000000-0.700000` | −30.894 |
| F5 | 4 | `0.000000-0.000000-0.500000-0.500000` | 83.963 |
| F6 | 5 | `0.300000-0.400000-0.600000-0.200000-0.600000` | −1.2239 |
| F7 | 6 | `0.000000-0.150000-0.000000-0.100000-0.000000-0.100000` | 0.0236 |
| F8 | 8 | `0.100000-0.000000-0.800000-0.000000-0.999999-0.380000-0.350000-0.999999` | 8.5129 |

**⚠ Post-mortem:** Round 4 notebook used minimisation EI (`y_best = min(Y)`). Four functions (f4, f5, f7, f8) deteriorated. Corrected in Round 5.

---

## Round 5 — W16 | Submitted: 23/04/2026

| Function | d | Query String | Strategy | Y (portal feedback) |
|----------|---|-------------|---------|-------------------|
| F1 | 2 | `0.472781-0.505546` | GP-EI free exploration | 8.1686e-8 |
| F2 | 2 | `0.695211-0.395970` | GP-EI tight exploit | 0.6239 |
| F3 | 3 | `0.511275-0.215264-0.371049` | GP-EI exploit near R1 | −0.0707 |
| F4 | 4 | `0.455000-0.415000-0.385000-0.395000` | Manual interior perturbation | −0.3997 |
| F5 | 4 | `0.000000-0.999999-0.999999-0.999999` | GP-EI + NN x2-dominant push | 4440.481 |
| F6 | 5 | `0.758817-0.272673-0.522143-0.999999-0.000000` | GP-EI x4/x5 pattern | −0.9106 |
| F7 | 6 | `0.000000-0.260000-0.340000-0.232000-0.395000-0.752000` | Manual + NN gradient nudge | 2.1133 |
| F8 | 8 | `0.040000-0.000000-0.090000-0.005000-0.999999-0.367013-0.020000-0.780000` | GP-EI tight exploit | 9.8387 |

**Methodology:** Corrected maximisation EI (`y_best = max(Y)`). GP Matérn-5/2 ARD + WhiteKernel. Round 5 highlights: f1 new best (8.17e-8); f3 new best (−0.071); f5 major breakthrough (4440 vs 2933 R3).

---

## Round 6 — W17 | Submitted: 01/05/2026

| Function | d | Query String | Strategy | Y (portal feedback) |
|----------|---|-------------|---------|-------------------|
| F1 | 2 | `0.445562-0.511092` | GP-EI exploit near R5 best | −5.3166e-7 |
| F2 | 2 | `0.693000-0.397000` | GP-EI tight exploit near R1 | 0.3979 |
| F3 | 3 | `0.490000-0.230000-0.395000` | GP-EI exploit near R5 best | −0.0529 |
| F4 | 4 | `0.430000-0.430000-0.375000-0.400000` | Manual interior perturbation near R3 | 0.4636 ✓ best |
| F5 | 4 | `0.005000-0.999999-0.999999-0.999999` | GP-EI push x1→0, x2=x3=x4→1 | 4440.483 ✓ best |
| F6 | 5 | `0.450000-0.240000-0.580000-0.999999-0.000000` | GP-EI exploit R1 x4/x5 pattern | −0.5765 |
| F7 | 6 | `0.000000-0.238000-0.325000-0.215000-0.370000-0.743000` | GP-EI exploit near R1/R5 cluster | 2.2378 |
| F8 | 8 | `0.063000-0.008000-0.123000-0.000000-0.999999-0.382000-0.031000-0.807000` | GP-EI tight exploit near R1 best | 9.8591 |

**Round 6 highlights:** f4 new best (0.464); f5 marginal new best (4440.483); f7 new best (2.238). f1 returned negative — GP surface unreliable in raw scale.

---

## Round 7 — W18 | Submitted: 06/05/2026

| Function | d | Query String | Strategy | Y (portal feedback) |
|----------|---|-------------|---------|-------------------|
| F1 | 2 | `0.475000-0.503000` | Exploit: tight ball around R5 best (GP flat surface) | 1.3111e-7 ✓ best |
| F2 | 2 | `0.697000-0.393000` | Exploit: micro-shift north-east of R1 peak | 0.4872 |
| F3 | 3 | `0.478000-0.223000-0.408000` | Extrapolate: gradient direction R5→R6 (x1↓ x3↑) | −0.0353 ✓ best |
| F4 | 4 | `0.420000-0.440000-0.373000-0.403000` | Exploit: x1↓ x2↑ axis from R6 best | 0.3635 |
| F5 | 4 | `0.000000-0.999999-0.999999-0.999999` | CONVERGED: repeat [0,1,1,1] corner | 4440.481 |
| F6 | 5 | `0.468000-0.241000-0.572000-0.999999-0.000000` | Exploit: interpolate R1↔R6 in [x1,x2,x3] | −0.6361 |
| F7 | 6 | `0.000000-0.235000-0.322000-0.212000-0.367000-0.740000` | Exploit: micro-decrement x2–x6 from R6; x1=0 anchored | 2.2502 ✓ best |
| F8 | 8 | `0.064016-0.008062-0.124000-0.000000-0.999999-0.381742-0.031402-0.806010` | CONVERGED: micro-refine x3 0.123→0.124 | 9.8596 ✓ best |

**Methodology:** GP Matérn-5/2 ARD + WhiteKernel, maximisation EI, 35-restart L-BFGS-B, 6-round history (48 obs). Kernel bounds: length-scale (1e-3, 10.0), WhiteKernel (1e-8, 1e1).

**Round 7 highlights:** f1 new best (1.31e-7, surpassing R5's 8.17e-8); f3 new best (−0.035, monotone trend R5→R6→R7 confirmed, Δ≈+0.018/round); f7 new best (2.250); f8 marginal new best (9.8596 via x3 nudge). f4 regression: R7=0.364 < R6=0.464, confirming R6 as local max. f6 deteriorated: R7=−0.636, R1 remains the unchallenged best.

---

## Round 8 — W19 | Submitted: 18/05/2026

| Function | d | Query String | Strategy | Y (portal feedback) |
|----------|---|-------------|---------|-------------------|
| F1 | 2 | `0.477000-0.501000` | Exploit near R7 positive cluster (x1 nudge +0.002, x2 nudge −0.002) | *pending* |
| F2 | 2 | `0.695000-0.394000` | Re-exploit best region; probe x2=0.394, below R1's 0.396 | *pending* |
| F3 | 3 | `0.465000-0.222000-0.421000` | Directional continuation: x1↓ x3↑ (Δx≈0.013 from R7) | *pending* |
| F4 | 4 | `0.428000-0.432000-0.374000-0.401000` | Fine exploitation between R6 (best) and R3 | *pending* |
| F5 | 4 | `0.000000-0.999999-0.999999-0.999999` | CONVERGED: hold confirmed global maximum | *pending* |
| F6 | 5 | `0.460000-0.242000-0.575000-0.999999-0.000000` | Fine probe: x1 ↓ from R1 best (0.464→0.460); x3=0.575 | *pending* |
| F7 | 6 | `0.000000-0.232000-0.319000-0.209000-0.364000-0.737000` | Uniform −0.003 step from R7; x1=0 locked | *pending* |
| F8 | 8 | `0.064016-0.008062-0.126000-0.000000-0.999999-0.381742-0.031402-0.806010` | x3 monotone gradient: 0.124→0.126 | *pending* |

**Methodology:** GP ConstantKernel × Matérn-5/2 ARD + WhiteKernel (W19 kernel). Maximisation EI (ξ=0.01), 35-restart L-BFGS-B with warm-start from top-35 of 5,000 random candidates. Full 7-round history (56 observations). Strategic overrides applied where GP-EI diverged from confirmed directional signals: f3/f7 (directional continuation confirmed), f5 (converged — hold), f8 (x3 gradient only). GP posterior at R8 query points: f3 EI=0.006, f4 μ=0.466>y_best, f7 EI=0.001 — positive signals on all three active improvement functions.

---

## Best Results Summary (through Round 7)

| Function | Best Y | Best Round | Best Query |
|----------|--------|------------|-----------|
| F1 | 1.3111e-7 | R7 | `0.475000-0.503000` |
| F2 | 0.7237 | R1 | `0.695196-0.395970` |
| F3 | −0.0353 | R7 | `0.478000-0.223000-0.408000` |
| F4 | 0.4636 | R6 | `0.430000-0.430000-0.375000-0.400000` |
| F5 | 4440.483 | R6 | `0.005000-0.999999-0.999999-0.999999` |
| F6 | −0.5508 | R1 | `0.464677-0.242110-0.574863-0.999999-0.000000` |
| F7 | 2.2502 | R7 | `0.000000-0.235000-0.322000-0.212000-0.367000-0.740000` |
| F8 | 9.8596 | R7 | `0.064016-0.008062-0.124000-0.000000-0.999999-0.381742-0.031402-0.806010` |

*Updated after R7 portal feedback. R8 feedback pending.*

**Cumulative best trajectory — key observations:**

| Function | R1 | R3 | R5 | R6 | R7 | Trend |
|---|---|---|---|---|---|---|
| F3 | −0.0891 | −0.0891 | −0.0707 | −0.0529 | −0.0353 | Monotone ↑, Δ≈+0.018/round |
| F4 | 0.2596 | 0.2748 | 0.2748 | 0.4636 | 0.4636 | Peaked R6; R7 regressed |
| F5 | 2105.9 | 2932.7 | 4440.5 | 4440.5 | 4440.5 | Plateau R5–R7 |
| F7 | 2.2073 | 2.2073 | 2.2073 | 2.2378 | 2.2502 | Steady ↑ |
| F8 | 9.8595 | 9.8595 | 9.8595 | 9.8595 | 9.8596 | Near-plateau; x3 signal |

**New bests established in R7:** F1 (1.31e-7 vs 8.17e-8 R5), F3 (−0.035 vs −0.053 R6), F7 (2.250 vs 2.238 R6), F8 (9.8596 vs 9.8595 R1).
