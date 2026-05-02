# Data Sources

Large `.npy` arrays are **not stored in this repository** per project guidelines.

## Initial Dataset

Provided by the Imperial College London BBO Capstone Project portal at project start.  
Download location: Imperial College BBO Portal (authenticated access, course participants only).

Files structure (one per function):
```
initial_inputs_f1.npy   # shape (10, 2)
initial_inputs_f2.npy   # shape (10, 2)
initial_inputs_f3.npy   # shape (15, 3)
initial_inputs_f4.npy   # shape (30, 4)
initial_inputs_f5.npy   # shape (20, 4)
initial_inputs_f6.npy   # shape (20, 5)
initial_inputs_f7.npy   # shape (30, 6)
initial_inputs_f8.npy   # shape (40, 8)

initial_outputs_f1.npy  # shape (10,)
...
initial_outputs_f8.npy  # shape (40,)
```

## Cumulative Submission Data

After each portal submission, cumulative `.npy` files are available for download.  
These include all data from the initial dataset plus all submitted queries to date.

Files naming convention (portal):
```
inputs.npy   # cumulative inputs, all rounds
outputs.npy  # cumulative outputs, all rounds
```

## Submitted Query Data (inline)

All submitted query points (R1–R5) and their portal-returned output values are documented in full in `../SUBMISSIONS_LOG.md` and hardcoded into each round's notebook in `../notebooks/`. This ensures full reproducibility without requiring access to the portal.
