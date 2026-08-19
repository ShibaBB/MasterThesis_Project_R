# Current Segmented Symbolic Model Overview

The retained architecture expands every material curve into scalar samples:

```text
[phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f]
```

It will train two separate piecewise symbolic models, one for `R_real` and one
for `R_imag`. Frequency segmentation, PySR hyperparameters, and shared
source-curve splitting remain unchanged for the first migration.

The branch is not yet runnable because its loaders and evaluators still use the
old single-target interface. Required changes and acceptance criteria are in:

```text
surrogate_model/current_project_handoff.md
surrogate_model/segmented_symbolic_regression/README.md
```

No generated data or formal artifacts are currently present.
