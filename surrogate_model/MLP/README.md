# MLP For Reflection-Coefficient Components

The MLP branch trains two independent curve-to-curve regressors:

```text
X -> Y_re
X -> Y_im
```

Each network keeps the existing architecture, preprocessing, optimizer, shared
curve split, and 128-output frequency grid for the first migration. A target
selector must load either `Y_re` or `Y_im`; the two targets must never create
or overwrite the same artifact directory.

## Evaluation

Report raw RMSE, MAE, maximum absolute error, R2, component range, error versus
frequency, scatter, and curve comparisons. Do not clip predictions.

## Planned Artifacts

```text
artifacts/re/20260809_run1/
artifacts/im/20260809_run1/
```

The date is illustrative. Use the execution date and keep all detailed
hyperparameters inside saved metadata.

Both component interfaces completed a one-epoch smoke training. Full training
has not been launched. `mlp_strategy.md` and `mlp_baseline_build.md` describe the superseded target
workflow. They are historical references only and are not authoritative for
this repository. Use `../current_project_handoff.md` for migration.
