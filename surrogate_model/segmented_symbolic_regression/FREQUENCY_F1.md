# Segmented SR Frequency Representation F1

F1 reuses the phase-one validation-selected material feature policy without
retraining F0. It keeps `log10_f` and appends three deterministic local
frequency features per segment:

```text
local_t  = (f - segment_center) / segment_width
local_t2 = local_t^2
local_t3 = local_t^3
```

The features are constructed after the persisted training-fitted base
transform. Their definitions and per-segment centers/widths are saved in
`training_metadata.json`. F1 does not force selected equations to use a
frequency feature. Candidate selection remains validation RMSE only; first-
and second-difference and peak/trough diagnostics are reported separately.

Formal run from the repository root:

```powershell
surrogate_model/segmented_symbolic_regression/.venv_py311/Scripts/python.exe `
  surrogate_model/segmented_symbolic_regression/run_frequency_f1_experiments.py `
  --dataset-run run1 --execute
```

Use a separate temporary `--artifact-root`, one iteration, and capped rows for
a smoke test. F0 training artifacts must not be overwritten.

Use `compare_frequency_f1_results.py` after F1 evaluation. The comparison
reconstructs F0 from the phase-one validation winners and reports point-value,
first-difference, second-difference, peak/trough, and explicit frequency-use
diagnostics without using shape metrics for candidate selection.

## Formal Result

The formal run completed on 2026-08-15. F0 was not retrained.

```text
Target/model   Validation RMSE   Test RMSE   Test R2
Re F0 hybrid   0.035577          0.034823    0.965604
Re F1          0.032830          0.032171    0.970643
Im F0 hybrid   0.030429          0.030686    0.921142
Im F1          0.030213          0.031974    0.914382
```

Re benefits materially, but its second-difference diagnostic worsens in six
segments. Im does not generalize as a whole and contains unstable high-curvature
expressions involving local polynomial features inside division or logarithm.
The comparison summaries deliberately record `shape_acceptance_decision` as
`not yet made`; F1 is an experiment result, not yet the final segmented model.
