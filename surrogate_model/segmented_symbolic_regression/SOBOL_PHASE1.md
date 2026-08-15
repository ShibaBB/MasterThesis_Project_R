# Segmented SR Sobol Phase 1

Phase 1 tests whether teacher-Sobol-guided feature exposure improves each
fixed frequency segment. It does not add overlap, blending, smoothing, or a
continuity penalty.

## Matched branches

- `all`: all five varying material parameters plus frequency.
- `sobol_subset`: the target- and segment-specific subset in
  `sobol_feature_policy.py`, plus frequency.
- `sobol_plus_high_lambda_prime`: the Sobol subset with `lambda_prime` restored
  only for 4000-4950 Hz. This is a diagnostic branch, not part of the initial
  matched A/B run.

Feature selection occurs after the existing training-fitted log10 transform
and constant-column removal. It changes only the matrix view passed to PySR;
the canonical dataset is never modified.

## Formal matched run

Use the project PySR environment from the repository root:

```powershell
surrogate_model/segmented_symbolic_regression/.venv_py311/Scripts/python.exe `
  surrogate_model/segmented_symbolic_regression/run_sobol_phase1_experiments.py `
  --dataset-run run1 --execute
```

Omit `--execute` to create a manifest and inspect every command without
starting PySR. The initial formal comparison keeps seed 42, 100 iterations, 12
populations, population size 80, max complexity 24, full segment training
rows, and the shared run1 split for both branches.

For a pipeline smoke run, set a separate label and a small matched budget:

```powershell
surrogate_model/segmented_symbolic_regression/.venv_py311/Scripts/python.exe `
  surrogate_model/segmented_symbolic_regression/run_sobol_phase1_experiments.py `
  --targets re --niterations 1 --populations 2 --population-size 10 `
  --maxsize 8 --max-samples-per-segment 100 --run-label sobol_phase1_smoke --execute
```

Each evaluator writes `candidate_metrics.csv`,
`selected_segment_metrics.csv`, and `selected_combination_summary.json`.
Feature-policy and candidate decisions must be made from validation rows only;
test metrics remain final reporting and must not choose A versus B. The
comparison also reports aggregate metrics for the validation-selected hybrid
of A/B segments in `hybrid_selected_segment_metrics.csv` and
`validation_branch_decision.json`.

After both matched evaluations finish, create the validation-only A/B decision
table with `compare_sobol_phase1_results.py`. The comparison script verifies
the dataset, split hash, target, random seed, training budget, sampling cap,
model-selection setting, and operator contract before comparing branches. It
does not read test metrics for the feature-branch decision.
