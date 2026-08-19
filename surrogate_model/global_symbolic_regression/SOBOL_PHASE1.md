# Global SR Sobol Phase 1

This experiment applies the validated Segmented SR Sobol Phase 1 method to
Global SR while preserving Global SR as one equation over one frequency
domain: 100-4950 Hz. It does not create frequency segments, gates, local
experts, blending, or a segment hybrid.

## Matched branches

- `all`: all five varying material parameters plus frequency.
- `sobol_subset`: `sigma`, `alpha_infinity`, `lambda`, and `k0_prime`, plus
  frequency. `lambda_prime` is omitted because it has the lowest teacher
  variance-weighted full-range total-effect index for both Re and Im.

The feature view is applied after the training-fitted base transform and
constant-column removal. Frequency (`log10_f`) is always retained. The
canonical dataset is not modified.

## Experimental contract

For each target, both branches use the same run1 rows, shared curve split,
random seed, training-row cap, iterations, populations, population size,
maximum complexity, operators, and candidate-selection rule. Candidate
selection and the A/B feature decision use full-range validation rows only.
The frozen winner is then reported on test rows. Because Global SR has one
domain, the outcome is one whole-range branch per target; no hybrid is built.

## Run

Create a manifest without starting PySR:

```powershell
surrogate_model/segmented_symbolic_regression/.venv_py311/Scripts/python.exe `
  surrogate_model/global_symbolic_regression/run_global_sobol_phase1_experiments.py `
  --dataset-run run1
```

Run the formal matched Re/Im experiment:

```powershell
surrogate_model/segmented_symbolic_regression/.venv_py311/Scripts/python.exe `
  surrogate_model/global_symbolic_regression/run_global_sobol_phase1_experiments.py `
  --dataset-run run1 --execute
```

The formal defaults match the current Global baseline: seed 42, 100
iterations, 12 populations, population size 80, max complexity 24, and all
full-range training rows.

## Formal run1 results

The matched experiment completed on 2026-08-15. Both branches used the same
shared split hash, full training rows, seed, search budget, operators, and
validation-only candidate selection.

```text
Target/branch       Validation RMSE   Test RMSE   Test R2
Re all              0.081930          0.080918    0.814274
Re sobol_subset     0.078143          0.077609    0.829153
Im all              0.085952          0.079741    0.467469
Im sobol_subset     0.089289          0.083572    0.415070
```

The validation-only whole-range decision selects `sobol_subset` for Re and
`all` for Im. Re validation RMSE improves by 4.62 percent and its frozen test
RMSE improves by 4.09 percent. The Im subset worsens validation RMSE by 3.88
percent and is rejected; its test RMSE is also 4.80 percent worse.

Formal comparison artifacts:

```text
artifacts/re/comparison/20260815_run1_sobol_phase1
artifacts/im/comparison/20260815_run1_sobol_phase1
```
