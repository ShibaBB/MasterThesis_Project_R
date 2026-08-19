# Global SR For Reflection-Coefficient Components

The global PySR branch fits two independent full-range equations:

```text
R_real = g_re(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
R_imag = g_im(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
```

The completed matched teacher-Sobol Phase 1 experiment is documented in
[`SOBOL_PHASE1.md`](SOBOL_PHASE1.md). Global SR remains one equation and one
frequency domain over 100-4950 Hz. For each target it compares all five
varying material parameters against a Sobol-guided subset, with frequency
always retained. It does not use frequency segments, gates, local experts, or
a segment hybrid.

The validation-only full-range decision retains `sobol_subset` for Re and
`all` for Im. The subset improves Re test RMSE from `0.080918` to `0.077609`;
for Im it worsens validation and test RMSE, so the existing all-feature model
remains the checkpoint.

The subsequent all-parameter smooth frequency-modulation experiment is
documented in [`FREQUENCY_MODULATION.md`](FREQUENCY_MODULATION.md). It retains
every base parameter and adds one teacher-Sobol-guided smooth modulation
terminal per parameter. Validation selects the modulated model for both
targets, with test RMSE `0.069470` for Re and `0.070212` for Im. These are the
F1 Global SR checkpoints.

The target-specific F2 experiment is documented in
[`FREQUENCY_MODULATION_F2.md`](FREQUENCY_MODULATION_F2.md). Re stopped after
its nine-branch onset screen because no branch passed the frozen regional
guardrails, so Re retains F1. Im selected split `k0_prime` envelopes plus the
L0 `lambda_prime` terminal and passed formal validation acceptance. Its new
test RMSE is `0.063032` with R2 `0.667266`, making Im F2 the current Im Global
SR checkpoint.

The controlled PySR runtime/seed benchmark is documented in
[`PYSR_ACCELERATION_BENCHMARK.md`](PYSR_ACCELERATION_BENCHMARK.md). The
8-thread C configuration failed exact same-seed replay, so subsequent formal
PySR experiments use plan B: serial deterministic search with batching
(`batch_size=4096`) and turbo. The MLP architecture remains unchanged.

The completed Re follow-up is documented in
[`F3_STAGED_PLAN.md`](F3_STAGED_PLAN.md). The accepted formal pair is Re F3
plus Im F2. Re F3 adds the frozen, formula-safe F3 residual to Re F1; it must
not be retrained or tuned against the consumed test partition.

## Formal Inference And Export

[`formal_models/active_model.json`](formal_models/active_model.json) is the
single formal model pointer. It selects the self-contained run1 Re F3 / Im F2
manifest. [`formal_global_symbolic_model.py`](formal_global_symbolic_model.py)
loads raw inputs in this exact order:

```text
phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f
```

It replays the stored transform and target-specific frequency terminals,
returns independent `R_real` and `R_imag` arrays, and combines them as
`Reflect = R_real + 1j*R_imag`. By default it fails on changed fixed inputs or
frequencies outside 100-4950 Hz. It never clips or smooths predictions.

Run file-based inference with:

```powershell
python formal_global_symbolic_model.py --input inputs.csv --output predictions.csv
```

Regenerate the immutable bundle from the accepted source artifacts only with:

```powershell
python export_formal_global_symbolic_model.py
```

The exporter verifies both acceptance decisions and the frozen Re candidate
hash. It performs no fitting or candidate selection. Formal paired evaluation
and exact Re F3-F2 replay are available through the commands below. The formal
evaluator also writes Re/Im teacher scatter, frequency-error, paired-curve, and
complex-output figures under its `figures/` directory.

```powershell
python evaluate_formal_global_symbolic_model.py --partition test --require-exact-regression
python -m unittest discover -s tests -p "test_*.py" -v
```

## Evaluation

Candidate selection is validation-only and independent for Re and Im.
Final test reporting uses raw RMSE, MAE, maximum absolute error, R2, ranges,
frequency error, scatter, and curve comparisons. Predictions are evaluated
raw; there is no `[0,1]` clipping.

## Planned Artifacts

```text
artifacts/re/train/20260809_run1/
artifacts/re/eval/20260809_run1_c21/
artifacts/im/train/20260809_run1/
artifacts/im/eval/20260809_run1_c21/
```

Keep names short; store detailed search parameters in training metadata.
Formal full-range baseline models exist for both targets. Sobol A/B branches
must use matched rows, split, seed, search budget, operators, and candidate
selection. The whole-range branch decision is validation-only; test data is
final reporting only. Follow `../current_project_handoff.md` for current
results and invariants.
