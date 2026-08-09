# MasterThesis Project R - Current Handoff

Last updated: 2026-08-09.

## Project Purpose

This repository builds surrogate models for the complex acoustic reflection
coefficient produced by the JCAL teacher model. The learning targets are two
independent real-valued frequency-response curves:

```text
R_real(f) = real(Reflect(f))
R_imag(f) = imag(Reflect(f))
```

The repository contains three surrogate families:

1. MLP
2. segmented symbolic regression with PySR
3. global symbolic regression with PySR

Every family trains and evaluates `R_real` and `R_imag` separately while using
the same material samples, frequency grid, source-curve split, and target
definition. No model output is clipped. Absorption prediction and impedance
component prediction are outside the project scope.

## Repository And Git State

```text
Repository: C:/MasterThesis_Project_R
GitHub:     https://github.com/ShibaBB/MasterThesis_Project_R
```

The repository history contains the paired-target implementation, the full
`run1` datasets, formal PySR artifacts, dynamic evaluation-axis fix, shared
symbolic feature transform, and the training/evaluation updates described in
this handoff. Use `git log -1` and `git status` to verify the exact checkout.

The previously tracked obsolete artifact files were removed. New artifacts are
generated independently under short `re/` and `im/` directories.

## Target Definition

`jcal_reflection.m` returns the complex reflection coefficient as its first
output. Teacher generation uses that output directly:

```matlab
[Reflect, ~, ~, ~, ~, ~, ~] = jcal_reflection(...);
R_real = real(Reflect);
R_imag = imag(Reflect);
```

The fourth and fifth outputs of `jcal_reflection.m` are normalized surface
impedance components and must not be used as the surrogate targets.

## Active Dataset Run

The active dataset is `run1`.

```text
Teacher curves:          1000
Material inputs:         7 per curve
Frequency range:         100-4950 Hz
Frequency points:        128
Teacher X shape:         [1000, 7]
Teacher Y_re shape:      [1000, 128]
Teacher Y_im shape:      [1000, 128]
Symbolic X shape:        [128000, 8]
Symbolic target shape:   [128000, 1] for each target
Curve split:             700 train / 150 validation / 150 test
Split hash:              cdd4e1c726bf5f980a911bc50ce3fd37d7e9023df3adee6f7de81e12a365d5f0
```

Canonical files:

```text
surrogate_model/datasets/run1/dataset_config.json
surrogate_model/datasets/run1/dataset_manifest.json
surrogate_model/datasets/run1/shared_curve_split.json
surrogate_model/datasets/run1/MLP/Wool_R.mat
surrogate_model/datasets/run1/global_SR/Wool_R_global.mat
surrogate_model/datasets/run1/segmented_SR/Wool_R_segmented.mat
```

Both symbolic MAT files contain `y_re_symbolic` and `y_im_symbolic` aligned to
the same scalar rows, curve IDs, frequency values, and segment IDs. Splitting
is always performed by source curve, never by scalar frequency row.

## Symbolic Feature Processing

The current global and segmented Python trainers share
`surrogate_model/symbolic_feature_transform.py`.

The transform is fitted on the training partition only:

1. read the recommended base-10-log feature indices from dataset metadata;
2. apply `log10` to the five positive scale-sensitive inputs, including
   frequency;
3. detect and remove columns that are exactly constant in the training split;
4. store the complete transform specification in `training_metadata.json`;
5. require evaluators to reuse that stored transform without refitting it.

For `run1`, two fixed material columns are removed and the PySR input width is
reduced from eight to six. Transformed variables have explicit names such as
`log10_f`, so exported equations cannot be mistaken for formulas operating on
raw values. Evaluators retain a separate physical-frequency array for plots
and boundary diagnostics.

Older training runs without transform metadata remain evaluable through the
identity-transform compatibility path.

## Current Progress Summary

| Area | Re | Im | Current state |
|---|---|---|---|
| Paired teacher dataset | Complete | Complete | Full `run1` generated and inspected |
| Shared curve split | Complete | Complete | Same split hash used by all families |
| Scalar global dataset | Complete | Complete | Paired targets in one MAT file |
| Scalar segmented dataset | Complete | Complete | Paired targets and eight segment IDs |
| MLP code contract | Complete | Complete | Independent target loading and smoke training verified |
| Full MLP training | Pending | Pending | No formal full `run1` result yet |
| Global SR full run | Baseline available | Complete | Re still needs a run with the current feature transform |
| Segmented SR full run | Complete | Complete | Latest formal run uses 100 iterations per segment |
| Validation-only candidate selection | Complete | Complete | Full validation rows used before final test evaluation |
| Raw unclipped evaluation | Complete | Complete | Dynamic plot limits support negative values |

## Latest Global SR Results

Global SR trains one equation over the complete 100-4950 Hz range.

### Re baseline

The available Re result predates the shared log-scale feature transform. It is
valid as a baseline but is not configuration-matched to the latest Im run.

```text
Training: surrogate_model/global_symbolic_regression/artifacts/re/train/20260809_run1
Eval:     surrogate_model/global_symbolic_regression/artifacts/re/eval/20260809_run1_best_loss_02

Test RMSE:          0.089386
Test MAE:           0.059882
Test max abs error: 0.407179
Test R2:            0.773368
```

### Im with current feature transform

```text
Training: surrogate_model/global_symbolic_regression/artifacts/im/train/20260809_run1_log10_i100_p12
Eval:     surrogate_model/global_symbolic_regression/artifacts/im/eval/20260809_run1_log10_i100_p12_best_loss

Iterations:         100
Populations:        12
Population size:    80
Max complexity:     24
Train rows:         89600
Validation rows:    19200
Test rows:          19200
Test RMSE:          0.079741
Test MAE:           0.057977
Test max abs error: 0.350998
Test R2:            0.467469
```

The Im global equation now captures the main low-frequency dip and recovery,
but still underfits material-dependent trough depth, peak position, and the
high-frequency downturn. A simple per-frequency training-mean baseline reaches
test RMSE `0.077103` and R2 `0.502130`, so the global symbolic model remains
below that baseline.

## Latest Segmented SR Results

The configured frequency domains are:

```text
100-700, 700-1000, 1000-1300, 1300-1650,
1650-2000, 2000-3000, 3000-4000, 4000-4950 Hz
```

Each target has eight independent PySR equations. The latest formal runs use:

```text
Iterations per segment: 100
Populations:            12
Population size:        80
Max complexity:         24
Training data:          all rows in each segment's train curves
Candidate selection:    full validation partition, independently per segment
Final reporting:        full test partition after candidate selection
Output postprocessing:  none
```

### Re

```text
Training: surrogate_model/segmented_symbolic_regression/artifacts/re/train/20260809_run1_log10_i100_p12
Eval:     surrogate_model/segmented_symbolic_regression/artifacts/re/eval/20260809_run1_log10_i100_p12_best_loss

Validation RMSE:    0.035596
Validation R2:      0.964944
Test RMSE:          0.034924
Test MAE:           0.023899
Test max abs error: 0.267284
Test R2:            0.965403
```

### Im

```text
Training: surrogate_model/segmented_symbolic_regression/artifacts/im/train/20260809_run1_log10_i100_p12
Eval:     surrogate_model/segmented_symbolic_regression/artifacts/im/eval/20260809_run1_log10_i100_p12_best_loss

Validation RMSE:    0.031412
Validation R2:      0.926237
Test RMSE:          0.032371
Test MAE:           0.020815
Test max abs error: 0.286863
Test R2:            0.912239
```

Increasing Im from 60 to 100 iterations improved test RMSE from `0.035293` to
`0.032371` and test R2 from `0.895681` to `0.912239`. It improved within-segment
fit but did not solve boundary discontinuities.

## Known Segmented Boundary Problem

The segmented models are accurate within most frequency domains, but the eight
equations are trained independently and have no continuity constraint.

For the latest test results:

```text
Re largest mean absolute predicted boundary change: 0.077895 at 3000 Hz
Re largest maximum excess boundary change:          0.375519
Im largest mean absolute predicted boundary change: 0.055364 at 3000 Hz
Im largest maximum excess boundary change:          0.404880
```

Teacher changes across the same adjacent samples are much smaller. Increasing
iterations improves segment-local regression but is not a direct solution to
cross-segment continuity. Do not add smoothing silently: any overlap, blending,
continuity penalty, or joint boundary-selection rule must be explicit and must
be evaluated against the unmodified test targets.

Boundary diagnostics are stored in each evaluation directory as
`boundary_transition_metrics.csv`.

## Evaluation Contract

Every formal evaluation must:

- select candidates using validation data only;
- report final metrics once on the shared test curves;
- keep Re and Im selection independent;
- report RMSE, MAE, maximum absolute error, R2, and prediction range;
- generate predicted-versus-teacher scatter and error-versus-frequency plots;
- generate full-curve comparisons with dynamic combined teacher/prediction
  vertical limits;
- preserve raw predictions without clipping or smoothing;
- record dataset run, target, dataset path, split file, and split hash.

An optional paired diagnostic may combine predictions from the same model
family, dataset run, split, and source rows into a complex reflection error.
This paired diagnostic has not yet been implemented as a formal pipeline step.

## Runtime Environment

The active PySR environment is available at:

```text
surrogate_model/segmented_symbolic_regression/.venv_py311
```

Verified components:

```text
Python 3.11 environment
PySR 1.5.10
Julia 1.11.9
SymbolicRegression 1.11.3
```

The Python trainers currently use deterministic serial PySR execution. PySR
warns for segments with more than 10,000 rows and recommends batching. The
formal runs intentionally used all training rows; batching remains a possible
future search-speed experiment, not part of the reported results.

## Artifact Layout

Use short target parent directories and keep detailed parameters in metadata:

```text
surrogate_model/MLP/artifacts/re/<run>/
surrogate_model/MLP/artifacts/im/<run>/

surrogate_model/global_symbolic_regression/artifacts/re/train/<run>/
surrogate_model/global_symbolic_regression/artifacts/re/eval/<run>/
surrogate_model/global_symbolic_regression/artifacts/im/train/<run>/
surrogate_model/global_symbolic_regression/artifacts/im/eval/<run>/

surrogate_model/segmented_symbolic_regression/artifacts/re/train/<run>/
surrogate_model/segmented_symbolic_regression/artifacts/re/eval/<run>/
surrogate_model/segmented_symbolic_regression/artifacts/im/train/<run>/
surrogate_model/segmented_symbolic_regression/artifacts/im/eval/<run>/
```

Do not infer target, split, or transform semantics from directory names alone;
validate `training_metadata.json` and the evaluation summary.

## Next Priorities

1. Run full `run1` MLP training and evaluation for both Re and Im.
2. Retrain global Re with the current feature transform and configuration so
   it is directly comparable to global Im.
3. Decide on an explicit segmented-boundary strategy: overlapping domains with
   blending, continuity-aware candidate selection, or a continuity penalty.
4. Add a paired complex-reflection diagnostic after both targets from a model
   family are available on identical test rows.
5. Consider batching or a curve/frequency-balanced search subset only as a
   controlled PySR speed experiment; always select on full validation data and
   report on full test data.

## Invariants For Future Work

- The targets are always the real and imaginary parts of `Reflect`.
- Re and Im share one teacher call, material matrix, frequency grid, and curve
  split.
- Scalar symbolic rows retain their source-curve IDs.
- No evaluator splits scalar frequency rows independently.
- No target output is clipped.
- Feature transforms are fitted on training data and persisted in metadata.
- Evaluation reuses the training transform exactly.
- Test data never participates in candidate selection.
- Re and Im artifacts never overwrite one another.
- Target, dataset-run, or split-hash mismatches fail loudly.
