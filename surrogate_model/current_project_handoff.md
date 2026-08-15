# MasterThesis Project R - Current Handoff

Last updated: 2026-08-14.

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
`run1` datasets, formal MLP and PySR artifacts, dynamic evaluation-axis fix,
shared symbolic feature transform, and the training/evaluation updates
described in this handoff. Use `git log -1` and `git status` to verify the exact
checkout.

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
| Full MLP training | Complete | Complete | Formal full `run1` training and evaluation completed |
| Global SR full run | Complete | Complete | Both targets now use the current feature transform and matched search settings |
| Segmented SR full run | Complete | Complete | Latest formal run uses 100 iterations per segment |
| MLP Sobol pilot | Complete | Complete | Frequency-resolved S1/ST at N=4096 |
| Teacher Sobol validation | Complete | Complete | Shared-design N=1024 check confirms MLP sensitivity structure |
| Segmented SR Sobol feature experiment | Complete | Complete | Matched all-feature/subset runs and validation-only hybrid selection completed |
| Segmented SR frequency F1 | Complete | Complete | F0 reused; local frequency features evaluated on validation/test and shape diagnostics |
| Segmented SR curve-aware F1 reselection | Complete | Complete | Existing Hall-of-Fame candidates gated against F0; no retraining |
| Validation-only SR candidate selection | Complete | Complete | Full validation rows used before final test evaluation |
| Raw unclipped evaluation | Complete | Complete | Dynamic plot limits support negative values |

All three model families now run end to end for both targets. The MLP is the
clear accuracy baseline. Both symbolic-regression families remain substantially
less accurate and are the main focus of the next optimization phase.

## Latest MLP Results

The MLP branch trains one independent 128-output network per target. Both
formal runs used the shared `run1` curve split, training-only preprocessing,
Adam, a `[128, 128, 64]` hidden-layer layout, a maximum of 400 epochs, and
validation patience of 25. Test data was used only for final metrics and plots.

### Re

```text
Artifacts: surrogate_model/MLP/artifacts/re/20260810_run1

Output network iteration: 2370 (237 epochs)
Validation RMSE:         0.003727
Validation R2:           0.998446
Test RMSE:               0.003426
Test MAE:                0.002314
Test max abs error:      0.032793
Test R2:                 0.998623
Prediction range:        [-0.173306, 0.985038]
```

### Im

```text
Artifacts: surrogate_model/MLP/artifacts/im/20260810_run1

Output network iteration: 2810 (281 epochs)
Validation RMSE:         0.002899
Validation R2:           0.998678
Test RMSE:               0.002914
Test MAE:                0.001914
Test max abs error:      0.031752
Test R2:                 0.998547
Prediction range:        [-0.643229, 0.205147]
```

The MLP predictions cover the negative target regions without clipping and
generalize closely from validation to test. They provide the current reference
accuracy for evaluating future symbolic-regression improvements.

## Latest Global SR Results

Global SR trains one equation over the complete 100-4950 Hz range.

### Re with current feature transform

```text
Training: surrogate_model/global_symbolic_regression/artifacts/re/train/20260810_run1_log10_i100_p12
Eval:     surrogate_model/global_symbolic_regression/artifacts/re/eval/20260810_run1_log10_i100_p12_best_loss

Iterations:         100
Populations:        12
Population size:    80
Max complexity:     24
Train rows:         89600
Validation rows:    19200
Test rows:          19200
Validation RMSE:    0.081930
Validation R2:      0.814286
Test RMSE:          0.080918
Test MAE:           0.062006
Test max abs error: 0.329674
Test R2:            0.814274
Prediction range:   [0.050088, 1.255914]
```

The prior identity-transform Re baseline had test RMSE `0.089386` and R2
`0.773368`. The current transform and larger matched search improve RMSE by
about 9.5 percent and reduce maximum error, but the selected equation still
misses the negative Re region and overshoots above one. Predictions remain raw
and unclipped by design.

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

## Segmented SR Sobol Phase-One Results

The matched phase-one experiment completed on 2026-08-14. For each target,
the `all` and `sobol_subset` branches used the same run1 rows, split, random
seed 42, 100 iterations, 12 populations, population size 80, max complexity
24, operator set, and validation-only candidate selection. The training stage
did not access test rows.

The full Sobol-subset branch was not superior as a single replacement:

```text
Target/branch       Validation RMSE   Test RMSE   Test R2
Re all              0.035596          0.034924    0.965403
Re Sobol subset     0.038239          0.039506    0.955730
Im all              0.031412          0.032371    0.912239
Im Sobol subset     0.031752          0.032854    0.909605
```

Validation-only per-segment selection retained the Sobol subset only where it
won its matched comparison:

```text
Re: 100-700 Hz uses sobol_subset; the other seven segments use all.
Im: 1300-1650 and 4000-4950 Hz use sobol_subset; the other six use all.
```

The resulting validation-selected hybrid metrics are:

```text
Target   Validation RMSE   Validation R2   Test RMSE   Test R2
Re       0.035577          0.964981        0.034823    0.965604
Im       0.030429          0.930782        0.030686    0.921142
```

Relative to the matched all-feature baseline, the hybrid changes Re test RMSE
by only about -0.29 percent, but improves Im test RMSE by about 5.21 percent.
This supports local validation-based feature restriction rather than replacing
every segment with its Sobol subset. The phase-one feature decision remains
separate from the later continuity/blending experiment.

Formal artifacts:

```text
surrogate_model/segmented_symbolic_regression/artifacts/re/comparison/20260814_run1_sobol_phase1
surrogate_model/segmented_symbolic_regression/artifacts/im/comparison/20260814_run1_sobol_phase1
```

## Segmented SR Frequency-Representation F1 Results

The F1 experiment completed on 2026-08-15 without retraining F0. It froze the
phase-one validation-selected material features, retained `log10_f`, and added
segment-local `local_t`, `local_t2`, and `local_t3`. Frequency use was optional
and candidate selection remained validation RMSE only. Shape metrics were
diagnostics and did not influence candidate selection.

```text
Target/model   Validation RMSE   Test RMSE   Test R2
Re F0 hybrid   0.035577          0.034823    0.965604
Re F1          0.032830          0.032171    0.970643
Im F0 hybrid   0.030429          0.030686    0.921142
Im F1          0.030213          0.031974    0.914382
```

Re F1 improves validation RMSE by 7.72 percent and test RMSE by 7.61 percent.
Its first-difference RMSE improves in all eight validation segments, although
second-difference error worsens in six, so it is not yet accepted as the final
shape model. Seven of eight selected Re equations use some frequency feature;
the 1650-2000 Hz equation still ignores frequency.

Im F1 improves validation RMSE by only 0.71 percent but worsens test RMSE by
4.20 percent. Several Im formulas use local polynomial terminals inside
division or logarithm expressions, producing excessive curvature. Therefore
the complete Im F1 replacement is rejected on generalization evidence, and no
shape-acceptance decision has been made.

A validation-RMSE-only F0/F1 per-segment diagnostic would select six F1 Re
segments and three F1 Im segments. Its test RMSE is `0.032079` for Re and
`0.030103` for Im. This diagnostic is not the final model because shape has not
yet been included in the acceptance rule.

Formal artifacts:

```text
surrogate_model/segmented_symbolic_regression/artifacts/re/comparison/20260815_run1_frequency_f1
surrogate_model/segmented_symbolic_regression/artifacts/im/comparison/20260815_run1_frequency_f1
surrogate_model/segmented_symbolic_regression/artifacts/frequency_f1_manifests/20260815_120054_run1_frequency_f1.json
```

### Curve-Aware F1 Hall-of-Fame Reselection

On 2026-08-15, every existing F1 Hall-of-Fame candidate was rescored on full
validation curves. F0 remained the segment fallback. An F1 candidate had to
pass gates on overall and per-curve RMSE, first and second differences,
prediction bounds, mean curve range, and excess turning points. No model was
retrained, and choices were frozen before test evaluation.

```text
Target/model          Validation RMSE   Test RMSE   Test R2
Re F0 hybrid          0.035577          0.034823    0.965604
Re curve-aware hybrid 0.034318          0.033009    0.969094
Im F0 hybrid          0.030429          0.030686    0.921142
Im curve-aware hybrid 0.027730          0.030104    0.924101
```

Re retained F1 only at 100-700, 1300-1650, 3000-4000, and 4000-4950 Hz.
Im retained F1 only at 700-1000 and 3000-4000 Hz. All other segments reverted
to their phase-one F0 winner. Among the retained Re segments, three improve
second-difference error; 3000-4000 Hz worsens it by 7.4 percent, within the
declared 10 percent gate. Both retained Im segments improve value, first-
difference, and second-difference errors. Cross-segment boundary discontinuity
is unchanged as a separate unresolved problem.

```text
surrogate_model/segmented_symbolic_regression/artifacts/re/comparison/20260815_run1_frequency_f1_curve_aware
surrogate_model/segmented_symbolic_regression/artifacts/im/comparison/20260815_run1_frequency_f1_curve_aware
```

### Segmented SR Pause Decision

Segmented SR optimization is paused after the curve-aware F0/F1 reselection.
This is a deliberate project-scope decision, not a claim that the segmented
model is finished. Each of the eight frequency domains still needs individual
formula/grammar tuning, and any final segmented model would also require a
separate validation-only treatment of the seven frequency-boundary handoffs.
Those two work streams are deferred because their cost is high and neither is
a direct substitute for improving the single-equation Global SR model.

The curve-aware hybrid above is the current segmented checkpoint to preserve.
Do not resume segmented training implicitly. If this branch is revisited,
start from its persisted F0/F1 candidates and declared shape gates, then treat
within-segment fitting and cross-segment continuity as separate experiments.
The active optimization target now moves to Sobol-guided Global SR.

## Current Cross-Family Comparison

```text
Family                    Re test RMSE   Re test R2   Im test RMSE   Im test R2
MLP                       0.003426       0.998623     0.002914       0.998547
Segmented SR baseline     0.034924       0.965403     0.032371       0.912239
Segmented curve-aware F1  0.033009       0.969094     0.030104       0.924101
Global SR                 0.080918       0.814274     0.079741       0.467469
```

The MLP RMSE is roughly one order of magnitude lower than segmented SR for both
targets. Segmentation is substantially better than one global equation, but it
still trails the MLP and introduces boundary discontinuities. Global SR is the
weakest family, especially for Im, and requires improvements beyond simply
increasing the number of search iterations.

## Frequency-Resolved Sobol Sensitivity

Sensitivity-analysis code and artifacts are isolated under
`surrogate_model/sobol`; no Sobol output is written into any model-family
directory. The analysis fixes the two constant `run1` inputs (`phi=0.92` and
`h=0.03 m`) and varies the five sampled material parameters independently over
the same raw linear-uniform bounds used by teacher generation. Frequency is an
output axis, not an uncertain material parameter.

The formal MLP pilot uses nested base sample sizes `1024`, `2048`, and `4096`,
estimates Jansen first-order and total-effect indices at all 128 frequencies,
and reports both full-range and eight-segment variance-weighted functional
indices:

```text
Code:      surrogate_model/sobol/run_mlp_sobol_pilot.m
Artifacts: surrogate_model/sobol/artifacts/mlp/20260810_run1_n4096
```

A JCAL teacher run validates the first `N=1024` points of the identical
scrambled Sobol design, requiring 7,168 teacher material-curve evaluations:

```text
Code:      surrogate_model/sobol/run_teacher_sobol_validation.m
Artifacts: surrogate_model/sobol/artifacts/teacher/20260810_run1_n1024

Re MLP-teacher S1 RMSE:                0.004840
Re MLP-teacher ST RMSE:                0.007870
Re flattened ST correlation:           0.999645
Re top-ST parameter agreement:         100% of frequency points
Im MLP-teacher S1 RMSE:                0.003436
Im MLP-teacher ST RMSE:                0.003490
Im flattened ST correlation:           0.999947
Im top-ST parameter agreement:         100% of frequency points
```

Teacher variance-weighted full-range total-effect indices are:

```text
Parameter          Re ST      Im ST
sigma              0.849552   0.840944
alpha_infinity     0.171057   0.215920
lambda             0.078213   0.096305
k0_prime           0.032073   0.025709
lambda_prime       0.006380   0.004819
```

The global ordering hides strong local structure. For Re, `k0_prime` reaches
ST `0.710769` near 482 Hz, `sigma` dominates most middle frequencies, and
`alpha_infinity` reaches ST `0.722313` near 3422 Hz. For Im, `k0_prime` reaches
ST `0.435505` at 100 Hz; high-frequency `alpha_infinity` and `lambda` effects
are largely interaction-driven. The MLP pilot is reliable for screening, but
teacher values remain authoritative, especially for Re `k0_prime` above 4 kHz
where the emulator-index discrepancy is largest.

Neither the current SR equations nor their omitted variables were used to
define these sensitivities. Sobol results describe the teacher input-output
mapping and should guide, not replace, validation-based SR model selection.

## Sobol-Guided SR Optimization Plan

The next optimization phase should use one shared principle: Sobol indices set
feature and interaction priorities, but they do not select the final equation.
Every feature-set decision must be tested against a matched all-parameter
control with the same training rows, search budget, random seed, validation
partition, and raw target. Frequency remains available to every SR model.

### Segmented SR Strategy

Segmented SR is the most direct consumer of the frequency-resolved sensitivity
results because each local equation can use a different material feature set.
Use the following priorities for both feature-subset experiments and operator
design:

| Frequency segment (Hz) | Re parameter priority | Im parameter priority |
|---|---|---|
| 100-700 | primary: `sigma`, `k0_prime`; secondary: `alpha_infinity` | primary: `sigma`; secondary: `k0_prime` |
| 700-1000 | primary: `sigma`, `k0_prime`; secondary: `alpha_infinity`, `lambda` | primary: `sigma`; secondary: `alpha_infinity`, `lambda` |
| 1000-1300 | primary: `sigma`; secondary: `k0_prime`, `lambda`, `alpha_infinity` | primary: `sigma`, `alpha_infinity`; secondary: `lambda` |
| 1300-1650 | primary: `sigma`; secondary: `lambda`, `alpha_infinity` | primary: `alpha_infinity`, `sigma`; secondary: `lambda` |
| 1650-2000 | primary: `sigma`, `lambda`; secondary: `alpha_infinity` | primary: `sigma`, `alpha_infinity`; secondary: `lambda` |
| 2000-3000 | primary: `sigma`, `alpha_infinity`; secondary: `lambda` | primary: `sigma`, `alpha_infinity`; secondary: `lambda` |
| 3000-4000 | primary: `alpha_infinity`, `sigma`; secondary: `lambda`, `k0_prime` | primary: `sigma`; interaction priority: `alpha_infinity`, `lambda` |
| 4000-4950 | primary: `sigma`, `alpha_infinity`, `k0_prime`, `lambda`; diagnostic: `lambda_prime` | primary: `sigma`, `alpha_infinity`, `lambda`; secondary: `k0_prime`; diagnostic: `lambda_prime` |

For every target and segment, run at least these two matched branches:

```text
A. all five varying material parameters plus frequency
B. Sobol-prioritized subset plus frequency
```

Do not remove a parameter from the shared dataset. Branch B only limits the
features exposed to that particular PySR search. In bands where `ST-S1` is
large, retain multiplication/division and nonlinear operators so PySR can
represent interactions even when a parameter has a small first-order index.
`lambda_prime` has the lowest global ST, but its removal is accepted only when
the validation result is non-inferior; its high-frequency local effect remains
a diagnostic.

Use two optimization phases:

1. improve within-segment validation accuracy with the current fixed domains;
2. address continuity through explicit overlapping domains and smooth
   blending, or through continuity-aware joint candidate selection.

For phase 2, never choose adjacent formulas independently if the selection
objective includes continuity. Define the validation-only joint objective in
metadata, report its accuracy and boundary terms separately, and preserve the
unblended predictions as a diagnostic. Test data must not choose overlap width,
blend shape, boundary penalty, or candidate combination.

### Global SR Strategy: Smooth Frequency-Local Features

The selected direction for Global SR is to retain one full-range equation but
add explicit smooth frequency-local features. The goal is to let a material
parameter have a strong effect in one frequency region and a negligible effect
elsewhere without requiring PySR to discover the entire gating function from
primitive operators.

Start from the current six transformed base features and define fixed smooth
overlapping gates in `log10_f`, for example normalized Gaussian or raised-cosine
basis functions:

```text
g_k(f) >= 0
sum_k g_k(f) = 1
g_k is smooth across adjacent frequency regions
```

Use the existing eight domains as the first gate centers/supports, but make the
gates overlap so this is a single continuous global representation rather than
hard piecewise selection. Create only Sobol-supported material/gate products:

```text
z_parameter,k = transformed_parameter * g_k(log10_f)
```

Examples include low-frequency `log10_k0_prime * g_low`, mid/high-frequency
`alpha_infinity * g_k`, and high-frequency `log10_lambda * g_k`. Always retain
the ungated base features and `log10_f`, so the model can still learn a compact
global trend. Do not use the numerical Sobol ST curve itself as a feature; use
Sobol only to decide which generic parameter/gate products to expose.

Run the following matched Global experiments in order:

```text
G0. current transformed six-feature Global SR baseline
G1. base features plus Sobol-supported smooth gated interactions
G2. validation ablations of individual gate families that were selected in G1
```

Keep the G0/G1 search budget and random seed matched for the first comparison.
Only increase iterations, populations, or complexity after isolating the effect
of the local features. Candidate selection remains validation-only. In addition
to overall validation RMSE, report validation RMSE per current frequency domain
so a global improvement cannot hide a severe local regression.

Engineered gates make PySR's internal complexity count optimistic because each
gate appears as one input variable. Export every gate definition and report
both the PySR expression complexity and an expanded complexity that includes
the gate construction. Final equations must be reproducible from raw physical
inputs, the persisted base transform, and the persisted gate specification.

The smooth-gated Global model remains a distinct single-equation experiment.
If it evolves into separately fitted local equations or independently selected
gate experts, classify it as a soft-segmented model instead of replacing the
pure Global SR result silently.

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

- use validation data only for model decisions: early stopping for MLP and
  candidate selection for symbolic regression;
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
MATLAB R2025b with Deep Learning Toolbox
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

1. Implement the Global SR smooth frequency-gate transform and run G0 versus
   G1 with matched search settings. Persist gate definitions and expanded
   complexity metadata; do not use the teacher ST curves themselves as inputs.
2. Compare Global G0/G1 on overall validation RMSE and validation RMSE in each
   of the existing eight frequency domains. Keep candidate selection validation
   only and evaluate the frozen winner once on test data.
3. If G1 selects gated feature families, run validation ablations before
   increasing PySR iterations, populations, or expression complexity.
4. Preserve the current MLP results as the accuracy reference while optimizing
   interpretable symbolic models. Any SR improvement must be reported on the
   same shared split and raw test targets.
5. Add a paired complex-reflection diagnostic after both targets from a model
   family are available on identical test rows.
6. Keep further segmented work deferred. If explicitly resumed, separately
   address constrained within-segment frequency grammar and validation-only
   boundary continuity/blending.
7. Consider batching or a curve/frequency-balanced search subset only as a
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
