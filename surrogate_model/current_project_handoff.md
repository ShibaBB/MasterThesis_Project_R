# MasterThesis Project R — Current Handoff

Last updated: 2026-08-09.

## Purpose

This repository is the clean starting point for a new surrogate-learning task.
The three retained model families are:

1. MLP
2. segmented symbolic regression (PySR)
3. global symbolic regression (PySR)

Their input side remains unchanged:

```text
[phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime]
```

The new targets are two separate real-valued response curves:

```text
R_real(f) = real(Reflect(f))
R_imag(f) = imag(Reflect(f))
```

Every model family must train and evaluate one model for `R_real` and another
for `R_imag`. Absorption-coefficient prediction, clipping, reconstruction, and
evaluation are outside the scope of this repository.

## Current Repository State

The repository is located at:

```text
C:/MasterThesis_Project_R
```

GitHub:

```text
https://github.com/ShibaBB/MasterThesis_Project_R
```

The new repository has an independent root history. The generation and model
code now implements the paired `R_real` / `R_imag` contract. The active
dataset run is `run1`; its configuration and manifest exist, while its full
MAT datasets and formal model runs have not yet been generated.

The 51 tracked legacy artifact files were removed. A temporary 50-curve smoke
dataset verified exact component extraction, paired scalar expansion, shared
curve splitting, and independent one-epoch MLP Re/Im training; those temporary
outputs were removed after verification. PySR loaders and CLI contracts were
checked for both targets, but actual PySR searches still require installation
or recreation of the project Python/Julia environment.

## Critical JCAL Target Definition

Read `jcal_reflection.m` before changing any dataset code.

Its first output, `Reflect`, is the complex reflection coefficient required by
this project. The function's existing fourth and fifth outputs named `Re` and
`Im` are **not** `real(Reflect)` and `imag(Reflect)`: they are the normalized
real and imaginary components of surface impedance `Zs`.

Therefore teacher generation must use:

```matlab
[Reflect, ~, ~, ~, ~, ~, ~] = jcal_reflection(...);
R_real = real(Reflect);
R_imag = imag(Reflect);
```

Do not silently train on the existing fourth/fifth outputs. During the
implementation, rename or document those impedance outputs more clearly if
needed, but do not change the JCAL physics merely to adapt the surrogate.

## Dataset Strategy

### One paired teacher dataset per run

Generate the material parameters and call JCAL once per source curve. Store
both targets in the same teacher MAT file so they cannot drift apart:

```text
X         : [n_curves, 7]
Y_re      : [n_curves, n_frequency_points]
Y_im      : [n_curves, n_frequency_points]
freq_grid : [1, n_frequency_points]
dataset_info
sample_metadata
```

Recommended metadata:

```text
target_names = ["R_real", "R_imag"]
complex_source = "Reflect"
target_definition.R_real = "real(Reflect)"
target_definition.R_imag = "imag(Reflect)"
```

Do not generate Re and Im with independent LHS draws. They must share the same
`X`, frequency grid, source-curve indices, and teacher call.

### One shared curve split

Create exactly one `shared_curve_split.json` per dataset run. Both targets and
all three model families must use the same train/validation/test curve IDs.
Never split scalar frequency rows independently.

### First R dataset run

The old run directories were deleted and had different target semantics. The
first new dataset should therefore be `run1`, not a continuation of old run3.
During implementation:

1. set the central default to `run1`;
2. create `datasets/run1/dataset_config.json`;
3. retain the current material, sampling, frequency grid, split ratios, and
   model hyperparameters unless the user explicitly changes them;
4. record the target schema in the run config and manifest.

The intended initial frequency grid remains 100–4950 Hz with 128 points.

### Symbolic datasets

Expand the paired teacher curves into scalar rows once and retain both targets:

```text
X_symbolic        : [n_curves * n_frequency_points, 8]
y_re_symbolic     : [n_curves * n_frequency_points, 1]
y_im_symbolic     : [n_curves * n_frequency_points, 1]
source_curve_index
segment_index
freq_grid
symbolic_dataset_info
```

The eighth feature remains frequency. Segmented and global datasets may be
separate files as today, but each file should contain both target arrays.

## Training Strategy

### Common target selector

Add one explicit target selector to every generation/training/evaluation entry
point that needs it:

```text
target = re | im
```

Reject missing, unknown, or mismatched target metadata. Store `target`,
`dataset_run`, dataset path, and shared split hash in every training and
evaluation summary.

### MLP

Train two independent networks with the existing architecture and optimizer:

```text
X -> Y_re
X -> Y_im
```

Do not combine them into one 256-output network in the first migration. Keeping
two 128-output networks preserves the mature single-target training behavior
and makes errors attributable to one component.

The MLP loader must select `Y_re` or `Y_im`, then reuse the existing
standardization, shared curve split, training loop, and inverse transform.

### Segmented SR

For each target, train one PySR equation per configured frequency segment. With
eight segments this produces 16 equations in total: eight for `R_real`, eight
for `R_imag`. Keep the existing PySR search settings for the first full run.

Candidate selection remains validation-only and is performed independently for
each target and segment. Boundary diagnostics must also be reported separately
for Re and Im.

### Global SR

Train two global PySR equations over the full frequency domain:

```text
R_real = g_re(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
R_imag = g_im(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
```

Keep the existing PySR settings for the first run. Candidate selection remains
validation-only and independent for the two targets.

## Evaluation Contract

Do not clip either component. The previous physical clamp to `[0, 1]` must be
removed from both SR evaluators.

Report per target and per final test split:

- RMSE
- MAE
- maximum absolute error
- R2
- raw prediction minimum and maximum
- error versus frequency
- predicted-versus-teacher scatter
- random and worst-case curve comparisons

After both component models are available, an optional paired diagnostic may
report complex reflection error:

```text
R_error = (R_real_pred - R_real_true) + i*(R_imag_pred - R_imag_true)
complex_RMSE = sqrt(mean(abs(R_error).^2))
```

This diagnostic must use predictions from the same model family, same dataset
run, same split hash, and same source rows. Do not derive or evaluate any other
target from the predicted components.

## Artifact Layout And Short Naming

Keep target names in short parent directories instead of repeating them in
every long filename:

```text
MLP/artifacts/re/20260809_run1/
MLP/artifacts/im/20260809_run1/

segmented_symbolic_regression/artifacts/re/train/20260809_run1/
segmented_symbolic_regression/artifacts/re/eval/20260809_run1_c21/
segmented_symbolic_regression/artifacts/im/train/20260809_run1/
segmented_symbolic_regression/artifacts/im/eval/20260809_run1_c21/

global_symbolic_regression/artifacts/re/train/20260809_run1/
global_symbolic_regression/artifacts/re/eval/20260809_run1_c21/
global_symbolic_regression/artifacts/im/train/20260809_run1/
global_symbolic_regression/artifacts/im/eval/20260809_run1_c21/
```

Rules:

- use `YYYYMMDD_runN` for normal training directories;
- append only necessary selectors such as `_c21` or `_limited`;
- store full hyperparameters in JSON metadata, not path names;
- validate the predicted PySR internal path length before launching;
- never recreate descriptive run names containing the complete parameter list.

## Implemented Migration And Next Execution

Items 1-8 are implemented. Item 9 passed for teacher generation, scalar
expansion, shared splitting, and both MLP targets; PySR execution remains to
be verified after its environment is recreated. The next execution steps are:

1. recreate/install the PySR Python and Julia environment;
2. run `run_active_dataset_generation.ps1` to generate the full paired `run1`
   teacher data, shared split, and both symbolic MAT files;
3. synchronize and inspect the manifest;
4. run full MLP Re and Im training;
5. run segmented and global PySR searches independently for `--target re` and
   `--target im`;
6. evaluate validation-selected candidates on the final shared test curves and
   update this handoff with measured results.

## Files Expected To Change

At minimum inspect and update:

```text
jcal_reflection.m                         (target naming/documentation only if needed)
surrogate_model/dataset_run_config.json
surrogate_model/dataset_run_config.py
surrogate_model/resolve_dataset_run_config.m
surrogate_model/create_dataset_run.py
surrogate_model/sync_dataset_manifest.py
surrogate_model/generate_shared_curve_split.py
surrogate_model/data_generation/generate_teacher_dataset.m
surrogate_model/data_generation/inspect_teacher_dataset.m
surrogate_model/MLP/mlp_train_surrogate_baseline.m
surrogate_model/segmented_symbolic_regression/generate_segmented_symbolic_dataset.m
surrogate_model/segmented_symbolic_regression/inspect_segmented_symbolic_dataset.m
surrogate_model/segmented_symbolic_regression/segmented_run_paths.py
surrogate_model/segmented_symbolic_regression/train_segmented_symbolic_models.py
surrogate_model/segmented_symbolic_regression/evaluate_segmented_symbolic_candidates.py
surrogate_model/global_symbolic_regression/global_run_paths.py
surrogate_model/global_symbolic_regression/train_global_symbolic_model.py
surrogate_model/global_symbolic_regression/evaluate_global_symbolic_candidates.py
```

## Acceptance Criteria Before Full Training

- No surrogate dataset or active model path uses the old target variable.
- `R_real` and `R_imag` numerically equal `real(Reflect)` and `imag(Reflect)` on
  a small checked sample.
- Both arrays are finite and have identical curve/frequency dimensions.
- One shared curve split is used by every target and model family.
- MLP, segmented SR, and global SR can each complete a small Re run and a small
  Im run without overwriting one another.
- Evaluation is raw and unclipped.
- Artifact paths follow the short naming policy.
- Training metadata makes target/run/split mismatches fail loudly.

## Non-Goals For The First Migration

- changing the JCAL equations;
- changing the seven input features;
- tuning MLP or PySR hyperparameters;
- training one joint multi-target MLP;
- coupling Re and Im candidate selection;
- recreating legacy datasets or artifacts;
- deriving or evaluating absorption coefficient.
