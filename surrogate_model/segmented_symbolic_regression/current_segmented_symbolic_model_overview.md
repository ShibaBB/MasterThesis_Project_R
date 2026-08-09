# Current Symbolic Model Overview

## Purpose

This branch builds an interpretable surrogate model for the JCAL absorption model.

The target mapping is:

```text
[phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f] -> alpha
```

Compared with the MLP branch, this model does not predict a full absorption curve in one shot. Instead, it expands each curve into scalar frequency samples and uses PySR to search for explicit symbolic formulas.

## Current Dataset

The current teacher dataset is:

```text
surrogate_model/datasets/run2/MLP/Wool_surrogate_dataset.mat
```

It contains:

```text
1000 curve samples
64 frequency points per curve
```

The symbolic dataset generated from it is:

```text
surrogate_model/datasets/run2/segmented_SR/Wool_symbolic_segmented.mat
```

It contains:

```text
1000 * 64 = 64000 scalar samples
```

Each scalar sample has:

```text
X_symbolic = [phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f]
y_symbolic = alpha
```

## Frequency Segmentation

The symbolic model is trained segment-wise. The current default segments are:

```text
Segment 1: low_100_700        = 100-700 Hz
Segment 2: midlow_700_1000    = 700-1000 Hz
Segment 3: midhigh_1000_1300  = 1000-1300 Hz
Segment 4: highlow_1300_1650  = 1300-1650 Hz
Segment 5: high_1650_2000     = 1650-2000 Hz
```

Current full symbolic dataset segment counts:

```text
low_100_700        20000 samples
midlow_700_1000    10000 samples
midhigh_1000_1300  10000 samples
highlow_1300_1650  12000 samples
high_1650_2000     12000 samples
```

The intended model structure is therefore:

```text
alpha_low      = g1(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
alpha_midlow   = g2(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
alpha_midhigh  = g3(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
alpha_highlow  = g4(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
alpha_high     = g5(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
```

## Current Scripts

All active entry points read the default dataset run from:

```text
surrogate_model/dataset_run_config.json
```

The current default is `run2`. Change that one value to switch all MATLAB and
Python segmented-SR entry points together. Python training and evaluation also
accept `--dataset-run runN`. Standard `datasets/run*/...` paths are checked
against the selected run, and evaluation rejects training artifacts whose
`training_metadata.json` records a different dataset run.

Generate the full symbolic dataset:

```text
surrogate_model/segmented_symbolic_regression/run_full_segmented_symbolic_dataset_generation.m
```

Inspect the full symbolic dataset:

```text
surrogate_model/segmented_symbolic_regression/run_full_segmented_symbolic_dataset_inspection.m
```

Train segment-wise PySR models:

```text
surrogate_model/segmented_symbolic_regression/train_segmented_symbolic_models.py
```

Evaluate candidate formulas and generate plots:

```text
surrogate_model/segmented_symbolic_regression/evaluate_segmented_symbolic_candidates.py
```

Evaluation applies `clip(alpha_raw, 0, 1)` to every candidate prediction before
candidate selection, metrics, plots, and boundary diagnostics. The summary
retains both raw and clipped prediction diagnostics for traceability.

Early smoke-test drivers and environment checks have been archived here:

```text
surrogate_model/segmented_symbolic_regression/archive/smoke_tests
```

They are not part of the normal workflow anymore.

## PySR Training Setup

The training script reads MATLAB `-v7.3` `.mat` files with `h5py`.

The Python environment is:

```text
surrogate_model/segmented_symbolic_regression/.venv_py311
```

PySR calls Julia through:

```text
C:/Users/liuzi/.julia/juliaup/julia-1.11.9+0.x64.w64.mingw32/bin/julia.exe
```

The first operator set is intentionally conservative:

```text
binary operators: +, -, *, /
unary operators:  log, sqrt
```

The script also maps unsafe variable names before passing them to PySR. In particular:

```text
lambda -> lambda_var
```

This avoids SymPy export errors while keeping the original dataset column unchanged.

## Current Verification Status

Already verified:

```text
MATLAB -> Python environment works
Python -> PySR import works
PySR -> Julia works
full symbolic dataset generation works
full symbolic dataset inspection works
full segment-wise PySR training works
symbolic candidate evaluation and plotting works
```

The full symbolic dataset inspection report is here:

```text
surrogate_model/segmented_symbolic_regression/artifacts/wool_segmented_symbolic_dataset_inspection/symbolic_dataset_inspection_report.txt
```

Archived smoke-test material is here:

```text
surrogate_model/segmented_symbolic_regression/archive/smoke_tests
```

The current real training run is stored under:

```text
surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_pysr_segments_runs
```

The current real evaluation runs are stored under:

```text
surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_candidate_evaluation_runs
```

## Latest Training Run

Latest full PySR training run:

```text
surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_pysr_segments_runs/20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28
```

Training configuration:

```text
niterations = 100
populations = 12
population_size = 100
maxsize = 28
rows used = all 64000 scalar samples
segments = 5
```

Training-script selected formulas:

```text
low_100_700
  complexity = 10
  test RMSE  = 0.02293
  test MAE   = 0.01756
  test R2    = 0.97900

midlow_700_1000
  complexity = 22
  test RMSE  = 0.03525
  test MAE   = 0.02702
  test R2    = 0.71953

midhigh_1000_1300
  complexity = 20
  test RMSE  = 0.03326
  test MAE   = 0.02410
  test R2    = 0.74776

highlow_1300_1650
  complexity = 13
  test RMSE  = 0.02791
  test MAE   = 0.01899
  test R2    = 0.87928

high_1650_2000
  complexity = 9
  test RMSE  = 0.02224
  test MAE   = 0.01502
  test R2    = 0.92930
```

## Latest Evaluations

Two useful evaluations were run on the latest full training run.

### Best-loss selection

This selects the lowest-error candidate per segment, even if the formula is complex.

```text
surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_candidate_evaluation_runs/20260718_160542_20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28_best_loss
```

Overall metrics:

```text
RMSE = 0.02595
MAE  = 0.01839
R2   = 0.98986
max absolute error = 0.21218
```

Selected candidates:

```text
low_100_700        candidate 15, complexity 27
midlow_700_1000    candidate 20, complexity 26
midhigh_1000_1300  candidate 20, complexity 27
highlow_1300_1650  candidate 20, complexity 25
high_1650_2000     candidate 19, complexity 27
```

This is more accurate but less interpretable.

### Complexity-limited selection

This selects the best candidate with complexity not greater than 12.

```text
surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_candidate_evaluation_runs/20260718_160548_20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28_max_complexity
```

Evaluation command:

```powershell
& 'surrogate_model\segmented_symbolic_regression\.venv_py311\Scripts\python.exe' `
  'surrogate_model\segmented_symbolic_regression\evaluate_segmented_symbolic_candidates.py' `
  --training-dir 'surrogate_model\segmented_symbolic_regression\artifacts\archived_pre_segmented_rename_artifacts\wool_symbolic_pysr_segments_runs\20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28' `
  --selection-rule max_complexity `
  --max-complexity 12 `
  --max-plot-points 6000 `
  --num-curves 6
```

Overall metrics:

```text
RMSE = 0.03141
MAE  = 0.02196
R2   = 0.98514
max absolute error = 0.20731
```

Selected candidates:

```text
low_100_700        candidate 8, complexity 12
midlow_700_1000    candidate 10, complexity 12
midhigh_1000_1300  candidate 9, complexity 12
highlow_1300_1650  candidate 11, complexity 12
high_1650_2000     candidate 9, complexity 12
```

This is less accurate than best-loss selection, but the formulas are shorter and more suitable for interpretation.
Compared with the previous 5-segment `iter40/pop8/ps80/size24` run, the larger search budget improves both best-loss and complexity-limited overall metrics.
The remaining bottleneck is still the 700-1000 Hz region.

### Complexity-limited selection, max complexity 16

This selection is the current best compromise between compactness and accuracy.

```text
surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_candidate_evaluation_runs/20260718_160553_20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28_max_complexity
```

Overall metrics:

```text
RMSE = 0.02875
MAE  = 0.02014
R2   = 0.98755
max absolute error = 0.18068
```

Selected candidates:

```text
low_100_700        candidate 9,  complexity 14
midlow_700_1000    candidate 13, complexity 15
midhigh_1000_1300  candidate 12, complexity 15
highlow_1300_1650  candidate 14, complexity 16
high_1650_2000     candidate 12, complexity 16
```

Parameter takeaway:

```text
niterations: increasing 40 -> 100 helped substantially.
populations/population_size: increasing 8*80 -> 12*100 improved search diversity.
maxsize: increasing 24 -> 28 helped best-loss candidates, but formulas above complexity 20 are harder to discuss.
Recommended reporting candidate: max_complexity=16.
```

Important: evaluation only reads and analyzes existing formulas. It does not retrain PySR and does not modify the training results.

## Example Full Training Command

Run from the project root:

```powershell
& 'surrogate_model\segmented_symbolic_regression\.venv_py311\Scripts\python.exe' `
  'surrogate_model\segmented_symbolic_regression\train_segmented_symbolic_models.py' `
  --dataset-run run2 `
  --niterations 40 `
  --populations 8 `
  --population-size 80 `
  --maxsize 24 `
  --run-name full_iter40_size24
```

If `--output-dir` is omitted, the script automatically creates a unique directory under:

```text
surrogate_model/segmented_symbolic_regression/artifacts/wool_segmented_symbolic_pysr_runs
```

Example directory name:

```text
20260718_124242_iter1_pop1_ps20_size8_cap30_autosave_smoke
```

For a quick debug run on the full dataset:

```powershell
& 'surrogate_model\segmented_symbolic_regression\.venv_py311\Scripts\python.exe' `
  'surrogate_model\segmented_symbolic_regression\train_segmented_symbolic_models.py' `
  --niterations 2 `
  --populations 2 `
  --population-size 30 `
  --maxsize 12 `
  --max-samples-per-segment 200 `
  --run-name debug_cap200 `
  --procs 1
```

## Example Evaluation Command

Evaluate a specific training run:

```powershell
& 'surrogate_model\segmented_symbolic_regression\.venv_py311\Scripts\python.exe' `
  'surrogate_model\segmented_symbolic_regression\evaluate_segmented_symbolic_candidates.py' `
  --dataset-run run2 `
  --training-dir 'surrogate_model\segmented_symbolic_regression\artifacts\wool_segmented_symbolic_pysr_runs\<training_run_directory>'
```

If `--output-dir` is omitted, the evaluation script automatically creates a unique directory under:

```text
surrogate_model/segmented_symbolic_regression/artifacts/wool_segmented_symbolic_candidate_evaluation_runs
```

If `--training-dir` is omitted, the evaluation script tries to use the latest run under:

```text
surrogate_model/segmented_symbolic_regression/artifacts/wool_segmented_symbolic_pysr_runs
```

## Output Artifacts

The training script exports:

```text
training_metadata.json
segment_model_summary.csv
segment_model_summary.json
equations/<segment>_equations.csv
models/<segment>_pysr_model.pkl
pysr_runs/<segment>/...
```

The most useful first file to inspect is:

```text
segment_model_summary.csv
```

The evaluation script exports:

```text
candidate_metrics.csv
selected_candidates.csv
selected_combination_summary.json
boundary_transition_metrics.csv
figures/<segment>_complexity_vs_rmse.png
figures/selected_predicted_vs_true.png
figures/selected_error_vs_frequency.png
figures/selected_curve_comparisons.png
```

It includes:

```text
segment name
selected equation
complexity
loss
train RMSE / MAE
test RMSE / MAE
test R2
number of rows used
```

## Design Goal

The symbolic model is not expected to beat the MLP baseline in pure accuracy.

The goal is to find formulas that are:

```text
reasonably accurate
compact
interpretable
physically discussable
```

Model selection should therefore consider both:

```text
prediction error
formula complexity
```

not error alone.

## Next Steps

Recommended next steps:

```text
1. Inspect the latest evaluation figures:
   - selected_curve_comparisons.png
   - selected_predicted_vs_true.png
   - selected_error_vs_frequency.png
2. Inspect candidate_metrics.csv for each evaluation run.
3. Decide whether the complexity-limited formulas are interpretable enough.
4. If accuracy or interpretability is not sufficient, run another PySR training pass with adjusted settings.
5. Compare symbolic model errors against the existing MLP baseline artifacts.
```
