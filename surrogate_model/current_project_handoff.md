# Surrogate Model Project Handoff

This handoff describes the current state of `surrogate_model`: model branches,
run logic, shared dataset structure, artifact layout, and Git/GitHub sync
policy. It is intended to give a new conversation enough context to continue
work without rediscovering the repository.

Last updated: 2026-08-08, after validating MLP, segmented SR, and global SR
end to end following the workspace-root rename.

## Run3 Transition (active configuration)

The active workflow uses a centralized dataset-run protocol. All three active
model branches now have completed run3 baselines, and all active generation
and model entry points resolve `run3` from:

```text
surrogate_model/dataset_run_config.json
```

Run3 teacher data, shared curve split, both SR scalar datasets, and all three
model baselines have now been generated. Its per-run configuration is:

```text
surrogate_model/datasets/run3/dataset_config.json
Wool / porosity 92 / 1000 LHS curves / seed 44
100-4950 Hz / 128 linearly spaced frequency points
```

Future run4/run5 experiments clone a per-run configuration with
`create_dataset_run.py`; model code must not be changed just to select a new
run. See `DATASET_RUN_WORKFLOW.md`. Existing run1/run2 manifests and artifacts
retain their historical 100-2000 Hz metadata for provenance.

The first full run3 MLP baseline completed successfully on 2026-08-08:

```text
artifact: MLP/artifacts/wool_baseline_mlp_runs/20260808_164951_run3_baseline
split: 700 train / 150 validation / 150 test curves
raw test RMSE: 0.0025906994
raw test MAE:  0.0018582958
raw test R2:   0.9978877902
```

The 2000-4950 Hz test band was not harder than the retained lower band: raw
RMSE was 0.00229737 above 2000 Hz versus 0.00299138 below 2000 Hz. Raw MLP
predictions ranged from 0.0107055 to 1.0096585; 22 of 19,200 test predictions
were above 1. Clipping to [0,1] changed full-range RMSE only slightly, from
0.00259070 to 0.00258475. See the run's `extended_evaluation_report.txt`.

The first full run3 segmented SR baseline also completed successfully on
2026-08-08:

```text
dataset: datasets/run3/segmented_SR/Wool_symbolic_segmented.mat
training: segmented_symbolic_regression/artifacts/
  wool_segmented_symbolic_pysr_runs/20260808_run3_full
recommended evaluation: segmented_symbolic_regression/artifacts/
  wool_segmented_symbolic_candidate_evaluation_runs/20260808_run3_c21
```

The dataset contains 128,000 scalar rows across eight configured frequency
segments. Training used the 89,600 rows belonging to the 700 training curves;
validation and test each used 19,200 rows. Each model used 100 iterations, 12
populations, population size 100, maxsize 28, and deterministic serial
execution. The eight models completed in about 37 minutes. PySR emitted only
its expected performance advisory for segments containing more than 10,000
training points.

Validation comparison favored `max_complexity=21` over the historical c16
ceiling (validation RMSE 0.0255686 versus 0.0270282). The recommended c21
test-only clipped result is:

```text
RMSE          0.0252275807
MAE           0.0183131128
max_abs_error 0.1904615675
R2            0.9853420448
```

The selected formula complexities are 20, 21, 21, 20, 19, 20, 20, and 20.
Raw predictions ranged from -0.0067388 to 1.0177128; 46 of 19,200 test values
were clipped to [0,1]. Boundary discontinuity remains the main segmented-model
limitation. The largest c21 excess boundary change occurred at 2000 Hz
(maximum 0.18431), and visible steps also remain near 3000 and 4000 Hz. The c16
evaluation is retained as the simpler comparison and has test RMSE 0.0270428.

The first full run3 global SR baseline completed successfully on 2026-08-08:

```text
dataset: datasets/run3/global_SR/Wool_symbolic_global.mat
inspection: global_symbolic_regression/artifacts/
  wool_symbolic_global_dataset_inspection_run3
training: global_symbolic_regression/artifacts/
  wool_global_symbolic_pysr_runs/20260808_run3_full
recommended evaluation: global_symbolic_regression/artifacts/
  wool_global_symbolic_candidate_evaluation_runs/20260808_run3_c21
```

The dataset contains 128,000 scalar rows in one `global_100_4950` domain.
Training used 89,600 rows from the 700 training curves; validation and test each
used 19,200 rows. PySR used 100 iterations, 12 populations, population size 100,
maxsize 28, and completed in about 46 minutes 50 seconds. Validation selection
with `max_complexity=21` chose candidate 14, complexity 19. Its test result is
RMSE `0.0578571752`, MAE `0.0456781372`, max absolute error `0.2638560413`, and
R2 `0.9229031506`. Predictions stayed in `[0.031824, 0.927263]`, so no clipping
was required. The formula depends only on frequency and captures the mean curve
shape, but not enough material-to-material variation. A retained c28 sensitivity
run improved test RMSE to `0.0553558465` and R2 to `0.9294252787`, but required
clipping 150 negative predictions; c21 remains the cleaner recommended result.

## Project Goal

The project compares surrogate models for a MATLAB JCAL acoustic absorption
teacher model.

Target mapping:

```text
material/acoustic parameters + frequency -> absorption coefficient alpha
```

The planned horizontal comparison has three model branches:

```text
1. MLP
2. segmented symbolic regression
3. global symbolic regression
```

All three branches must use data from the same dataset run for a fair
comparison. The current shared dataset run is now:

```text
surrogate_model/datasets/run3
```

`run1` and `run2` remain historical dataset runs. Do not mix models trained on
different runs in one horizontal comparison.

## Repository Location And Remote

The active local working copy has been moved off OneDrive. The current intended
workspace is:

```text
C:/MasterThesis_Project_alpha
```

The previous root `C:/MasterThesis_Project` is obsolete. Active code and the
latest model artifacts were checked for references to that old absolute path.

GitHub remote:

```text
https://github.com/ShibaBB/MasterThesis_Project_alpha
```

Current working branch as last verified on 2026-08-03:

```text
agent/add-segmented-evaluation
```

The repository default branch remains `main`. The current working branch adds
the latest formal segmented SR evaluation and is published in draft PR #1:

```text
https://github.com/ShibaBB/MasterThesis_Project_alpha/pull/1
```

Current synchronization logic:

```text
Each computer keeps its own local hard-drive clone.
Training and analysis are run from the local clone, not from OneDrive.
GitHub is used to synchronize code, shared datasets, and analysis-ready
artifacts between computers.
Only one computer should train/write/push a given run at a time.
```

OneDrive is no longer part of the intended training workflow. This avoids file
locking, delayed sync, and slow frequent writes during PySR runs.

## Top-Level Structure

Relevant project layout:

```text
surrogate_model/
  README.md
  current_project_handoff.md
  data_generation/
  datasets/
    run1/
      historical baseline dataset run
    run2/
      dataset_manifest.json
      MLP/
        Wool_surrogate_dataset.mat
      segmented_SR/
        Wool_symbolic_segmented.mat
      global_SR/
        Wool_symbolic_global.mat
  MLP/
  segmented_symbolic_regression/
  global_symbolic_regression/
  docs/
```

Directory roles:

```text
data_generation/
  Shared MATLAB teacher dataset generation and inspection scripts.

datasets/
  Shared dataset runs. Horizontal comparisons must use files from the same
  run folder.

MLP/
  MLP baseline code, notes, and MLP artifacts.

segmented_symbolic_regression/
  Active segmented symbolic regression branch. Uses PySR and trains separate
  formulas on predefined frequency segments.

global_symbolic_regression/
  Global symbolic regression branch. Fits and evaluates one PySR formula over
  the full 100-2000 Hz range.

docs/
  Background notes, original task descriptions, and strategy/reference files.
```

## Dataset Run Structure

The current dataset run is `run2`. It was generated successfully on 2026-08-03
with the same Wool/porosity/frequency configuration as run1, but with LHS
sampling seed `43` instead of run1's seed `42`. The different seed makes run2 a
new teacher sample rather than a byte-for-byte copy of run1.

Manifest:

```text
surrogate_model/datasets/run2/dataset_manifest.json
```

Run contents:

```text
surrogate_model/datasets/run2/MLP/Wool_surrogate_dataset.mat
surrogate_model/datasets/run2/segmented_SR/Wool_symbolic_segmented.mat
surrogate_model/datasets/run2/global_SR/Wool_symbolic_global.mat
```

Comparison rule:

```text
Use only datasets within the same run_id for horizontal model comparison.
Do not compare a model trained on one dataset run with a model trained on
another dataset run.
```

### Shared Source-Curve Split

The current cross-model split is:

```text
surrogate_model/datasets/run2/shared_curve_split.json
```

It assigns the 1000 source curves to 700 train, 150 validation, and 150 test
curves. Its SHA-256 split hash is:

```text
512522db31338b940ba555d36ea567646fb112dbcd802c460e134dd6ff58d437
```

The split is defined only over 1-based source curve indices and is deliberately
independent of frequency range and frequency count. MLP consumes these curve
indices directly. Segmented and global SR select scalar rows using
`source_curve_index`. Candidate selection is performed on validation curves;
final reported evaluation is performed on test curves. Training and evaluation
artifacts record both the resolved split path and split hash, and SR evaluation
rejects a training run with a missing or different hash.

### Teacher / MLP Dataset

Teacher dataset file:

```text
surrogate_model/datasets/run2/MLP/Wool_surrogate_dataset.mat
```

Format:

```text
X: 1000 x 7
Y: 1000 x 64
```

Inputs:

```text
phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime
```

Output:

```text
alpha curve on freq_grid
```

Frequency range:

```text
100-2000 Hz, 64 frequency points
```

### Segmented SR Dataset

Segmented SR dataset file:

```text
surrogate_model/datasets/run2/segmented_SR/Wool_symbolic_segmented.mat
```

Format:

```text
scalar_frequency_expanded
```

Inputs:

```text
phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f
```

Output:

```text
alpha
```

Size:

```text
64000 scalar samples
```

Current frequency segments:

```text
low_100_700        100-700 Hz
midlow_700_1000    700-1000 Hz
midhigh_1000_1300  1000-1300 Hz
highlow_1300_1650  1300-1650 Hz
high_1650_2000     1650-2000 Hz
```

### Global SR Dataset

Global SR dataset file:

```text
surrogate_model/datasets/run2/global_SR/Wool_symbolic_global.mat
```

Format:

```text
scalar_frequency_expanded
```

Inputs:

```text
phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f
```

Output:

```text
alpha
```

Size:

```text
64000 scalar samples
```

Current global segment:

```text
global_100_2000    100-2000 Hz
```

## Teacher Model Logic

The MATLAB teacher dataset is generated from the JCAL acoustic model. The
important conceptual chain is:

```text
getFluidProperties
-> getFiberConstraints
-> jcal_reflection
-> generate_teacher_dataset
```

Current material setup:

```text
material: Wool
porosity case: 92
frequency range: 100-2000 Hz
```

The MLP uses the curve-format teacher data directly. The symbolic regression
branches use scalar frequency-expanded versions derived from the same teacher
dataset.

## MLP Branch State

Branch directory:

```text
surrogate_model/MLP
```

Current active status:

```text
The run2 MLP teacher dataset has been generated successfully.
A formal run2 MLP baseline has completed using the shared source-curve split.
The older 50-curve small baseline remains a historical pipeline smoke test.
```

Known artifact:

```text
surrogate_model/MLP/artifacts/wool_baseline_mlp_small
```

Latest formal shared-split artifact after the workspace rename:

```text
surrogate_model/MLP/artifacts/wool_baseline_mlp_runs/20260808_113400_run2_baseline
```

The older fixed-name `wool_baseline_mlp_run2` directory is preserved for
traceability. New MLP executions use:

```text
surrogate_model/MLP/artifacts/wool_baseline_mlp_runs/
  <timestamp>_<experiment_name>/
```

Automatic names are collision-safe, and an explicitly supplied nonempty
artifact directory is rejected rather than overwritten.

Formal run2 test metrics are RMSE `0.0023771671`, MAE `0.0016958551`, and R2
`0.9984784126`. These replace the earlier run2 MLP result that used an
internally generated split.

The 2026-08-08 run completed successfully from
`C:/MasterThesis_Project_alpha`, recorded the new resolved dataset and split
paths, and generated all expected model, prediction, metric, and figure
artifacts. No MLP path-code change was required after the rename.

The MLP branch should use:

```text
surrogate_model/datasets/run2/MLP/Wool_surrogate_dataset.mat
```

## Segmented Symbolic Regression Branch State

Branch directory:

```text
surrogate_model/segmented_symbolic_regression
```

This is the most mature model branch. It trains separate PySR equations for
five frequency segments.

Important entry points:

```text
run_full_segmented_symbolic_dataset_generation.m
run_full_segmented_symbolic_dataset_inspection.m
train_segmented_symbolic_models.py
evaluate_segmented_symbolic_candidates.py
```

All active model and data-generation entry points now resolve their default
dataset run from the central selector:

```text
surrogate_model/dataset_run_config.json
```

The current default is `run3`. MATLAB generation/inspection and Python
training/evaluation therefore use the same run by default. Python also accepts
`--dataset-run`; standard `datasets/run*/...` paths are checked against it,
and evaluation refuses to combine a dataset with training metadata from a
different run.

Current output roots:

```text
artifacts/wool_segmented_symbolic_pysr_runs
artifacts/wool_segmented_symbolic_candidate_evaluation_runs
artifacts/wool_segmented_symbolic_dataset_inspection
```

Archived old pre-rename artifacts are grouped here:

```text
artifacts/archived_pre_segmented_rename_artifacts
```

Those archived folders are retained for traceability only. Current scripts do
not write to the old unsegmented names:

```text
wool_symbolic_pysr_segments_runs
wool_symbolic_candidate_evaluation_runs
wool_symbolic_dataset_inspection
```

### Latest Segmented SR Dataset Generation

The current segmented SR dataset in `run2` has been generated successfully from
the run2 teacher dataset:

```text
surrogate_model/datasets/run2/segmented_SR/Wool_symbolic_segmented.mat
```

### Latest Full Segmented SR Training

Latest completed full shared-split segmented SR training run:

```text
surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_segmented_symbolic_pysr_runs/20260808_113903_run2_rootrename
```

This completed shared-split run is retained under the archive for traceability.
Its metadata paths have been synchronized with the archived location. Future
runs write directly under the active standard `wool_segmented_symbolic_*_runs`
roots with short run names.

Configuration:

```text
niterations = 100
populations = 12
population_size = 100
maxsize = 28
rows used = all 64000 scalar samples
parallelism = serial
dataset run = run2
```

Training summary:

```text
segment              complexity   test_rmse   test_mae   test_r2
low_100_700          21           0.019449    0.015189   0.984828
midlow_700_1000      18           0.030287    0.023426   0.757797
midhigh_1000_1300    13           0.036732    0.028008   0.660112
highlow_1300_1650    14           0.027023    0.019230   0.875655
high_1650_2000        9           0.021219    0.015567   0.931240
```

### Latest Segmented SR Evaluation

Latest complete shared-split evaluation run:

```text
surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_segmented_symbolic_candidate_evaluation_runs/20260808_120119_run2_rootrename_c16
```

Selection rule:

```text
max_complexity = 16
```

Selected candidates are identified by 1-based row/index in each segment's
equations CSV. Candidate index and expression complexity are different fields:

```text
segment              candidate_index   complexity
low_100_700           9                15
midlow_700_1000      12                16
midhigh_1000_1300     9                15
highlow_1300_1650    13                16
high_1650_2000        9                11
```

Overall selected-formula metrics:

```text
RMSE          0.0271962706
MAE           0.0200443252
max_abs_error 0.1576901805
R2            0.9884268272
```

This `max_complexity=16` evaluation is the current shared-split segmented SR
result for run2 comparison. Candidate selection used validation curves and the
reported metrics use only test curves. Training row counts were 44,800 train,
9,600 validation, and 9,600 test rows across the five segments.

Evaluation now applies output clipping to every candidate and selected formula:
`alpha_final = clip(alpha_raw, 0, 1)`. Candidate selection, reported metrics,
figures, and boundary diagnostics all use the clipped predictions. For this
run, the raw range was `-0.012258` to `1.026859`; 29 of 9,600 test predictions
were clipped, and the final range is exactly `[0,1]`. Raw and clipped
diagnostics are retained in `selected_combination_summary.json`.

The earlier 2026-08-03 `run2_iface_full` / `run2_iface_maxc16` artifacts predate
the shared source-curve split and are historical only. They must not replace
the 2026-08-08 result in the final three-model comparison.

The first validation attempt used an automatically generated directory whose
deepest PySR path reached 265 characters and failed while Julia was opening
`hall_of_fame.csv`. The incomplete trace directory is:

```text
surrogate_model/segmented_symbolic_regression/artifacts/wool_segmented_symbolic_pysr_runs/20260803_115855_run2_iter100_pop12_ps100_size28_allrows_interface_validation_full_serial
```

The training entry point was then hardened to use shorter automatic names,
preflight Windows nested-path length, pass Julia forward-slash paths, prefer
the project-venv Julia executable, and use deterministic serial PySR settings.
The successful full run used the short explicit output directory above. The
deterministic/serial follow-up was verified with a five-segment smoke run under
the ignored `artifacts/_scratch` tree; the full run itself completed before the
deterministic flag was added, so an exact bit-for-bit retrain is not guaranteed.

### Local Environment And Smoke-Test Status

The full segmented training/evaluation pipeline has also been smoke-tested
successfully on `C:/MasterThesis_Project_alpha`. The verified local environment is:

```text
Python 3.11.9
PySR 1.5.10
Julia 1.11.9, installed inside the project venv through JuliaCall/JuliaPkg
MATLAB R2025b
venv: surrogate_model/segmented_symbolic_regression/.venv_py311
```

The venv is ignored by Git and must be recreated on another machine. The smoke
test verified all of the following:

```text
MATLAB teacher dataset: X = 1000 x 7, Y = 1000 x 64
segmented scalar dataset: X = 64000 x 8, y = 64000
all five segment names decoded correctly
PySR successfully called Julia/SymbolicRegression.jl
five-segment minimal training completed
candidate evaluation and figure generation completed
```

Committed local-disk smoke artifacts are retained under:

```text
surrogate_model/segmented_symbolic_regression/archive/smoke_tests/artifacts/
  segmented_local_disk_smoke_training_20260727_193327/
  segmented_local_disk_smoke_evaluation_20260727_193327/
```

An additional later smoke test exists only as untracked local temporary output
under the old path requested for that test:

```text
surrogate_model/symbolic_regression/archive/smoke_tests/artifacts/
  current_machine_smoke_20260727_201458/
  current_machine_smoke_20260727_201458_eval/
```

That untracked `surrogate_model/symbolic_regression` tree contains smoke output
only. It is not an active model branch, was intentionally excluded from the
formal evaluation commit/PR, and should not be committed or deleted without an
explicit decision.

## Global Symbolic Regression Branch State

Branch directory:

```text
surrogate_model/global_symbolic_regression
```

Current status:

```text
The run3 global scalar dataset has been generated successfully.
Dataset generation, inspection, training, and evaluation code exist.
Formal run3 global training and max-complexity evaluation completed successfully.
```

Important entry points:

```text
run_global_symbolic_dataset_generation.m
run_global_symbolic_dataset_inspection.m
train_global_symbolic_model.py
evaluate_global_symbolic_candidates.py
```

Current global dataset:

```text
surrogate_model/datasets/run3/global_SR/Wool_symbolic_global.mat
```

Current global inspection artifact:

```text
surrogate_model/global_symbolic_regression/artifacts/wool_symbolic_global_dataset_inspection_run3
```

The global inspection driver reuses the segmented inspection logic but
explicitly overrides the output directory so global artifacts stay under:

```text
surrogate_model/global_symbolic_regression/artifacts
```

Latest completed full shared-split global SR training run:

```text
surrogate_model/global_symbolic_regression/artifacts/wool_global_symbolic_pysr_runs/20260808_run3_full
```

Configuration:

```text
niterations = 100
populations = 12
population_size = 100
maxsize = 28
train rows = 89600 of 128000 scalar samples
validation rows = 19200
test rows = 19200
parallelism = serial
dataset run = run3
```

The PySR `model_selection=best` training equation has complexity 7 and test
RMSE `0.0668924`, test MAE `0.0523545`, and test R2 `0.8969435`. This training
summary is not the final reported candidate-selection result below.

Latest complete global evaluation run:

```text
surrogate_model/global_symbolic_regression/artifacts/wool_global_symbolic_candidate_evaluation_runs/20260808_run3_c21
```

The `max_complexity=21` rule selected candidate 14, complexity 19:

```text
(2.185653 + f)
* (0.0028456913 + f * (4.0260718e-7 - 6.216629e-5 / sqrt(f))
   - 0.019528603 / sqrt(f))
```

Test metrics are RMSE `0.0578571752`, MAE `0.0456781372`, max absolute error
`0.2638560413`, and R2 `0.9229031506`. The raw test prediction range was
`0.031824` to `0.927263`; none of 19,200 test predictions required clipping.
Candidate selection used validation curves; final reporting used test curves.

The retained run3 c28 sensitivity evaluation has test RMSE `0.0553558465`, but
150 raw negative predictions require clipping. Run1/run2 and earlier 2026-08-03
results remain historical; run3 c21 is the current result eligible for the
horizontal comparison.

## Current Artifact Layout

Segmented SR current artifact roots:

```text
surrogate_model/segmented_symbolic_regression/artifacts/
  archived_pre_segmented_rename_artifacts/
  wool_segmented_symbolic_candidate_evaluation_runs/
  wool_segmented_symbolic_pysr_runs/
```

Segmented SR smoke-test artifacts are kept separately from formal artifacts:

```text
surrogate_model/segmented_symbolic_regression/archive/smoke_tests/artifacts/
```

If segmented dataset inspection is run again, this current-name folder may also
appear:

```text
surrogate_model/segmented_symbolic_regression/artifacts/wool_segmented_symbolic_dataset_inspection
```

Global SR current artifact root:

```text
surrogate_model/global_symbolic_regression/artifacts/
  wool_symbolic_global_dataset_inspection/
  wool_global_symbolic_pysr_runs/
  wool_global_symbolic_candidate_evaluation_runs/
```

Global SR smoke-test artifacts are kept under:

```text
surrogate_model/global_symbolic_regression/archive/smoke_tests/artifacts/
```

MLP current formal artifact root and latest verified run:

```text
surrogate_model/MLP/artifacts/wool_baseline_mlp_runs/
  20260808_164951_run3_baseline/
```

The old `wool_baseline_mlp_small` output is a historical smoke result, not the
current comparison baseline.

## Run Logic

A full fair comparison should use only one dataset run at a time. For the
current project state that means all model branches should read from:

```text
surrogate_model/datasets/run3
```

Current data generation sequence is:

```text
1. Generate MLP teacher curve data:
   surrogate_model/data_generation/generate_teacher_dataset.m

2. Generate segmented SR scalar data:
   surrogate_model/segmented_symbolic_regression/run_full_segmented_symbolic_dataset_generation.m

3. Generate global SR scalar data:
   surrogate_model/global_symbolic_regression/run_global_symbolic_dataset_generation.m
```

Current segmented SR training/evaluation sequence:

```text
1. train_segmented_symbolic_models.py
   --niterations 100
   --populations 12
   --population-size 100
   --maxsize 28

2. evaluate_segmented_symbolic_candidates.py
   --selection-rule max_complexity
   --max-complexity 21
```

Current global SR training/evaluation sequence:

```text
1. train_global_symbolic_model.py
   --niterations 100
   --populations 12
   --population-size 100
   --maxsize 28

2. evaluate_global_symbolic_candidates.py
   --selection-rule max_complexity
   --max-complexity 21
```

MLP, segmented SR, and global SR now all have completed run2 results tied to
the same `shared_curve_split.json`. The three runs use identical source-curve
membership: 700 training curves, 150 validation curves, and 150 test curves.
The next stage is metric harmonization and horizontal comparison; retraining is
not required merely to establish split comparability. Do not regenerate the
datasets unless intentionally creating another dataset run.

## Git And GitHub Sync Policy

Intended workflow for two computers:

```text
Computer A:
  git pull
  run local training/evaluation
  inspect outputs
  git add curated outputs and any code changes
  git commit
  git push

Computer B:
  git pull
  analyze the committed run or continue from the synchronized state
```

Use local hard-drive clones on both computers. Do not train from OneDrive.
Do not train from two computers at the same time against the same run/artifact
folder unless a separate run name is intentionally chosen.

Before starting work on either computer:

```powershell
git pull --ff-only
git status
```

After a successful training/evaluation run:

```powershell
git status
git add <code changes> <dataset/run manifest if changed> <analysis-ready artifacts>
git commit -m "Add segmented SR run ..."
git push
```

Files that should be committed when another computer needs to analyze a run
without retraining:

```text
dataset_manifest.json
datasets/run*/**/*.mat, when the dataset run is part of the shared comparison
segment_model_summary.csv
segment_model_summary.json
training_metadata.json
*_best_equation.json
equations/*.csv
candidate_metrics.csv
selected_candidates.csv
selected_combination_summary.json
figures/*.png
models/*.pkl, only when another computer needs to load PySR model objects
```

Files that should normally stay out of Git:

```text
.venv/
.venv_py311/
__pycache__/
*.pyc
pysr_runs/**/checkpoint.pkl
pysr_runs/**/*.bak
local scratch/tmp output folders
MATLAB autosave files
OS metadata files
```

`models/*.pkl` are intentionally not ignored, but they should be committed only
when they are needed for cross-computer analysis. Most analysis should work
from:

```text
equations/*.csv
*_best_equation.json
candidate_metrics.csv
selected_candidates.csv
selected_combination_summary.json
```

`pysr_runs/**/hall_of_fame.csv` is also intentionally not globally ignored,
but the preferred portable candidate table is `equations/*.csv`. Commit raw
`hall_of_fame.csv` files only if a specific follow-up requires PySR's raw
output.

The current `.gitignore` policy ignores transient environments, Python caches,
PySR checkpoints, PySR `.bak` files, and local scratch/tmp folders while keeping
analysis-ready artifacts visible to Git.

On Windows, long artifact paths can exceed the default Git path limit. The
local setup should keep this enabled:

```powershell
git config --global core.longpaths true
```

`core.longpaths` helps Git but does not guarantee that Python can create every
deep output path. The evaluation script's automatic directory name can exceed
the Windows path limit because it includes the full training-run name. If that
happens, pass an explicit short output directory, for example:

```powershell
--output-dir surrogate_model/segmented_symbolic_regression/artifacts/wool_segmented_symbolic_candidate_evaluation_runs/<timestamp>_<run_id>_maxc16
```

## Naming And Separation Rules

The old branch name:

```text
surrogate_model/symbolic_regression
```

has been replaced by:

```text
surrogate_model/segmented_symbolic_regression
```

Global SR must remain separate under:

```text
surrogate_model/global_symbolic_regression
```

Current segmented scripts and artifacts should use `segmented` in names where
possible. Old unsegmented artifact directories under
`archived_pre_segmented_rename_artifacts` are historical only and should not be
used as active output locations.

## Current Git State

The local hard-drive clone at `C:/MasterThesis_Project_alpha` was last verified
on 2026-08-08 after the root-rename validation runs:

```text
current branch: agent/add-segmented-evaluation
tracking branch: origin/agent/add-segmented-evaluation
working tree: dirty; it contains intentional code, split, artifact, and handoff changes
latest MLP run: 20260808_164951_run3_baseline
latest segmented shared-split run: 20260808_run3_full
latest segmented shared-split evaluation: 20260808_run3_c21
latest global shared-split run: 20260808_run3_full
latest global shared-split evaluation: 20260808_run3_c21
```

Do not discard, reset, or overwrite the current dirty worktree. Historical
smoke artifacts, raw PySR outputs, and user changes are mixed with the current
uncommitted work and must be curated intentionally before committing.

If a future conversation starts from this handoff, first run:

```powershell
cd C:\MasterThesis_Project_alpha
git status --short --branch
git pull --ff-only
```

Because the working tree is currently dirty, do not run `git pull` until the
local changes have been reviewed and safely committed or otherwise protected.
Remote/PR state was not revalidated during the 2026-08-08 model runs.

## Current Maturity Summary

All three model branches now have complete run3 runs using the same source-curve
split and the 100-4950 Hz, 128-point grid. The workspace has been validated end
to end for MLP, segmented SR, and global SR.

Current comparable test results are:

```text
model          RMSE          MAE           R2
MLP            0.00259070    0.00185830    0.99788779
Segmented SR   0.02522758    0.01831311    0.98534204
Global SR      0.05785718    0.04567814    0.92290315
```

These values share test curve membership, but the reporting protocol still
needs one explicit harmonization pass before presenting the final comparison:
MLP currently reports its native curve prediction metrics, while SR evaluation
explicitly clips predictions to `[0,1]` and retains raw/clipped diagnostics.

Immediate project next steps are:

```text
1. Define and compute identical raw and clipped test metrics for all models.
2. Produce a three-model comparison table and common diagnostic figures.
3. Review segmented boundary discontinuities and global-formula complexity.
4. Only after the untouched baselines are documented, decide whether to tune MLP.
5. Curate the dirty worktree and decide which code/datasets/artifacts to commit.
```
