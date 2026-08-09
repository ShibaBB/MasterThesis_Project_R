# Global Symbolic Regression

This folder contains the global symbolic-regression model, which fits one
PySR equation over the full frequency range defined by the active dataset run.
For run3 this is 100-4950 Hz and the domain is named `global_100_4950`.

The intended comparison structure is:

- `surrogate_model/MLP`: MLP baseline model.
- `surrogate_model/segmented_symbolic_regression`: segmented symbolic-regression model.
- `surrogate_model/global_symbolic_regression`: one-formula global symbolic-regression model.

The global SR branch reuses the shared teacher dataset and scalar symbolic
dataset format. Training validates that exactly one configured domain is present.

Active generated dataset:

- `surrogate_model/datasets/run3/global_SR/Wool_symbolic_global.mat`

Latest completed historical dataset:

- `surrogate_model/datasets/run2/global_SR/Wool_symbolic_global.mat`

Current data entry points:

- `run_global_symbolic_dataset_generation.m`
- `run_global_symbolic_dataset_inspection.m`

Training and evaluation entry points:

- `train_global_symbolic_model.py`
- `evaluate_global_symbolic_candidates.py`

All entry points resolve the default dataset run from the central
`surrogate_model/dataset_run_config.json`.
Training and evaluation also read the selected run's
`shared_curve_split.json` (currently `surrogate_model/datasets/run3/shared_curve_split.json`). Scalar rows are mapped
through `source_curve_index`; training, candidate selection, and final
evaluation use train, validation, and test source curves respectively. The
split is independent of frequency range and frequency-point count.
Training artifacts are written under `artifacts/wool_global_symbolic_pysr_runs`;
evaluation artifacts are written under
`artifacts/wool_global_symbolic_candidate_evaluation_runs`.

## Current Formal Run3 Result

Full training:

- `artifacts/wool_global_symbolic_pysr_runs/20260808_run3_full`
- 100 iterations, 12 populations, population size 100, maxsize 28
- 128000 scalar rows total; fitting uses the 89600 training rows only
- shared split rows: 89600 train, 19200 validation, 19200 test

Recommended evaluation:

- `artifacts/wool_global_symbolic_candidate_evaluation_runs/20260808_run3_c21`
- selection rule: `max_complexity=21`
- selected candidate: 14, complexity 19
- test RMSE: `0.0578571752`
- test MAE: `0.0456781372`
- test R2: `0.9229031506`
- raw/final range: `[0.031824, 0.927263]`; no clipping required

This result uses the shared run3 source-curve split and is eligible for the
three-model horizontal comparison. Candidate selection uses validation curves;
the reported metrics use test curves. Predictions are clipped to the physical
range `[0,1]` during candidate evaluation and reporting. A c28 sensitivity run
is retained under `20260808_run3_c28`; it improves test RMSE to `0.0553558465`
but clips 150 negative values, so c21 remains the recommended clean result.
Run1/run2 and the 2026-08-03 runs are historical comparison points only.
