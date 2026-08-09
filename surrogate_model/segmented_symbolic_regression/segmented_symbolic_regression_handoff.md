# Symbolic Regression Handoff

## Run3 Update

The active 100-4950 Hz, 128-point run3 pipeline has completed end to end.
The scalar dataset has 128,000 rows across eight segments. Full training is in
`artifacts/wool_segmented_symbolic_pysr_runs/20260808_run3_full`; the
recommended validation-selected evaluation is
`artifacts/wool_segmented_symbolic_candidate_evaluation_runs/20260808_run3_c21`.
Its clipped test metrics are RMSE 0.0252275807, MAE 0.0183131128, and R2
0.9853420448. The older run2 details below remain historical reference.

## Current Status

- The symbolic-regression branch is separated from the MLP branch under [segmented_symbolic_regression](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/segmented_symbolic_regression>).
- The shared teacher-model pipeline is still reused from `data_generation`.
- The symbolic workflow already has dataset generation and dataset inspection scaffolding.
- The PySR runtime environment is ready.
- The segment-wise PySR training and candidate-evaluation workflow is implemented.
- The active full symbolic dataset is generated from `datasets/run3/MLP/Wool_surrogate_dataset.mat`.
- All active entry points share the central `surrogate_model/dataset_run_config.json`; its current default is `run3`.
- Training and evaluation now also require the run-level shared source-curve
  split at `datasets/run2/shared_curve_split.json`.
- Scalar rows are assigned through `source_curve_index`: training uses train
  curves, candidate selection uses validation curves, and final evaluation uses
  test curves. Training/evaluation metadata records and validates the split
  path and SHA-256 hash.
- Standard dataset paths are run-validated, and evaluation rejects training artifacts from a different run.
- Early smoke-test material has been archived under [archive/smoke_tests](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/archive/smoke_tests/README.md:1).

## Existing Files

- Strategy note:
  - [strategy_segmented_symbolic.md](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/strategy_segmented_symbolic.md:1)
- Scalar symbolic dataset builder:
  - [generate_segmented_symbolic_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/generate_segmented_symbolic_dataset.m:1)
- Scalar symbolic dataset inspection:
  - [inspect_segmented_symbolic_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/inspect_segmented_symbolic_dataset.m:1)
- Full symbolic dataset drivers:
  - [run_full_segmented_symbolic_dataset_generation.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/run_full_segmented_symbolic_dataset_generation.m:1)
  - [run_full_segmented_symbolic_dataset_inspection.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/run_full_segmented_symbolic_dataset_inspection.m:1)
- Segment-wise PySR training:
  - [train_segmented_symbolic_models.py](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/train_segmented_symbolic_models.py:1)
- Symbolic candidate evaluation and plotting:
  - [evaluate_segmented_symbolic_candidates.py](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/evaluate_segmented_symbolic_candidates.py:1)
- Archived smoke-test files:
  - [archive/smoke_tests](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/archive/smoke_tests/README.md:1)

## Current Data Flow

The current symbolic data flow is:

`teacher curve dataset -> scalar symbolic dataset -> symbolic dataset inspection`

The reused upstream shared chain is:

`getFluidProperties -> getFiberConstraints -> jcal_reflection -> generate_teacher_dataset`

## Symbolic Dataset Format

The symbolic dataset is already converted from curve form to scalar form:

- Input:
  - `[phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f]`
- Output:
  - `alpha`

The generated `.mat` file currently stores:

- `X_symbolic`
- `y_symbolic`
- `freq_grid`
- `source_curve_index`
- `source_porosityfolder`
- `segment_index`
- `segment_definitions`
- `symbolic_dataset_info`

The current default generated symbolic dataset is:

- `datasets/run2/segmented_SR/Wool_symbolic_segmented.mat`
- Curve samples: `1000`
- Frequency points: `64`
- Symbolic scalar samples: `64000`

## Default Scope Already Implemented

- Fiber: `Wool`
- Porosity: currently `92`
- Frequency range: `100-2000 Hz`
- Default segments:
  - `100-700 Hz`
  - `700-1000 Hz`
  - `1000-1300 Hz`
  - `1300-1650 Hz`
  - `1650-2000 Hz`

## What Has Already Been Verified

- MATLAB can use the prepared Python environment.
- Python can import `pysr`.
- PySR can call Julia successfully.
- Python can read the MATLAB `-v7.3` symbolic dataset through `h5py`.
- The full symbolic dataset generation driver ran successfully.
- The full symbolic dataset inspection driver ran successfully.
- `train_segmented_symbolic_models.py` ran successfully on the full symbolic dataset.
- `evaluate_segmented_symbolic_candidates.py` ran successfully on the full training artifacts.
- `train_segmented_symbolic_models.py` now creates a unique timestamped output directory by default under `artifacts/wool_segmented_symbolic_pysr_runs`.
- `evaluate_segmented_symbolic_candidates.py` now creates a unique timestamped output directory by default under `artifacts/wool_segmented_symbolic_candidate_evaluation_runs`.
- If `--training-dir` is omitted during evaluation, the latest training run matching the selected dataset run is used.
- A five-segment shared-split smoke training/evaluation completed successfully:
  the 700/150/150 curve split maps to 44800/9600/9600 scalar rows over the
  current 64-point grid, with no source curve crossing partitions.

The existing reporting-level segmented artifacts were trained before this
shared-split protocol. They remain historical and must not be used in the final
horizontal comparison. A new full run is required.

Current useful artifact locations:

- [artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_pysr_segments_runs](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_pysr_segments_runs)
- [artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_candidate_evaluation_runs](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_candidate_evaluation_runs)
- [artifacts/wool_segmented_symbolic_dataset_inspection](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/artifacts/wool_segmented_symbolic_dataset_inspection)

Current Git sync policy:

- Commit analysis-ready segmented SR outputs needed by another computer:
  `segment_model_summary.csv`, `segment_model_summary.json`,
  `training_metadata.json`, `*_best_equation.json`, `equations/*.csv`,
  `candidate_metrics.csv`, `selected_candidates.csv`,
  `selected_combination_summary.json`, and `figures/*.png`.
- Do not commit transient PySR recovery/cache files:
  `pysr_runs/**/checkpoint.pkl`, `pysr_runs/**/*.bak`, virtual
  environments, Python caches, or local scratch/tmp folders.
- `models/*.pkl` are not ignored by default. Commit them only when a follow-up
  machine needs to load PySR model objects directly.
- `pysr_runs/**/hall_of_fame.csv` is not globally ignored, but
  `equations/*.csv` is the preferred portable candidate table for later
  analysis and evaluation.
- For long runs, write PySR output to a local non-OneDrive scratch location
  first, then copy curated artifacts back into the repo before committing.

Latest full training run:

- [20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_pysr_segments_runs/20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28)

Latest useful evaluations:

- Best loss selection:
  - [20260718_160542_20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28_best_loss](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_candidate_evaluation_runs/20260718_160542_20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28_best_loss)
  - Overall metrics: `RMSE=0.02595`, `MAE=0.01839`, `R2=0.98986`
- Complexity-limited selection (`max_complexity=12`):
  - [20260718_160548_20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28_max_complexity](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_candidate_evaluation_runs/20260718_160548_20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28_max_complexity)
  - Overall metrics: `RMSE=0.03141`, `MAE=0.02196`, `R2=0.98514`
- Complexity-limited selection (`max_complexity=16`):
  - [20260718_160553_20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28_max_complexity](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/artifacts/archived_pre_segmented_rename_artifacts/wool_symbolic_candidate_evaluation_runs/20260718_160553_20260718_155023_iter100_pop12_ps100_size28_allrows_five_segments_iter100_pop12_ps100_size28_max_complexity)
  - Overall metrics: `RMSE=0.02875`, `MAE=0.02014`, `R2=0.98755`

Current 5-segment conclusion:

- The larger PySR search budget improves all segments compared with the `iter40/pop8/ps80/size24` run.
- `niterations=100`, `populations=12`, `population_size=100`, `maxsize=28` is the current best training configuration.
- `max_complexity=16` is the current best interpretability/accuracy compromise.
- The remaining bottleneck is mainly `700-1000 Hz`; higher-complexity candidates improve it, but at the cost of readability.

Important implementation note:

- PySR/SymPy cannot safely export a variable literally named `lambda`, so the training script maps feature names to PySR-safe names.
- Current mapping keeps the original data columns unchanged and only changes formula variable names, e.g. `lambda -> lambda_var`.
- The mapping is saved in `training_metadata.json`.

## Recommended Next Steps

1. Inspect the latest 5-segment evaluation figures and candidate metrics.
2. Decide whether the `max_complexity=12` formulas are interpretable enough.
3. If accuracy is not sufficient, try a middle-band focused run or allow `max_complexity=18`.
4. Keep the operator library conservative at first:
   - `+`
   - `-`
   - `*`
   - `/`
   - `log`
   - `sqrt`
5. Use `evaluate_segmented_symbolic_candidates.py` to inspect:
   - segment-wise symbolic error
   - full-range symbolic error
   - complexity-vs-error trade-offs
   - teacher-vs-symbolic curve plots
6. Add comparison against the MLP baseline.

## Suggested Implementation Order

1. Inspect the latest `selected_curve_comparisons.png`, `selected_predicted_vs_true.png`, and `selected_error_vs_frequency.png`.
2. Inspect `candidate_metrics.csv` for formula complexity versus RMSE.
3. If needed, run another training pass with `train_segmented_symbolic_models.py`.
4. Build comparison artifacts against the MLP baseline.

## Practical Reminder

- Keep the symbolic workflow independent from `mlp_train_surrogate_baseline.m`.
- Reuse shared teacher-data scripts, but do not reuse the MLP training logic itself.
- Keep the segmentation configurable from the start.
