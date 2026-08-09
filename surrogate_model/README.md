# Surrogate Model Workspace

## Active Configuration

The active dataset run is selected centrally in `dataset_run_config.json`.
All active teacher-generation, split-generation, MLP, segmented SR, and global
SR entry points resolve that same value. The current active run is `run3`:

```text
Wool, porosity 92, 1000 LHS curves, seed 44
100-4950 Hz, 128 frequency points
```

See `DATASET_RUN_WORKFLOW.md` for generation, training, and future run4/run5
commands. Historical run1/run2 data remain traceability records, not active
defaults.

## Folder Layout

- [data_generation](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/data_generation>)
  - Shared dataset generation and dataset inspection scripts used by both MLP and symbolic-regression workflows.
- [MLP](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/MLP>)
  - MLP-only training scripts, MLP artifacts, and MLP-specific notes.
- [segmented_symbolic_regression](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/segmented_symbolic_regression>)
  - Current segmented symbolic-regression workflow using PySR.
  - This is the active segmented SR model branch.
- [global_symbolic_regression](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/global_symbolic_regression>)
  - Current global symbolic-regression workflow using one full-range PySR
    formula over the active run's frequency range (`100-4950 Hz` for run3).
- [docs](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/docs>)
  - General background notes, original strategy documents, and reference material.

## Dataset Runs

All model comparisons should use one dataset run at a time.

Current comparison dataset:

- `datasets/run3`
  - Teacher data and the shared curve split are present.
  - A complete MLP baseline is present under `MLP/artifacts/wool_baseline_mlp_runs/20260808_164951_run3_baseline`.
  - A complete segmented SR baseline is present under `segmented_symbolic_regression/artifacts/wool_segmented_symbolic_pysr_runs/20260808_run3_full`; the recommended evaluation is `20260808_run3_c21`.
  - A complete global SR baseline is present under `global_symbolic_regression/artifacts/wool_global_symbolic_pysr_runs/20260808_run3_full`; the recommended evaluation is `20260808_run3_c21`.

Latest completed historical comparison dataset:

- `datasets/run2`
  - `MLP/Wool_surrogate_dataset.mat`: shared teacher curve dataset for MLP.
  - `segmented_SR/Wool_symbolic_segmented.mat`: scalar dataset for segmented SR.
  - `global_SR/Wool_symbolic_global.mat`: scalar dataset for global SR.
  - `shared_curve_split.json`: shared source-curve train/validation/test split.
  - `dataset_manifest.json`: source and comparison metadata.

`datasets/run1` remains as the earlier baseline dataset run.

Use only datasets from the same `run*` folder when comparing MLP, segmented SR, and global SR.

All model branches must also use the same `shared_curve_split.json`. The split
is defined over 1-based source curve indices, not scalar frequency rows. MLP
reads the curve indices directly; symbolic datasets select rows through
`source_curve_index`. Candidate selection uses validation curves and final
metrics use test curves.

The shared split is frequency-grid independent. Changing the frequency range
or number of frequency points does not require changing the split while the
source curve identities remain compatible. Generate or regenerate it with:

```powershell
python surrogate_model/generate_shared_curve_split.py
```

## Recommended Entry Points

- Shared dataset generation:
  - [generate_teacher_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/data_generation/generate_teacher_dataset.m:1)
- Shared dataset inspection:
  - [inspect_teacher_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/data_generation/inspect_teacher_dataset.m:1)
- MLP baseline training:
  - [mlp_train_surrogate_baseline.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/MLP/mlp_train_surrogate_baseline.m:1)
  - `MLP/mlp_run_run2_baseline_training.m`: reproducible run2 entry point.
  - Each execution creates a unique timestamped directory under
    `MLP/artifacts/wool_baseline_mlp_runs/`. Existing nonempty artifact
    directories are never overwritten.
- Symbolic scalar dataset generation:
  - [generate_segmented_symbolic_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/generate_segmented_symbolic_dataset.m:1)
- Symbolic scalar dataset inspection:
  - [inspect_segmented_symbolic_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/inspect_segmented_symbolic_dataset.m:1)
- Symbolic-regression strategy:
  - [strategy_segmented_symbolic.md](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/strategy_segmented_symbolic.md:1)
- Symbolic handoff note:
  - [segmented_symbolic_regression_handoff.md](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/segmented_symbolic_regression/segmented_symbolic_regression_handoff.md:1)
- Global symbolic-regression training and evaluation:
  - [global_symbolic_regression](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/global_symbolic_regression>)

## Naming Rule

- MLP-only scripts use the `mlp_` prefix.
- Shared teacher-data pipeline scripts live only under `data_generation/`.
- Segmented symbolic-regression files live under `segmented_symbolic_regression/`.
- Global symbolic-regression files live under `global_symbolic_regression/`.

This separation is intended to reduce accidental cross-calling between the MLP and symbolic-regression branches.
