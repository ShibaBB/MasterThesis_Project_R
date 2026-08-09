# Dataset-Run Workflow

The active surrogate workflow is configuration-driven. Model and generation
scripts must not hard-code `run1`, `run2`, `run3`, or a frequency endpoint.

## Configuration hierarchy

```text
surrogate_model/dataset_run_config.json
  -> selects the active dataset run

surrogate_model/datasets/<run>/dataset_config.json
  -> defines material, porosity cases, LHS seed, curve count, frequency grid,
     shared split, segmented SR domains, and the global SR domain
```

The current active run is `run3`, configured for Wool, porosity 92, 1000 LHS
curves, seed 44, and 128 linearly spaced points over 100-4950 Hz.

## Generate the active run

Validate configuration and show expected file state without generating data:

```powershell
powershell -ExecutionPolicy Bypass -File surrogate_model/run_active_dataset_generation.ps1 -ValidateOnly
```

Generate teacher data, the shared split, segmented SR data, global SR data,
and synchronize the manifest:

```powershell
powershell -ExecutionPolicy Bypass -File surrogate_model/run_active_dataset_generation.ps1
```

Generation resumes safely: completed stages are skipped, while individual
generators refuse to overwrite existing teacher, split, or derived datasets.
A changed experiment should be a new run rather than an in-place replacement.

## Train the active run

MLP:

```matlab
run('surrogate_model/MLP/mlp_run_baseline_training.m')
```

Segmented SR and global SR use their existing Python training/evaluation entry
points. Their default `--dataset-run` now comes from the same central active-run
configuration. Explicit `--dataset-run run4` remains supported.

Formal segmented SR baseline:

```powershell
surrogate_model\segmented_symbolic_regression\.venv_py311\Scripts\python.exe `
  surrogate_model\segmented_symbolic_regression\train_segmented_symbolic_models.py `
  --niterations 100 --populations 12 --population-size 100 --maxsize 28 `
  --run-name baseline
```

Formal global SR baseline:

```powershell
surrogate_model\segmented_symbolic_regression\.venv_py311\Scripts\python.exe `
  surrogate_model\global_symbolic_regression\train_global_symbolic_model.py `
  --niterations 100 --populations 12 --population-size 100 --maxsize 28 `
  --run-name baseline
```

Evaluation should then use the matching training directory and shared split.
The previous complexity ceilings (`16` segmented and `21` global) are run2
historical choices; run3 candidate selection should inspect the new validation
frontiers before fixing new ceilings.

## Create run4/run5

For another experiment with the same frequency domains, clone the current
configuration and select it as active:

```powershell
python surrogate_model/create_dataset_run.py run4 --from-run run3 --random-seed 45 --set-default
```

Add `--dry-run` to validate the clone request without creating files.

This creates only configuration and a `not_generated` manifest. It does not
generate data or train a model. If the frequency domain changes, edit only the
new run's `dataset_config.json`, including its segmented and global bounds.
Core scripts do not need changes.

## Invariants

- All three model branches in one comparison use the same dataset run.
- Split membership is defined over source curves, never independently over
  scalar frequency rows.
- Candidate selection uses validation curves; final metrics use test curves.
- Standard outputs must reside under their configured `datasets/<run>` folder.
- Existing dataset files are not overwritten implicitly.
- Historical run manifests and artifacts retain the frequency ranges that
  actually produced them.
