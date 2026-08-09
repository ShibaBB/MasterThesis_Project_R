# Dataset-Run Workflow For R Components

Centralized run selection now uses the paired `R_real` and `R_imag` dataset
schema.

## Intended Configuration Flow

```text
dataset_run_config.json
  -> selects run1
datasets/run1/dataset_config.json
  -> defines generation, frequency grid, split, SR domains, and target schema
dataset_run_config.py / resolve_dataset_run_config.m
  -> resolve all teacher, split, segmented, and global paths
```

The run config includes this target schema:

```json
{
  "target_schema": {
    "target_names": ["R_real", "R_imag"],
    "target_keys": {"re": "Y_re", "im": "Y_im"},
    "symbolic_target_keys": {"re": "y_re_symbolic", "im": "y_im_symbolic"},
    "complex_source": "Reflect"
  }
}
```

## Generation Order

1. Create the R `run1` configuration.
2. Generate one paired teacher dataset containing `Y_re` and `Y_im`.
3. Inspect both component arrays and verify them against `Reflect`.
4. Generate one shared source-curve split.
5. Generate paired segmented and global scalar datasets.
6. Inspect both scalar targets and synchronize the manifest.

## Training Order

For each model family, run Re and Im separately using the same split:

```text
MLP:          re, im
Segmented SR: re, im
Global SR:    re, im
```

Candidate selection uses validation curves; final metrics use test curves.
No component prediction is clipped.

## Naming

Use `YYYYMMDD_runN` plus only essential suffixes such as `_c21` or `_limited`.
Place target identity in a short `re/` or `im/` parent directory. Store full
settings in JSON metadata.

See `current_project_handoff.md` for the detailed migration checklist and
acceptance criteria. The active commands now use the migrated loaders and
explicit target selectors.
