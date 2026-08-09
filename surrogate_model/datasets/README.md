# Dataset Runs

The `run1` configuration and manifest exist, but the full generated MAT files
and shared split do not yet exist in this repository.

The first new reflection-coefficient dataset should be `run1`. A run must own:

```text
dataset_config.json
dataset_manifest.json
shared_curve_split.json
MLP/<material>_R.mat
segmented_SR/<material>_R_segmented.mat
global_SR/<material>_R_global.mat
```

The exact filenames may be finalized during implementation, but they must stay
short and be resolved from the central run configuration.

## Teacher Dataset Contract

The teacher MAT file should contain:

```text
X         [n_curves, 7]
Y_re      [n_curves, n_frequency_points]
Y_im      [n_curves, n_frequency_points]
freq_grid
dataset_info
sample_metadata
```

`Y_re` and `Y_im` must be generated from the same complex `Reflect` values,
not from separate sampling jobs. One shared curve split applies to both.

## Symbolic Dataset Contract

Both segmented and global scalar datasets should retain:

```text
X_symbolic, y_re_symbolic, y_im_symbolic,
source_curve_index, segment_index, freq_grid, symbolic_dataset_info
```

Training scripts select one target explicitly with `re|im`.

## Manifest Requirements

The manifest should record:

- dataset run and source paths;
- target schema and definitions;
- source curve count and frequency grid;
- shared split path and hash;
- generation state of teacher, segmented, and global datasets.

Do not mark a run complete until both Re and Im arrays are present and valid.
