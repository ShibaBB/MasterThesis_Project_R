# Dataset Runs

This folder stores comparison datasets by run.

Use one `run*` folder at a time when comparing MLP, segmented SR, and global SR.
This keeps all models tied to the same teacher data.

## Active Run

`run3` is the active configured run. It uses Wool porosity 92, 1000 LHS
teacher samples, seed 44, and 128 frequency points over 100-4950 Hz. Its
`dataset_config.json` is the single source for teacher generation and both SR
domain definitions. Teacher data, the shared split, and both segmented/global
SR datasets are generated. The manifest status is `generated`.

Use `../DATASET_RUN_WORKFLOW.md` for the generation protocol.

## Completed Historical Run

`run2` is the latest completed historical comparison run. It was generated with Wool porosity 92,
1000 LHS teacher samples, 64 frequency points over 100-2000 Hz, and random seed
43. It contains:

- `MLP/Wool_surrogate_dataset.mat`
  - Curve-format teacher dataset.
  - Used directly by the MLP baseline.
- `segmented_SR/Wool_symbolic_segmented.mat`
  - Scalar frequency-expanded dataset derived from the run2 teacher dataset.
  - Uses the current segmented SR frequency split.
- `global_SR/Wool_symbolic_global.mat`
  - Scalar frequency-expanded dataset derived from the run2 teacher dataset.
  - Uses one global `100-2000 Hz` segment.
- `dataset_manifest.json`
  - Records the source/derived dataset relationship.
- `shared_curve_split.json`
  - Defines the common 1-based source curve indices used by every model.
  - Current sizes: 700 train, 150 validation, and 150 test curves.
  - SR rows are selected through `source_curve_index`; scalar rows are never
    independently randomized across splits.
  - The split hash excludes frequency range and frequency count. A compatible
    teacher dataset may therefore change from the current frequency grid to a
    later grid such as 100-4950 Hz without changing curve membership.

Generate the split after creating a new teacher dataset run:

```powershell
python surrogate_model/generate_shared_curve_split.py --dataset-run run3
```

If source curve count or source curve identities change, create a new dataset
run and regenerate its split. Training and evaluation artifacts record both
the resolved split path and SHA-256 split hash.

`run1` is retained as the earlier baseline run. Do not mix models trained on
run1 with models trained on run2 in a horizontal comparison.

For future experiments, use `create_dataset_run.py`; core model scripts should
not be edited merely to advance from run3 to run4 or run5.
