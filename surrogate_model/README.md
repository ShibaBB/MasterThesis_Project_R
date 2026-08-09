# Surrogate Model R Workspace

This workspace trains three surrogate-model families for the two components
of the JCAL complex reflection coefficient:

```text
material/acoustic parameters -> R_real(f)
material/acoustic parameters -> R_imag(f)
```

The retained model families are MLP, segmented PySR, and global PySR. Their
data generators, loaders, trainers, evaluators, metadata checks, and artifact
paths now implement the paired-component contract. No full `run1` dataset or
formal training/evaluation run has been generated yet.

Read [current_project_handoff.md](current_project_handoff.md) before changing
code. It is the authoritative implementation plan.

## Target Definition

The target is derived from the first output of `jcal_reflection.m`:

```matlab
R_real = real(Reflect);
R_imag = imag(Reflect);
```

The existing JCAL outputs named `Re` and `Im` describe normalized surface
impedance, not the reflection coefficient. They must not be used as surrogate
targets by mistake.

## Shared Data Contract

One teacher dataset will contain common inputs plus paired targets:

```text
X, Y_re, Y_im, freq_grid, dataset_info, sample_metadata
```

One source-curve split will be shared by both targets and all model families.
The first new dataset run should be R `run1`, using the existing 100–4950 Hz,
128-point configuration unless explicitly changed.

## Model Branches

- [MLP](MLP/README.md): two independent 128-output networks.
- [Segmented SR](segmented_symbolic_regression/README.md): one equation per
  frequency segment and target.
- [Global SR](global_symbolic_regression/README.md): one full-range equation
  per target.

Both targets are evaluated as unconstrained real values. No `[0,1]` clipping is
allowed.

## Short Artifact Naming

Use target parent folders and concise run names:

```text
artifacts/re/.../20260809_run1
artifacts/im/.../20260809_run1
artifacts/re/.../20260809_run1_c21
```

Full hyperparameters belong in metadata files, not directory names.

## Current Status

The code migration is complete. A 50-curve deterministic interface test
verified exact `Reflect` component extraction, paired scalar expansion, the
shared curve split, and independent one-epoch MLP Re/Im runs. Install the
PySR Python/Julia environment before launching symbolic searches, then create
the full paired dataset through `run_active_dataset_generation.ps1`.

The previously used Python package versions are recorded in
`requirements-pysr.txt`. Create
`segmented_symbolic_regression/.venv_py311` with Python 3.11 and install that
file before running dataset splitting or either PySR branch.
