# Surrogate Model R Workspace

This workspace trains three surrogate-model families for the two components
of the JCAL complex reflection coefficient:

```text
material/acoustic parameters -> R_real(f)
material/acoustic parameters -> R_imag(f)
```

The retained model families are MLP, segmented PySR, and global PySR. Their
data generators, loaders, trainers, evaluators, metadata checks, and artifact
paths implement the paired-component contract. The full `run1` dataset and
formal Re/Im training and evaluation artifacts exist for all three families.

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

The active teacher dataset contains common inputs plus paired targets:

```text
X, Y_re, Y_im, freq_grid, dataset_info, sample_metadata
```

One source-curve split is shared by both targets and all model families. The
active dataset is `run1`, using the 100-4950 Hz, 128-point configuration.

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

The accepted formal Global SR pair is Re F3 plus Im F2. Its production
inference, evaluation, export, active-model pointer, exact prediction replay,
paired complex output, frequency-boundary checks, and serialization tests are
implemented under [global_symbolic_regression](global_symbolic_regression/README.md).
The frozen Global SR formulas must not be refitted or tuned against their
consumed test data. The formal MLP remains the accuracy reference, and further
segmented SR work is deferred.

The active Python package versions are recorded in `requirements-pysr.txt`.
The existing Python 3.11 environment is
`segmented_symbolic_regression/.venv_py311`.
