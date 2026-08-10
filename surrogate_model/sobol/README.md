# Frequency-Resolved Sobol Analysis

This directory owns sensitivity-analysis code and artifacts independently of
the MLP, global symbolic-regression, and segmented symbolic-regression model
branches. It may read canonical datasets and trained model artifacts, but it
must not write into those branches.

## Current Analyses

`run_mlp_sobol_pilot.m` uses the formal `run1` Re and Im MLP models as fast
emulators of the JCAL teacher. The analysis fixes the two inputs that do not
vary in `run1`:

```text
phi = 0.92
h   = 0.03 m
```

It varies the five sampled material parameters independently over the same
raw, linear-uniform physical bounds used by teacher-data generation:

```text
sigma, alpha_infinity, lambda, lambda_prime, k0_prime
```

The Sobol distribution is defined in raw physical units. The MLP then reuses
its persisted training preprocessing, including log10 transforms and
standardization. Changing the Sobol distribution to log-uniform would answer a
different sensitivity question and must be recorded as a separate run.

The pilot estimates pointwise first-order (`S1`) and total-effect (`ST`) Jansen
indices at all 128 frequencies. It also reports `ST-S1`, nested-sample
convergence, approximate bootstrap intervals, and variance-weighted functional
indices for the complete frequency range and the eight configured SR segments.

## Run

From the repository root:

```matlab
run('surrogate_model/sobol/run_mlp_sobol_pilot.m')
```

All outputs are written below:

```text
surrogate_model/sobol/artifacts/mlp/<run>/
```

The MLP pilot is a screening and experiment-design tool. Physical conclusions
must be checked against the JCAL teacher, especially for small effects,
interactions, and frequencies where emulator error or Sobol uncertainty is
large. Sobol indices calculated from a selected SR formula describe that
formula, not parameters omitted by the formula.

`run_teacher_sobol_validation.m` performs that check on the first `N=1024`
points of the same scrambled Sobol design. It evaluates 7,168 JCAL material
curves (`N * (D + 2)`), with every curve returning all 128 frequencies and both
reflection components. Its artifacts remain separate from the MLP pilot:

```text
surrogate_model/sobol/artifacts/teacher/<run>/
```

Run it after the MLP pilot:

```matlab
run('surrogate_model/sobol/run_teacher_sobol_validation.m')
```

## Formal Run1 Results

```text
MLP pilot:          artifacts/mlp/20260810_run1_n4096
Teacher validation: artifacts/teacher/20260810_run1_n1024
```

The MLP and teacher frequency-resolved indices agree closely on the shared
`N=1024` design:

```text
Target   S1 RMSE   ST RMSE   flattened ST correlation   top-ST agreement
Re       0.004840  0.007870  0.999645                   100% of frequencies
Im       0.003436  0.003490  0.999947                   100% of frequencies
```

Teacher variance-weighted full-range total-effect indices are:

```text
Parameter          Re ST      Im ST
sigma              0.849552   0.840944
alpha_infinity     0.171057   0.215920
lambda             0.078213   0.096305
k0_prime           0.032073   0.025709
lambda_prime       0.006380   0.004819
```

These global values must not replace the frequency-resolved results. Important
local effects include `k0_prime` for Re around 482 Hz, rising
`alpha_infinity` effects for Re around 3-4 kHz, and strong high-frequency
interaction effects involving `sigma`, `alpha_infinity`, and `lambda` for Im.
The largest MLP-teacher ST difference occurs for Re `k0_prime` above 4 kHz, so
teacher indices should be used for final SR decisions in that region.
