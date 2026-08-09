# Surrogate Model Strategy 1

## 1. Purpose of This Version

This document refines the earlier surrogate-model strategy into a **first practical implementation plan** that matches the current MATLAB codebase and current project decisions.

The goal of this first version is **not** to perform full uncertainty quantification yet.

The immediate goal is:

1. treat the existing MATLAB `jcal_reflection.m` model as a reliable teacher model,
2. generate synthetic input-output data from it,
3. train a first surrogate model to reproduce the MATLAB absorption predictions,
4. verify that the surrogate can approximate the MATLAB model accurately.

At this stage, the surrogate is only meant to emulate the MATLAB forward model.

## 2. Core Modelling Position

For the current thesis route:

- `jcal_reflection.m` is the **reference forward model**
- the future DNN is the **surrogate model**
- the surrogate is trained on data generated from the MATLAB model

So the modelling chain is:

`material/sample settings -> MATLAB JCAL forward model -> absorption curve -> surrogate model`

For now, the MATLAB model is assumed reliable enough to act as the teacher.
It does **not** need to be revalidated through UQ before building the first surrogate.

## 3. Scope of the First Version

To keep the first version controlled and feasible:

- only **one material** is used
- do **not** mix materials in the same first surrogate
- the first material is **`Wool`**

The script structure should still allow changing the material in the same way as the current MATLAB workflow:

- define `fiberfolder`
- switch material by changing that variable

This follows the current pattern already used in `Uncertainty_quantification.m`.

## 4. Treatment of Porosity

The first working case should start from:

- porosity folder `92`
- numerical porosity `phi = 0.92`

However, the script should be written so that it can later handle:

- `92`, `93`, `94`, `95`, `96`, `97`, `98`, `99`

The implementation should therefore keep the same logic as the current code:

- `porosityfolder = '92'`
- `phi = str2double(porosityfolder) / 100`

So even if the first run uses only `92`, the framework must already support `92-99`.

## 5. Treatment of Thickness h

The thickness `h` should be obtained in exactly the same way as in the current uncertainty script.

That means:

1. call `getFluidProperties(fiberfolder, porosityfolder)`
2. read `thickness`
3. convert it from mm to m

namely:

`h = thickness * 1e-3`

This should not be hard-coded manually if the current project data already provides it.

## 6. Reference Files to Reuse

The most important reusable files are:

- [jcal_reflection.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/jcal_reflection.m:1)
  - teacher forward model
- [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:1)
  - reference for material switching, porosity handling, and thickness extraction
- [getFiberConstraints.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/getFiberConstraints.m:1)
  - reference for physically meaningful parameter bounds

Useful parameter-reference files:

- `Wool_92_params.txt`
- `Wool_92_random_params.txt`
- `Extracted_All_Params.txt`

In this first version, these files are used primarily to help define reasonable training ranges, not to perform full UQ.

## 7. First-Version Input-Output Definition

### 7.1 Inputs

For the first surrogate, the practical input vector should be:

`[phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime]`

where:

- `phi` = porosity
- `h` = material thickness
- `sigma` = airflow resistivity
- `alpha_infinity` = tortuosity
- `lambda` = viscous characteristic length
- `lambda_prime` = thermal characteristic length
- `k0_prime` = static thermal permeability

This choice is recommended because:

- it preserves the actual variables used by `jcal_reflection.m`
- it keeps the future framework compatible with different porosities
- it keeps `h` explicit rather than hiding it as a fixed condition

### 7.2 Outputs

The output should be:

`[alpha(f1), alpha(f2), ..., alpha(f64)]`

that is, the absorption coefficient on a fixed frequency grid.

The first version should predict **absorption coefficient directly**, not complex reflection coefficient.

This keeps the surrogate target:

- physically meaningful,
- easier to train,
- and directly useful for later engineering interpretation.

## 8. Frequency Range and Grid

For the first version:

- minimum frequency: `100 Hz`
- maximum frequency: `2000 Hz`
- number of frequency points: `64`

The frequency grid should be defined through configurable variables, for example:

- `freq_min`
- `freq_max`
- `n_freq`

and then generated from those variables.

This is important because the grid may need to be changed later without rewriting the workflow.

## 9. Sampling Philosophy for the First Dataset

At this stage, the purpose is to build a first surrogate that can reproduce the MATLAB model well over a reasonable local parameter domain.

So the first dataset should use a **task-specific range**, not the broadest possible global range.

Recommended sources for defining the first training domain:

1. `getFiberConstraints.m`
2. `Wool_92_params.txt`
3. `Wool_92_random_params.txt`
4. `Extracted_All_Params.txt`

The purpose of these sources is:

- identify physically valid limits
- identify realistic parameter magnitudes
- avoid training first on an unnecessarily broad and difficult parameter space

## 10. Role of UQ in This Version

For this first strategy version:

- do **not** use UQ to validate the MATLAB model first
- do **not** start from posterior propagation studies
- do **not** mix the surrogate task with the inversion task

Instead:

- assume `jcal_reflection.m` is the trusted teacher
- first solve the surrogate approximation problem

UQ and sensitivity analysis come only **after** the surrogate has been shown to reproduce the MATLAB curves well.

## 11. Recommended Step-by-Step Workflow

### Step 1: Freeze the first-version configuration

Define clearly:

- `fiberfolder = 'Wool'`
- initial `porosityfolder = '92'`
- support for `92-99`
- `freq_min = 100`
- `freq_max = 2000`
- `n_freq = 64`

This creates a stable baseline problem definition.

### Step 2: Reuse the current project logic for sample metadata

Follow the same logic already used in `Uncertainty_quantification.m`:

1. choose `fiberfolder`
2. choose `porosityfolder`
3. compute `phi`
4. call `getFluidProperties`
5. compute `h`
6. build the `airProperties` structure

This avoids introducing inconsistent assumptions between the current project and the surrogate workflow.

### Step 3: Define the first training domain

For the first practical dataset:

1. use `Wool`
2. start from porosity `92`
3. determine realistic bounds for the 5 JCAL material parameters
4. keep the domain narrow enough for a stable first surrogate

This stage is about defining the parameter space, not yet training the network.

### Step 4: Write a dedicated dataset-generation script

The first new implementation should be a MATLAB script whose only job is:

1. generate sampled parameter combinations
2. call `jcal_reflection`
3. compute `alpha`
4. store the dataset as input matrix `X` and output matrix `Y`

This step should happen before any DNN training code is written.

### Step 5: Build the first dataset

For each sample:

- inputs go into one row of `X`
- the 64-point absorption curve goes into one row of `Y`

So:

- `X` has size `N x 7`
- `Y` has size `N x 64`

where `N` is the number of generated training samples.

### Step 6: Preprocess the dataset

Before training:

- normalize or standardize inputs
- consider log scaling for parameters with large dynamic ranges
- keep preprocessing rules fixed and reusable

Likely candidates for log scaling:

- `sigma`
- `lambda`
- `lambda_prime`
- `k0_prime`

### Step 7: Train a first baseline surrogate

The first training model should be simple and robust:

- feedforward MLP
- multi-output regression
- output size `64`

The first aim is not model complexity.
The first aim is to demonstrate that the MATLAB mapping can be learned accurately.

### Step 8: Validate only against MATLAB

For this first version, validation means:

- compare surrogate outputs against MATLAB outputs
- evaluate test error
- compare randomly chosen curves
- inspect worst-case examples

The key question is:

“Can the surrogate reproduce the teacher model well enough?”

### Step 9: Expand only after the baseline works

Only after the baseline is successful should the workflow be extended toward:

- more porosities
- broader parameter ranges
- more samples
- alternative surrogate architectures
- uncertainty propagation
- sensitivity analysis

## 12. Minimum Viable First Implementation

The minimum viable first implementation is:

1. choose `fiberfolder = 'Wool'`
2. choose `porosityfolder = '92'`
3. get `phi` and `h` using the current project logic
4. define the 64-point frequency grid from `100` to `2000 Hz`
5. sample valid JCAL parameter sets
6. generate absorption curves with `jcal_reflection.m`
7. build the first dataset
8. train a first simple surrogate
9. verify that it reproduces MATLAB absorption curves well

If this works, the surrogate route is practically established.

## 13. Immediate Next Action

The immediate next implementation task should be:

- write a dedicated MATLAB dataset-generation script for the first surrogate dataset

That script should:

- follow the current `fiberfolder` / `porosityfolder` logic,
- obtain `h` using `getFluidProperties`,
- use configurable frequency-grid variables,
- and generate `X` / `Y` data from `jcal_reflection.m`.
