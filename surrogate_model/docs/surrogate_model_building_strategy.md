# Surrogate Model Building Strategy

## 1. Core Positioning

The practical modelling route for this project is:

1. Use the existing MATLAB/JCAL model as the reference forward model.
2. Use this forward model to generate a large training dataset.
3. Train a DNN surrogate model to approximate the input-output mapping of the MATLAB model.
4. Use the trained surrogate model for uncertainty propagation, sensitivity analysis, and reliability evaluation.

This means the current MATLAB model is **not** the final surrogate model itself.

It is better described as:

- a physics-based forward model,
- a fast analytical / semi-analytical model,
- and the "teacher model" for generating surrogate training data.

In this project, the actual surrogate model is the later **data-driven DNN/GP/PCE model** trained to emulate the MATLAB model.

## 2. Relationship Between the Existing Materials

### 2.1 Existing MATLAB model

The current MATLAB code already provides the most important physical forward mapping:

`material parameters -> reflection / absorption response`

The most important reusable file is:

- [jcal_reflection.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/jcal_reflection.m:1)

This file can already compute:

- reflection coefficient `R`,
- absorption coefficient `alpha`,
- and related acoustic quantities

from a given parameter set.

### 2.2 Existing uncertainty quantification workflow

The current scripts also provide:

- physically meaningful parameter bounds,
- realistic parameter combinations,
- posterior parameter samples,
- and a validated uncertainty-analysis workflow.

Useful files:

- [getFiberConstraints.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/getFiberConstraints.m:1)
- [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:1)
- `*_params.txt`
- `*_random_params.txt`

### 2.3 Role of the reference paper

The paper *Deterministic and Statistical Characterization of Rigid Frame Porous Materials from Impedance Tube Measurements* is mainly useful for:

- understanding the physical meaning of the JCAL parameters,
- understanding parameter uncertainty,
- understanding posterior correlation between parameters,
- and motivating why uncertainty propagation and statistical analysis are needed.

The paper is therefore more important as a **methodological guide** than as a direct DNN implementation guide.

## 3. Recommended Overall Workflow

The recommended workflow is:

1. Define the surrogate model input parameters and output quantities.
2. Define physically meaningful parameter bounds.
3. Sample the parameter space.
4. Call the MATLAB forward model repeatedly to generate training data.
5. Preprocess the dataset.
6. Train a DNN surrogate model.
7. Validate the surrogate against the MATLAB model.
8. Use the surrogate for Monte Carlo uncertainty propagation and sensitivity analysis.

## 4. Recommended Surrogate Definition

### 4.1 Recommended inputs

For the first practical version, the surrogate inputs should be:

- porosity `phi`
- airflow resistivity `sigma`
- tortuosity `alpha_infinity`
- viscous characteristic length `lambda`
- thermal characteristic length `lambda_prime`
- static thermal permeability `k0_prime`
- material thickness `h`

This gives the mapping:

`[phi, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, h] -> absorption curve`

### 4.2 Recommended outputs

The most useful first output choice is:

- the full absorption coefficient curve over a fixed frequency grid

That is:

`[input parameters] -> [alpha(f1), alpha(f2), ..., alpha(fm)]`

This is a **multi-output surrogate model**.

Reasons:

- it directly predicts the final engineering quantity of interest,
- it is easy to compare with the MATLAB model,
- it is convenient for uncertainty bands and Monte Carlo studies,
- and it avoids the extra complexity of learning complex-valued `R` first.

## 5. Why This Is Better Than Starting From COMSOL

Based on the supervisor discussion, COMSOL is not the practical main data source for surrogate training because:

- parameter sweeps are too slow,
- repeated simulations are hard to automate at the required scale,
- and the thesis would become blocked by simulation cost rather than modelling logic.

The MATLAB/JCAL model is therefore the practical high-value foundation for the surrogate study.

It is computationally cheap enough to generate many samples and physically interpretable enough to justify the training data.

## 6. What Can Be Reused Directly

### 6.1 Files and functions that can be reused directly

- `jcal_reflection.m`
  - use as the forward solver to generate surrogate training targets
- `getFiberConstraints.m`
  - use to define parameter bounds
- `Uncertainty_quantification.m`
  - use as reference for parameter organization, frequency selection, and output interpretation
- `*_params.txt`
  - use as representative parameter estimates
- `*_random_params.txt`
  - use as realistic posterior samples or as references for distribution ranges

### 6.2 Things that are less central for surrogate training

- `metropolis_hastings.m`
  - useful for inversion and posterior sampling, but not essential for first-stage surrogate training
- `log_likelihood.m`
  - useful for comparing model predictions with experiments, but not the main component of a forward surrogate

## 7. Step-by-Step Surrogate Construction Plan

### Step 1: Fix the input-output definition

The first surrogate should learn:

`[phi, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, h] -> alpha(f)`

on a fixed frequency grid.

For a first version, the frequency grid can be limited to:

- 100 Hz to 2000 Hz,
- with a moderate number of frequency points

This keeps the output dimension manageable.

### Step 2: Define parameter bounds

Parameter bounds should come from a combination of:

- `getFiberConstraints.m`,
- existing inferred parameter files,
- existing posterior random sample files,
- and practical material ranges already observed in the current project.

Two possible strategies:

- broad physical range: better generality, harder training
- task-specific range: easier training, more immediately useful

For the first working surrogate, a **task-specific range** is recommended.

### Step 3: Generate sampled parameter combinations

Use a design-of-experiments method such as:

- Latin Hypercube Sampling

This is preferred because it covers high-dimensional parameter space more evenly than naive random sampling.

Each sample is one row of parameter values.

### Step 4: Generate the dataset using the MATLAB forward model

For each sampled parameter set:

1. call `jcal_reflection`
2. compute `alpha`
3. store the output on the selected frequency grid

The dataset structure should be:

- input matrix `X`: `N x d`
- output matrix `Y`: `N x m`

where:

- `N` = number of samples
- `d` = number of input parameters
- `m` = number of frequency points

### Step 5: Preprocess the data

Before training:

- normalize or standardize all inputs
- consider `log10` transform for parameters spanning several orders of magnitude
- keep the same preprocessing for train/validation/test data

Recommended candidates for log scaling:

- `sigma`
- `lambda`
- `lambda_prime`
- `k0_prime`

Outputs `alpha` usually stay in `[0, 1]`, so they may not need strong transformation.

### Step 6: Split the dataset

Recommended split:

- training set: 70%
- validation set: 15%
- test set: 15%

The test set must remain fully untouched during training.

### Step 7: Train the first DNN surrogate

Recommended first architecture:

- feedforward multi-layer perceptron
- 3 to 5 hidden layers
- 64 to 256 neurons per layer
- ReLU or tanh activation
- output dimension equal to the number of selected frequency points

Recommended training configuration:

- loss: mean squared error
- optimizer: Adam
- early stopping
- learning-rate scheduling if needed

This first model should be kept simple and robust rather than large and complex.

### Step 8: Validate the surrogate

The surrogate must be validated against the MATLAB model, not just by looking at loss values.

Recommended validation criteria:

- low test MSE / MAE
- predicted absorption curves close to MATLAB curves
- peak position and trend reproduced correctly
- no obvious systematic bias
- acceptable accuracy across the full parameter range

Recommended plots:

- predicted vs true scatter plot
- random curve comparisons
- worst-case example curves
- mean error versus frequency

### Step 9: Use the surrogate for uncertainty propagation

Once trained, the surrogate can replace repeated forward evaluations of the MATLAB model.

Then:

1. define parameter distributions
2. generate many Monte Carlo samples
3. evaluate the surrogate rapidly
4. estimate PDFs, mean curves, confidence bands, and quantiles

This is one of the main reasons for building the surrogate in the first place.

### Step 10: Use the surrogate for sensitivity analysis

After training, the surrogate can also be used for:

- correlation-based analysis
- Sobol indices
- Morris screening
- reliability threshold studies

This makes large-scale statistical studies feasible.

## 8. Most Important Conceptual Distinction

The following distinction should always be kept clear:

- The MATLAB model is a **physics-based forward model**.
- The DNN is the **surrogate model**.

So the modelling chain is:

`material parameters -> MATLAB/JCAL forward model -> absorption data -> DNN surrogate`

not:

`material parameters -> COMSOL -> DNN surrogate`

at least for the current practical thesis route defined by the supervisor discussion.

## 9. Minimum Viable First Implementation

A realistic first implementation should be:

1. choose the input parameters
2. choose a fixed frequency grid
3. sample 500 to 1000 parameter sets
4. generate the corresponding absorption curves with `jcal_reflection`
5. train a small multi-output DNN
6. check whether it can accurately reproduce the MATLAB curves

If this works, the workflow is proven feasible.

Then the model can be extended with:

- larger datasets
- refined parameter ranges
- better surrogate architectures
- uncertainty propagation
- sensitivity analysis

## 10. Practical Thesis Narrative

The thesis can be framed with the following logic:

1. Porous material performance depends on uncertain non-acoustic parameters.
2. The current JCAL-based MATLAB model provides a fast and physically interpretable forward model.
3. Large-scale statistical studies still require many model evaluations.
4. A DNN surrogate is therefore introduced to emulate the MATLAB forward model.
5. The surrogate is trained on synthetically generated data from the physics-based model.
6. The surrogate is validated against the MATLAB reference model.
7. The validated surrogate is then used for efficient uncertainty propagation and sensitivity analysis.

## 11. Immediate Next Actions

The next concrete tasks should be:

1. finalize the surrogate input and output definition
2. write a dedicated MATLAB data-generation script based on `jcal_reflection`
3. generate a first small dataset
4. train a first baseline DNN model
5. validate whether the DNN can reproduce the absorption curves accurately

Only after that should the workflow be expanded toward full UQ and sensitivity analysis.
