# Symbolic Regression Strategy

## 1. Purpose

The purpose of this strategy is to shift the surrogate-model route from a purely black-box neural-network approximation toward a more interpretable model form.

The current MLP surrogate learns a mapping of the form

`Y = f_NN(X)`

where the learned function is represented by neural-network weights and activation functions.

Although this is useful as a predictive baseline, it is not easily interpretable in physical terms.

The new target is to construct a surrogate of the form

`alpha = g(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)`

where `g` is an explicit symbolic expression that can be written down, inspected, discussed, and compared with physical intuition.

## 2. High-Level Position

This strategy does **not** replace the current JCAL MATLAB model as the physical teacher model.

The modelling chain remains:

`material/sample settings -> JCAL MATLAB teacher model -> generated data -> surrogate`

The change is only in the final surrogate type:

- previous route: multi-output MLP surrogate
- new route: symbolic regression surrogate

The existing MLP should still be retained as a baseline reference.

Its role is now:

- provide a performance benchmark,
- provide a black-box accuracy upper reference for the current pipeline,
- and provide a comparison point for the symbolic model.

## 3. Main Recommendation

The symbolic regression model should be built as a **new modelling branch**, not by modifying the current MLP training logic.

This is the recommended approach because:

1. the MLP and symbolic regression solve the approximation problem in fundamentally different ways,
2. the MLP is based on hidden nonlinear weights, while symbolic regression is based on explicit formula search,
3. symbolic regression requires different data organization, different complexity control, and different model-selection criteria.

So the correct strategy is:

- keep the current data-generation and validation infrastructure,
- keep the current MLP as baseline,
- build a separate symbolic-regression workflow on top of the same teacher model.

## 4. Recommended First Problem Definition

### 4.1 Recommended target quantity

For the first symbolic-regression model, the target should be the absorption coefficient directly:

`alpha = g(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)`

This is preferred over learning intermediate physical quantities such as:

- dynamic density,
- dynamic bulk modulus,
- surface impedance,
- reflection coefficient,
- or complex-valued intermediate expressions.

### 4.2 Why direct alpha prediction is recommended

This choice is recommended because:

- `alpha` is the final engineering quantity of interest,
- the current surrogate pipeline already produces `alpha` reliably from the teacher model,
- symbolic regression on complex-valued quantities would be much harder to control,
- and a first symbolic model should prioritize interpretability and stability.

The goal is **not** to rediscover the full JCAL derivation.

The goal is to produce a compact symbolic approximation of the teacher-model response.

## 5. Recommended Data Representation

### 5.1 Why the current MLP dataset format is not enough

The current MLP workflow uses:

- `X : N x 7`
- `Y : N x 64`

This is convenient for multi-output neural-network training, but it is not the most natural format for symbolic regression.

In symbolic regression, the preferred form is a scalar mapping:

`input variables -> one scalar output`

### 5.2 Recommended symbolic-regression dataset format

The dataset should therefore be reorganized into a frequency-expanded scalar form:

- `X_symbolic = [phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f]`
- `y_symbolic = alpha`

That means:

- each original absorption curve is split into many scalar samples,
- each frequency point becomes one row,
- and frequency `f` becomes an explicit input variable.

This is the most natural route if the final goal is a formula of the form

`alpha = g(..., f)`

## 6. Recommended Scope for the First Symbolic Model

To keep the problem controlled and interpretable, the first symbolic-regression study should be narrower than the current general surrogate ambition.

### 6.1 Material scope

Use only:

- `Wool`

Do not mix materials in the first symbolic model.

### 6.2 Porosity scope

Use only:

- porosity folder `92`
- numerical porosity `phi = 0.92`

The broader `92-99` flexibility can remain in the code infrastructure, but the first symbolic formula should not be forced to cover everything at once.

### 6.3 Frequency scope

Keep:

- `100 Hz` to `2000 Hz`

The first attempt can use the current frequency range directly.

However, the symbolic-regression workflow should remain open to frequency segmentation if one global expression becomes too complex.

## 7. Recommended Formula Structure Strategy

### 7.1 Recommended starting structure

The symbolic-regression model should be built directly as a **frequency-segmented model**.

That means:

- one material,
- one porosity,
- multiple symbolic formulas,
- each formula responsible for one frequency range.

This is recommended from the start rather than as a fallback.

### 7.2 Why segmentation should be used immediately

Frequency segmentation is recommended immediately because:

- the absorption response can behave differently across frequency regions,
- forcing one global expression usually leads to a longer and less interpretable formula,
- and segmented formulas are more likely to remain compact while keeping acceptable accuracy.

For this thesis route, several short interpretable expressions are more valuable than one large formula that is difficult to explain physically.

### 7.3 Recommended segmentation structure

The first symbolic-regression workflow should split the frequency axis into a few ranges, for example:

- low-frequency range,
- mid-frequency range,
- high-frequency range

and fit one symbolic model per range.

### 7.4 Recommended initial segmentation

For the current first-version scope of `100 Hz` to `2000 Hz`, the recommended initial segmentation is:

- Segment 1: `100-700 Hz`
- Segment 2: `700-1300 Hz`
- Segment 3: `1300-2000 Hz`

This is a recommended starting point, not a permanently fixed rule.

The reason for this initial choice is pragmatic:

- it divides the full frequency range into three manageable regions,
- it is simple enough for a first symbolic-regression implementation,
- and it gives the workflow enough flexibility to capture different curve behaviors without making the segmentation too fine.

### 7.5 Requirement for configurable segmentation

Although the above three-segment scheme is recommended as the first implementation, the workflow should be written so that frequency segmentation remains easy to modify later.

That means the implementation should allow:

- changing the number of segments,
- changing the lower and upper bounds of each segment,
- and changing the segmentation rule itself without rewriting the whole modelling pipeline.

So the segmentation should be treated as a configurable modelling choice, not a hard-coded physical law.

### 7.6 Recommended future segmentation rules

The workflow should be compatible with several possible future segmentation rules, for example:

- equal-width frequency segmentation
- manually defined engineering ranges
- segmentation guided by curve-shape changes
- segmentation guided by surrogate error concentration

For now, the recommended default is still the manually defined three-segment structure:

- `100-700 Hz`
- `700-1300 Hz`
- `1300-2000 Hz`

So the recommended structure is:

1. define frequency segments,
2. build one symbolic formula per segment,
3. compare segment-wise accuracy and formula complexity.

## 8. Recommended Operator Library

The first symbolic-regression model should use a conservative and physically interpretable operator set.

### 8.1 Recommended first operator set

Recommended operators:

- `+`
- `-`
- `*`
- `/`
- `log`
- `sqrt`
- simple powers such as:
  - `x^2`
  - `x^-1`
  - `x^0.5`

### 8.2 Operators not recommended at the start

Do not begin with highly flexible operators such as:

- `sin`
- `cos`
- `tan`
- arbitrary free exponents everywhere
- deep nested exponentials
- ad hoc piecewise logical switching

These often improve fitting freedom at the cost of physical interpretability.

For this thesis route, interpretability matters more than purely opportunistic symbolic flexibility.

## 9. Recommended Trade-Off Principle

The symbolic model should not be judged only by raw error.

It should be judged by both:

- predictive accuracy,
- and symbolic simplicity / interpretability.

### 9.1 Role of the MLP baseline

The MLP baseline should serve as:

- a high-accuracy benchmark,
- not necessarily a model to beat.

The symbolic model does not need to outperform the MLP.

A realistic and valuable result is:

- MLP gives the best predictive accuracy,
- symbolic regression gives slightly lower accuracy,
- but much better interpretability.

That comparison is scientifically meaningful and useful for the thesis.

### 9.2 Recommended model-selection principle

Model selection should follow a Pareto-style balance:

- lower prediction error,
- shorter formula,
- fewer operators,
- better physical readability,
- and more stable behavior across the target domain.

The preferred symbolic model is therefore not simply the numerically best one.

It is the best compromise between:

- accuracy,
- compactness,
- and interpretability.

## 10. What Can Be Reused from the Current Pipeline

The following parts of the current surrogate workflow should be reused directly:

- `jcal_reflection.m`
  - as the teacher model
- `getFluidProperties.m`
  - to obtain `h` and air properties consistently
- `getFiberConstraints.m`
  - to define physically meaningful sampling bounds
- `data_generation/generate_teacher_dataset.m`
  - as the basis for symbolic dataset generation logic
- `data_generation/inspect_teacher_dataset.m`
  - as the basis for data sanity checking
- the current MLP baseline results
  - as a benchmark for comparison

The following part should **not** be reused as the symbolic model itself:

- `MLP/mlp_train_surrogate_baseline.m`

That script is useful as a baseline workflow reference, but not as the direct foundation of symbolic regression.

## 11. What Must Change Relative to the MLP Workflow

The following elements must be redesigned for symbolic regression:

1. the dataset layout
   - from multi-output curve prediction to scalar-frequency samples
2. the training logic
   - from gradient-based neural training to symbolic formula search
3. the model-selection logic
   - from validation loss only to accuracy-complexity trade-off
4. the interpretation workflow
   - from hidden representation to explicit analytical expression

## 12. Recommended Immediate Next Questions

Before implementation, the following design questions should be fixed clearly:

1. which symbolic-regression tool or framework will be used,
2. how the frequency segments should be defined,
3. which operator set is allowed,
4. how formula complexity will be measured,
5. what error level is acceptable relative to the current MLP baseline.

## 13. Recommended Immediate Next Action

The next step should **not** be to start coding symbolic regression immediately.

The next step should be to define the symbolic-regression problem precisely in implementation terms:

1. freeze the target scope:
   - `Wool`
   - porosity `92`
   - `100-2000 Hz`
2. define the scalar symbolic dataset format:
   - `[phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f] -> alpha`
3. define the allowed symbolic operator library
4. define the complexity-versus-accuracy selection rule
5. choose the implementation framework

Only after that should the symbolic-regression workflow be built.

## 14. Summary of the Recommended Route

The recommended symbolic-regression route is:

1. keep the current MLP as a baseline,
2. do not modify the MLP into a symbolic model,
3. build a new symbolic-regression branch,
4. fit `alpha` directly rather than intermediate complex quantities,
5. use scalar frequency-expanded samples,
6. start with `Wool` and porosity `92`,
7. define the symbolic surrogate directly as frequency-segmented formulas,
8. begin with the recommended three-segment split:
   - `100-700 Hz`
   - `700-1300 Hz`
   - `1300-2000 Hz`
9. keep the segmentation rule configurable for later refinement,
10. use a conservative operator library,
11. evaluate the symbolic model against both error and interpretability.
