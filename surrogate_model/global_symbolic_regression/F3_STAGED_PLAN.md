# Re Global SR F3: Staged Execution Of The Seven Proposals

Date: 2026-08-16

This plan is a staged implementation of the seven previously proposed Re F3
ideas. It does not replace them with a different optimization plan. Their
execution order follows their dependencies rather than their original list
numbers.

## Frozen Contract For Every Stage

- Re F1 remains the accepted baseline: validation RMSE `0.069979`, test RMSE
  `0.069470`.
- Im F2 and the MLP architecture remain unchanged.
- The shared curve split, raw targets, training-only transforms and full
  100-4950 Hz validation curves remain fixed.
- Intermediate stages cannot access test rows.
- PySR uses accepted plan B: serial, deterministic, batching with batch size
  4096, turbo, one Julia thread.
- Regional guardrails include 700-1000, 1300-1650 and 1650-2000 Hz. No region
  may worsen by more than 10 percent relative to the declared control.
- Every stage writes its own manifest, metrics, figures and decision report,
  then stops. A later stage starts only after a new explicit user command.

## Stage F3-A: Cheap Linear Residual Diagnosis

Corresponds to proposal 4 and provides evidence for proposal 2. No PySR search
is run.

1. Reconstruct the frozen Re F1 predictions on training and validation curves.
2. Define `delta_Re = teacher - Re_F1`.
3. Standardize `log10_sigma` using training-set statistics only to obtain
   `z_sigma`.
4. Fit small ridge models on training residuals using these predeclared terms:
   - zero correction control;
   - `z_sigma * Gaussian(1450 Hz)`;
   - `z_sigma * Gaussian(1750 Hz)`;
   - both 1450 and 1750 Hz terms with independently fitted coefficients;
   - a smooth 1250-2100 Hz band-pass term.
5. Only if sigma shows a validation gain, add a small diagnostic containing
   `z_lambda` and `z_alpha` terms. This is still ridge, not PySR.

Evaluation: full validation RMSE/MAE/R2, curve-mean and worst-curve RMSE,
regional RMSE/bias and full-curve overlays. The stage passes only if a sigma
local correction improves full-range validation RMSE and 1300-2000 Hz without
breaking a guardrail.

Stop after the report. Next command, if accepted: `执行 F3-B`.

## Stage F3-B: Freeze The Sigma Local-Modulation Specification

Corresponds to proposal 2.

Using only F3-A training/validation evidence, freeze:

- the training-only center and scale used by `z_sigma`;
- whether the representation needs the 1450 Hz bump, 1750 Hz bump, both, or
  the smooth 1250-2100 Hz band-pass;
- exact centers, widths and analytic envelope definitions;
- atomic terminal names and feature order.

This stage implements and verifies the transform, serialization and replay of
the selected terminals. It does not launch the multi-branch PySR screen. A
small transform-level validation report must demonstrate finite values,
train-only normalization and exact reload/replay.

Stop after the report. Next command, if accepted: `执行 F3-C`.

## Stage F3-C: Fixed-F1 Residual PySR Pilot

Corresponds to proposal 1 and applies the fair-screen rules from proposal 5.

The model is

```text
Re_F3 = Re_F1 + delta_Re_PySR
delta_Re_PySR ~= teacher - Re_F1
```

Run the following matched 20-iteration, 6-population, population-size-40,
maximum-complexity-24 branches using three predeclared seeds:

- `C0`: original Re F1 representation trained at the same small budget;
- `A`: fixed Re F1 plus a residual PySR model using the frozen sigma-local
  atomic terminals from F3-B.

Compare the median validation metrics across the three seeds. F1 is an exact
fallback because a zero residual returns the accepted model. Advance only if
branch A beats the same-budget C0 median, improves the target region and passes
the guardrails. Do not compare a small-budget equation as if it were a formal
replacement for the 100/12/80 F1 model.

Stop after the report. Next command, if accepted: `执行 F3-D1`.

## Stage F3-D: Add Local Material Interactions Incrementally

Corresponds to proposal 3 and continues to apply proposal 5. Each substage is
a separate execution and requires a separate user command. All branches use
the same 20/6/40/24 budget and the same three seeds as F3-C.

### F3-D1

Compare branch `B = A + z_lambda * g_mid` against saved branch A results.
Stop, report and wait.

### F3-D2

Only if authorized, compare
`C = B + z_alpha * g_mid` against the accepted predecessor. Stop and wait.

### F3-D3

Only if authorized, compare
`D = C + z_sigma * z_lambda * g_mid` against the accepted predecessor. Stop
and wait.

At every substage, retain the simpler predecessor unless the new branch lowers
the three-seed median full validation RMSE, improves curve/regional behavior
and passes all guardrails. Do not add `lambda_prime` or `k0_prime` local
terminals in F3-D.

After the last useful interaction decision, the next command is either
`执行 F3-E` for controlled search tuning or `执行 F3-F` for formal confirmation.

## Stage F3-E: Controlled Formula-Search Adjustments

Corresponds to proposal 6. This stage is conditional: run it only if the
residual representation is useful but appears search-limited. Never change
several search factors in one comparison.

The local frequency bumps are already atomic terminals from F3-B. Remaining
experiments are separate substages, each using a matched control, the same
three seeds, full validation selection and a stop/report gate:

1. `F3-E1`: constrain nested `log`/`sqrt` expressions.
2. `F3-E2`: test maximum complexity 28 versus 24 as a single A/B change.
3. `F3-E3`: use a curve/frequency-balanced training subset for search speed;
   select only on complete validation curves.
4. `F3-E4`: test a smooth 1300-2000 Hz training-loss weight, first 1.5 and only
   then 2.0 if separately authorized; final validation metrics remain
   unweighted.

Increasing complexity is not the first F3 action. Any accepted E substage
becomes the new frozen search configuration; rejected changes are removed.
Each substage requires a command such as `执行 F3-E1` and stops afterward.

## Stage F3-F: Fair Formal Confirmation And One-Time Test

Corresponds to the formal part of proposal 5.

### F3-F1: formal validation-only run

Freeze the winning representation and search configuration. Train with seed
42, 100 iterations, 12 populations, population size 80 and maximum complexity
24, unless F3-E2 was explicitly accepted. Select the equation using complete
validation curves and compare it with formal Re F1. Stop and report without
opening test.

### F3-F2: final test, separately authorized

Only after an explicit `执行 F3-F2` command and a passing F3-F1 decision, open
the test partition once. Report raw full-range, curve-level, frequency-wise and
regional metrics. Accept Re F3 only if the frozen acceptance rules pass;
otherwise retain Re F1.

## Stage F3-G: PCA/SVD Frequency-Basis Fallback

Corresponds to proposal 7 and is not part of the default single-equation path.
Run it only if the user explicitly chooses it after the single residual PySR
formula has stalled.

1. `F3-G1`: learn `mean_curve(f)` and a small number of continuous frequency
   bases from training curves only; use validation reconstruction to choose a
   compact rank and report the attainable error before symbolic regression.
2. `F3-G2`: fit a small PySR formula for each material-dependent coefficient
   `c_k(x)` and evaluate the complete analytic combination on validation.
3. `F3-G3`: after a separate command, formally freeze and test the basis model
   under the same one-time-test rule.

Each G substage stops and waits. This branch changes the interpretation from
one compact formula to a small set of material-coefficient formulas plus fixed
continuous frequency bases, so it requires an explicit user decision.

## Traceability To The Original Seven Proposals

| Original proposal | Execution location |
| --- | --- |
| 1. Fixed F1 plus residual PySR | F3-C |
| 2. Move modulation toward local sigma terms | F3-A and F3-B |
| 3. Add sigma/lambda/alpha interactions incrementally | F3-D1 through F3-D3 |
| 4. Cheap linear diagnosis first | F3-A |
| 5. Same-budget control, three seeds, then formal run | F3-C through F3-F |
| 6. Controlled formula-search changes | F3-E1 through F3-E4 |
| 7. PCA/SVD frequency-basis fallback | F3-G1 through F3-G3 |

The immediate next action is only F3-A. Nothing after it is implied or
automatically authorized.
