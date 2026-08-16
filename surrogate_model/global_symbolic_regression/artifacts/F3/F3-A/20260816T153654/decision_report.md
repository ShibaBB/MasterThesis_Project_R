# Re Global SR F3-A Decision Report

- Stage decision: **PASS**
- Selected sigma candidate: `sigma_bandpass_1250_2100`
- Test rows accessed: `False`
- Next stage automatically authorized: `False`

## Decision basis

- sigma_bandpass_1250_2100 improves full validation RMSE.
- sigma_bandpass_1250_2100 improves the combined 1300-2000 Hz validation RMSE.
- All declared regional RMSE guardrails remain within the 10% limit.

Full metrics, regional guardrails, coefficients, figures, and the exact
training-only standardization are stored beside this report.
F3-B requires a new explicit user command even when F3-A passes.
