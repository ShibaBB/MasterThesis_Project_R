# Re Global SR F3-F1 Formal Validation Decision

- Predeclared validation gate: **PASS**
- F3-F2 readiness: **HOLD**
- Formal Re F1 validation RMSE: `0.069978655`
- Formal Re F3 candidate validation RMSE: `0.051734604`
- Formal Re F1 curve-mean RMSE: `0.062263782`
- Formal Re F3 candidate curve-mean RMSE: `0.044269534`
- Formal Re F1 worst-curve RMSE: `0.169273679`
- Formal Re F3 candidate worst-curve RMSE: `0.131737754`
- Formal Re F1 1300–2000 Hz RMSE: `0.091002824`
- Formal Re F3 candidate 1300–2000 Hz RMSE: `0.033620714`
- Selected residual equation: `(z_sigma__f3_mid * ((0.052617513 / sqrt(z_sigma__f3_mid * z_sigma__f3_mid)) + -0.3566766)) / ((sqrt(z_sigma__f3_mid * z_sigma__f3_mid) * -1.6001033) + -1.4250554)`
- Selected complexity: `19`
- Test rows accessed: `False`

## Formal Re F1 guardrails

- 700–1000 Hz: F3 `0.046770068`, F1 `0.045444911`, passed `True`
- 1300–1650 Hz: F3 `0.028336037`, F1 `0.088077401`, passed `True`
- 1650–2000 Hz: F3 `0.038180805`, F1 `0.093837089`, passed `True`

## Post-selection formula-domain audit

- Passed: `False`
- Value at `z_sigma__f3_mid = 0`: `nan`
- Left limit: `52617513/1425055400`
- Right limit: `-52617513/1425055400`
- The selected formula is held because it is non-finite and discontinuous at a valid terminal value.

This is validation-only confirmation. It does not replace formal Re F1.
The test partition remains sealed pending explicit formula-safety resolution.
