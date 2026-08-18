# Re Global SR F3-C Decision Report

- Stage decision: **PASS**
- C0 median validation RMSE: `0.098598002`
- A median validation RMSE: `0.057845318`
- Formal Re F1 validation RMSE: `0.069978655`
- A median 1300–2000 Hz RMSE: `0.031278497`
- C0 median 1300–2000 Hz RMSE: `0.157573576`
- Formal Re F1 1300–2000 Hz RMSE: `0.091002824`
- Test rows accessed: `False`

## Guardrails

- 700–1000 Hz: A median `0.044009890`, F1 `0.045444911`, passed `True`
- 1300–1650 Hz: A median `0.025116079`, F1 `0.088077401`, passed `True`
- 1650–2000 Hz: A median `0.036271537`, F1 `0.093837089`, passed `True`

This is a small-budget representation decision, not a formal Re model replacement.
A later stage requires a new explicit user command.
