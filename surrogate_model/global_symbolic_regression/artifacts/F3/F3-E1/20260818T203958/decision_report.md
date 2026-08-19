# Re Global SR F3-E1 Decision Report

- Stage decision: **STOP / NO PASS**
- Saved A median validation RMSE: `0.057845318`
- E1 median validation RMSE: `0.056030291`
- Saved A median curve-mean RMSE: `0.049486656`
- E1 median curve-mean RMSE: `0.048741020`
- Saved A median worst-curve RMSE: `0.147887364`
- E1 median worst-curve RMSE: `0.138451240`
- Saved A median 1300–2000 Hz RMSE: `0.031278497`
- E1 median 1300–2000 Hz RMSE: `0.031588864`
- Test rows accessed: `False`

## Predecessor guardrails

- 700–1000 Hz: E1 median `0.045525021`, saved A `0.044009890`, passed `True`
- 1300–1650 Hz: E1 median `0.026078502`, saved A `0.025116079`, passed `True`
- 1650–2000 Hz: E1 median `0.036271537`, saved A `0.036271537`, passed `True`

Only nested log/sqrt constraints changed in this small-budget screen.
This is not a formal Re model replacement decision.
A later stage requires a new explicit user command.
