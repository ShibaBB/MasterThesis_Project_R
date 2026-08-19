# F2 Evaluation And Final Acceptance

## Re

All nine Re onset-envelope screening branches completed on validation data.
The best branch was `r1_c1800_t12`:

```text
F1 validation RMSE:           0.069979
Best F2-screen validation RMSE: 0.084468
Relative full-range change:   +20.70%
```

Its regional validation RMSE changes relative to F1 were:

```text
700-1000 Hz:   +58.30%
1300-1650 Hz:  +59.00%
1650-2000 Hz:  +47.57%
```

Every Re branch failed at least one declared 10 percent regional guardrail.
The Re F2 experiment therefore stopped after stage one; no width screen or
formal Re F2 test evaluation was run.

Final Re decision: **reject F2 and retain the Re F1 checkpoint**.

Detailed files:

```text
decisions/re/re_stage1_onset_comparison.csv
decisions/re/re_stage1_onset_blocked.json
```

## Im

The first-stage validation screen selected split `k0_prime` envelopes with a
900 Hz low-pass center and 0.10 log10-frequency transition. The second-stage
`lambda_prime` ablation produced:

```text
L0 validation RMSE: 0.063067  (passed all guardrails)
L1 validation RMSE: 0.081258  (rejected)
L2 validation RMSE: 0.088496  (rejected)
```

L0 was frozen and retrained with the formal budget. Full-validation selection
chose candidate 14, complexity 23.

```text
Metric              Im F1       Im F2       Relative change
Validation RMSE     0.074101    0.066753    -9.92%
Test RMSE           0.070212    0.063032    -10.23%
Test MAE            0.054466    0.048076    -11.73%
Test max abs error  0.316217    0.297937    -5.78%
Test R2             0.587141    0.667266    improved
```

All five declared formal validation guardrails passed. Raw test predictions
span `[-0.432702, 0.364151]`; no clipping or smoothing was applied.

Final Im decision: **accept Im F2 as the current Global SR Im checkpoint**.

Detailed files:

```text
decisions/im/im_stage1_k0_low_comparison.csv
decisions/im/im_stage1_k0_low_decision.json
decisions/im/im_stage2_lambda_ablation_comparison.csv
decisions/im/im_stage2_lambda_ablation_decision.json
decisions/im/formal_acceptance_decision.json
eval/im_formal/20260816_f2_im_f2_formal_v/
```

## Test-access audit

All 21 screening summaries record `test_rows_accessed=false` and contain no
test metrics. In the first formal Im execution, the legacy trainer and final
evaluator each read test after the envelope was frozen. Test results did not
influence envelope or candidate selection. The runner was subsequently fixed
so future formal training is validation-only and only the final evaluator
opens test. The formal acceptance JSON records the actual access count of two
for this run.
