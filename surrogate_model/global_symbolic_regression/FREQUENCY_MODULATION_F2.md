# Global SR Target-Specific Frequency Modulation F2

The F2 experiment was executed on 2026-08-16. It keeps one full-range Global
SR equation per target, all five transformed material base features, and
`log10_f`. Re and Im use independent envelope screens and validation-only
decisions.

## Screening contract

Every screening branch used the complete `run1` training rows, shared split,
seed 42, 20 iterations, 6 populations, population size 40, max complexity 24,
and the unchanged `+ - * / log sqrt` grammar. Candidate selection used the
complete validation partition. The screening evaluator did not subset or
predict test rows. In total, 21 screening runs completed with zero recorded
test-access violations.

The declared 10 percent regional RMSE guardrails were evaluated against the F1
validation checkpoint. Regional bias and pointwise validation RMSE/bias were
also persisted for diagnosis.

## Re result: stopped after stage one

All nine onset branches completed. No branch passed the three regional
guardrails, so no Re envelope was frozen and the width screen and formal run
were not started. The best full-range screening branch was:

```text
onset center:       1800 Hz
onset transition:  0.12 log10-frequency
high lobe:          Gaussian(3422 Hz, width 0.22)
validation RMSE:    0.084468
F1 validation RMSE: 0.069979
```

Its regional RMSE changes relative to F1 were +58.30 percent at 700-1000 Hz,
+59.00 percent at 1300-1650 Hz, and +47.57 percent at 1650-2000 Hz. Proceeding
would therefore require a new, explicit experiment design rather than silently
waiving the frozen guardrails. Re retains the F1 checkpoint.

## Im result: F2 accepted

The 3x3 `k0_prime` low-pass screen selected center 900 Hz and transition 0.10.
With that split fixed, the `lambda_prime` ablation selected L0:

```text
k0_prime low:       low-pass(900 Hz, transition 0.10)
k0_prime high:      high-pass(3900 Hz, transition 0.10)
lambda_prime L0:    0.35 low-pass(700 Hz) + 1.00 high-pass(4000 Hz)
```

The stage-two screening validation RMSE values were L0 `0.063067`, L1
`0.081258`, and L2 `0.088496`. Only L0 passed every declared regional
guardrail.

The frozen L0 specification was retrained with 100 iterations, 12 populations,
population size 80, and max complexity 24. Full-validation candidate selection
chose candidate 14 with complexity 23.

```text
Model       Validation RMSE   Test RMSE   Test MAE   Test max error   Test R2
Im F1       0.074101          0.070212    0.054466   0.316217         0.587141
Im F2       0.066753          0.063032    0.048076   0.297937         0.667266
```

F2 improves validation RMSE by 9.92 percent and test RMSE by 10.23 percent.
All five formal validation guardrails pass; the largest regression is a 6.67
percent RMSE increase at 4000-4950 Hz, within the declared 10 percent limit.
Predictions remain raw and unclipped, with test range
`[-0.432702, 0.364151]`.

## Artifacts and test-access audit

```text
Re screen: artifacts/f2_manifests/20260816_f2
Im run:    artifacts/f2_manifests/20260816_f2_im
Im train:  artifacts/im/train/20260816_f2_im_f2_formal
Im eval:   artifacts/im/eval/20260816_f2_im_f2_formal_v
```

The 21 screening summaries all record `test_rows_accessed=false` and contain no
test metrics. In this first F2 execution, the frozen Im formal training command
used the legacy behavior that reported a training-selected candidate on test,
and the evaluator then reported the validation-selected candidate on test.
Neither test result influenced envelope or equation selection, but the test
partition was read twice after freezing. The runner has been corrected so
future formal training remains validation-only and only the final evaluator
opens test. The acceptance JSON records the actual access count of two for this
run rather than claiming a single access.
