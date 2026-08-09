# Global SR For Reflection-Coefficient Components

The global PySR branch fits two independent full-range equations:

```text
R_real = g_re(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
R_imag = g_im(phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, f)
```

Keep the current PySR search settings for the first migration. The dataset
loader must select `y_re_symbolic` or `y_im_symbolic` explicitly and verify
target/run/split metadata.

## Evaluation

Candidate selection is validation-only and independent for Re and Im.
Final test reporting uses raw RMSE, MAE, maximum absolute error, R2, ranges,
frequency error, scatter, and curve comparisons. Predictions are evaluated
raw; there is no `[0,1]` clipping.

## Planned Artifacts

```text
artifacts/re/train/20260809_run1/
artifacts/re/eval/20260809_run1_c21/
artifacts/im/train/20260809_run1/
artifacts/im/eval/20260809_run1_c21/
```

Keep names short; store detailed search parameters in training metadata.
Paired scalar generation and both target loaders were smoke-tested, but no
full dataset or trained PySR model currently exists. Follow
`../current_project_handoff.md` for the implementation order.
