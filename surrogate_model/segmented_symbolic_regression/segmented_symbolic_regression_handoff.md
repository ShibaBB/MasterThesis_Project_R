# Segmented Symbolic Regression Handoff

This branch is awaiting migration to two independent targets:

```text
R_real = real(Reflect)
R_imag = imag(Reflect)
```

No segmented dataset, training run, or evaluation run currently exists in the
R repository. Existing Python and MATLAB code still assumes the superseded
single-target schema.

The authoritative migration plan is:

```text
surrogate_model/current_project_handoff.md
```

Branch-specific requirements:

1. paired scalar data containing `y_re_symbolic` and `y_im_symbolic`;
2. explicit `re|im` selector in generation, training, and evaluation;
3. eight independent formulas per target using unchanged initial PySR settings;
4. validation-only candidate selection and test-only final reporting;
5. raw component evaluation without clipping;
6. per-target boundary diagnostics;
7. short target-separated artifact paths.

Do not use the older strategy document as the active specification.
