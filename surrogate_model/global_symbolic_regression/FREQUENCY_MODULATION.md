# Global SR Smooth Frequency Modulation

This experiment retains all five varying material parameters and `log10_f` in
one Global SR model over 100-4950 Hz. It appends exactly one smooth
frequency-modulated feature per material parameter. It does not train separate
frequency models and introduces no hard boundaries.

For a transformed material feature `x_p`, the added terminal is:

```text
x_p__sobol_mod = x_p * m_p(log10_f)
```

The base `x_p` remains available. Therefore the physical material parameter
does not become frequency-dependent; only its possible contribution to the
global equation receives a smooth frequency envelope.

## Teacher-Sobol rationale

The envelope locations come from the teacher N=1024 pointwise and segment ST
diagnostics. Numerical ST values are not model inputs.

- Re: `sigma` peaks near 1437 Hz; `alpha_infinity` peaks near 3422 Hz;
  `lambda` grows toward high frequency; `lambda_prime` has low- and
  high-frequency lobes; `k0_prime` is strong at low frequency, weak in the
  middle, and rises again at high frequency.
- Im: `sigma` is strong at low frequency and again over roughly 2-4 kHz;
  `alpha_infinity` and `lambda` have mid/high lobes; `lambda_prime` remains
  weak but has low/high structure; `k0_prime` is primarily low-frequency with
  a smaller highest-frequency lobe.

Exact Gaussian/logistic component centers, widths, weights, and normalization
constants are persisted in `training_metadata.json` and implemented in
`global_frequency_modulation.py`.

```text
Target  parameter          smooth envelope components
Re      sigma              Gaussian(center=1437 Hz, log10 width=0.366)
Re      alpha_infinity     Gaussian(center=3422 Hz, log10 width=0.220)
Re      lambda             high-pass(center=2200 Hz, transition=0.140)
Re      lambda_prime       0.30 low-pass(700 Hz) + 1.00 high-pass(3800 Hz)
Re      k0_prime           1.00 low-pass(850 Hz) + 0.65 high-pass(3400 Hz)
Im      sigma              1.00 low-pass(1200 Hz) + 0.90 Gaussian(3000 Hz, 0.250)
Im      alpha_infinity     0.75 Gaussian(1550 Hz, 0.160) + 1.00 high-pass(3700 Hz)
Im      lambda             0.70 Gaussian(1500 Hz, 0.180) + 1.00 high-pass(3900 Hz)
Im      lambda_prime       0.35 low-pass(700 Hz) + 1.00 high-pass(4000 Hz)
Im      k0_prime           1.00 low-pass(750 Hz) + 0.40 high-pass(3900 Hz)
```

## Formal contract

The modulated Re and Im models use the same run1 rows, split, seed 42, 100
iterations, 12 populations, population size 80, max complexity 24, operators,
and validation-only candidate selection as the existing all-base control.
Each target selects `all` or `sobol_modulated` using complete 100-4950 Hz
validation RMSE. Test metrics are reported only after that decision.

## Formal run1 results

The matched experiment completed on 2026-08-16. Both targets selected
`sobol_modulated` using validation RMSE only.

```text
Target/branch       Validation RMSE   Test RMSE   Test R2
Re all              0.081930          0.080918    0.814274
Re sobol_modulated  0.069979          0.069470    0.863110
Im all              0.085952          0.079741    0.467469
Im sobol_modulated  0.074101          0.070212    0.587141
```

Relative to the matched all-base control, modulation improves Re validation
RMSE by 14.59 percent and test RMSE by 14.15 percent. It improves Im
validation RMSE by 13.79 percent and test RMSE by 11.95 percent. The selected
Re equation uses the `alpha_infinity` modulation terminal. The selected Im
equation uses the `lambda_prime` and `k0_prime` modulation terminals.

```text
artifacts/re/comparison/20260816_run1_sobol_modulated
artifacts/im/comparison/20260816_run1_sobol_modulated
```
