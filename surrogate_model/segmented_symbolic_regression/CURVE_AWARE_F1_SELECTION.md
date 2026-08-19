# Curve-Aware F1 Candidate Reselection

This step reuses the completed F0 and F1 searches. It does not run PySR. Every
F1 Hall-of-Fame candidate is scored on validation curves against the phase-one
F0 winner for the same segment. F0 is retained unless an F1 candidate passes
every gate:

```text
overall validation RMSE             <= F0 * 1.02
median per-curve validation RMSE    <= F0 * 1.02
95th-percentile curve RMSE          <= F0 * 1.05
first-difference RMSE               <= F0 * 1.05
second-difference RMSE              <= F0 * 1.10
prediction range                    within teacher range + 10% span margin
mean predicted/teacher curve range  <= 1.25
mean excess turning points          <= F0 + 0.05 per curve
```

Passing candidates are ranked by validation RMSE, then second-difference RMSE,
then expression complexity. All segment choices are frozen before the test
split is evaluated.

Formal selections:

```text
Re F1: 100-700, 1300-1650, 3000-4000, 4000-4950 Hz
Re F0: all other segments

Im F1: 700-1000, 3000-4000 Hz
Im F0: all other segments
```

```text
Target/model          Validation RMSE   Test RMSE   Test R2
Re F0 hybrid          0.035577          0.034823    0.965604
Re curve-aware hybrid 0.034318          0.033009    0.969094
Im F0 hybrid          0.030429          0.030686    0.921142
Im curve-aware hybrid 0.027730          0.030104    0.924101
```

Relative to F0, selected Re validation segment changes are:

```text
Segment       value RMSE   first difference   second difference
100-700         -27.3%          -42.9%              -62.0%
1300-1650        +0.7%           -2.9%               -0.1%
3000-4000        -3.9%           -1.4%               +7.4%
4000-4950        -6.3%          -26.1%               -3.4%
```

Relative to F0, selected Im validation segment changes are:

```text
Segment       value RMSE   first difference   second difference
700-1000         -4.1%          -17.2%              -41.3%
3000-4000       -30.7%          -46.4%               -1.3%
```

The new hybrid is deliberately more conservative than the original F1. It
does not address discontinuities between independently fitted segments.

Implementation: `select_curve_aware_f1_candidates.py`.

