# PySR Acceleration And Seed-Replay Benchmark

Date: 2026-08-16

This validation-only benchmark tests the proposed PySR runtime settings on the
current Re F1 feature representation. The MLP code and architecture are not
part of this experiment and were not changed.

## Frozen Search Contract

- dataset: `run1`
- target: Re
- feature branch: `sobol_modulated`
- seed: 42
- search budget: 10 iterations, 6 populations, population size 40, max size 24
- training rows: all rows in the shared training-curve partition
- batch size: 4096
- candidate selection: full shared validation-curve partition only
- test rows: inaccessible during all four runs

Plan C uses PySR multithreading with 8 Julia threads, batching, and turbo.
Because PySR does not promise deterministic replay for multithreaded search,
two clean same-seed replicas were required. Plan B uses serial deterministic
search with the same batching and turbo settings.

## Results

| Plan | Replica | Fit time (s) | Validation RMSE | Validation R2 | Complexity | HOF SHA-256 prefix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C | 1 | 54.101 | 0.099731 | 0.724819 | 17 | `1d64d0a8897f` |
| C | 2 | 20.608 | 0.090421 | 0.773799 | 23 | `bbea447c4ff1` |
| B | 1 | 23.244 | 0.097903 | 0.734814 | 24 | `7ed8cf946bac` |
| B | 2 | 22.433 | 0.097903 | 0.734814 | 24 | `7ed8cf946bac` |

The first C timing includes Julia backend compilation/warm-up and is therefore
not a clean steady-state speed measurement. More importantly, C failed every
exact-replay check: the selected equation, validation metrics, and complete
Hall-of-Fame CSV hash all differed. Its two validation RMSE values differ by
`0.00931038` despite the same seed.

B passed every exact-replay check. The selected equation, validation metrics,
and complete Hall-of-Fame CSV hash are identical across both replicas.

## Acceptance Decision

Reject C for formal project runs. Accept B as the PySR runtime configuration
for subsequent experiments that use acceleration flags:

```text
parallelism=serial
deterministic=true
batching=true
batch_size=4096
turbo=true
JULIA_NUM_THREADS=1
```

This benchmark establishes reproducibility, not a model-quality improvement
and not a speedup factor versus the historical full-data serial configuration.
Future model comparisons must keep the same validation/test protocol and use
independent formal search budgets.

Machine-readable evidence is stored in
`artifacts/performance_benchmarks/20260816_perf_seed42/benchmark_decision.json`
and `benchmark_runs.csv`.
