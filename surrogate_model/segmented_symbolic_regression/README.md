# Segmented SR For Reflection-Coefficient Components

The segmented PySR branch fits independent formulas for each target and
frequency segment:

```text
R_real = g_re_segment(..., f)
R_imag = g_im_segment(..., f)
```

With eight configured segments, the initial migration trains 16 models. Keep
the existing PySR search settings until both target pipelines run end to end.

## Implemented Contract

- load `y_re_symbolic` or `y_im_symbolic` through an explicit target selector;
- record target/run/split metadata in every artifact;
- select candidates on validation curves independently by target and segment;
- evaluate raw, unclipped component predictions on test curves;
- retain per-target boundary-discontinuity diagnostics;
- prevent Re/Im artifact collisions.

## Planned Artifacts

```text
artifacts/re/train/20260809_run1/
artifacts/re/eval/20260809_run1_c21/
artifacts/im/train/20260809_run1/
artifacts/im/eval/20260809_run1_c21/
```

Paired scalar generation and both target loaders were smoke-tested. A full
PySR search has not been launched and requires a configured Python/Julia PySR
environment. The old overview, strategy, and branch handoff describe the superseded target
workflow and are not authoritative. Use `../current_project_handoff.md`.
