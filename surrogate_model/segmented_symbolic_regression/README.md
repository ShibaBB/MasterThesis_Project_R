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

Formal full-run Re and Im models now exist. The next active work is the matched
teacher-Sobol-guided feature experiment documented in
[`SOBOL_PHASE1.md`](SOBOL_PHASE1.md). The old overview, strategy, and branch
handoff describe the superseded target workflow and are not authoritative. Use
`../current_project_handoff.md` for the current project state.

The completed local-frequency representation experiment is documented in
[`FREQUENCY_F1.md`](FREQUENCY_F1.md). It reused F0 and added local polynomial
frequency features without forcing their use. Re improved materially; the full
Im F1 model did not generalize, and shape acceptance remains undecided.

The validation-only Hall-of-Fame reselection and its fixed gates are documented
in [`CURVE_AWARE_F1_SELECTION.md`](CURVE_AWARE_F1_SELECTION.md). It retains F0
for any segment where no existing F1 candidate passes every value and shape
gate; neither F0 nor F1 is retrained.

Further segmented optimization is currently paused. The preserved checkpoint
is the curve-aware F0/F1 hybrid. Per-segment grammar tuning and boundary
continuity remain explicit deferred tasks; active work has moved to
Sobol-guided Global SR.
