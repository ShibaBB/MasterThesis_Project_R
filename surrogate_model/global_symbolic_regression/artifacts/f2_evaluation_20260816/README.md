# Global SR F2 Evaluation Bundle

This directory consolidates all evaluation artifacts produced by the Re and Im
F2 experiment on 2026-08-16. The original artifact directories are retained so
that persisted training and evaluation paths remain valid.

## Directory layout

```text
evaluation_index.csv          one-row index for all 22 F2 eval folders
FINAL_ACCEPTANCE.md           evaluation summary and final checkpoint decision
eval/re_screen/               9 Re validation-only screening evaluations
eval/im_screen/               12 Im validation-only screening evaluations
eval/im_formal/               1 formal Im F2 validation/test evaluation
f1_validation_reference/      validation-only Re and Im F1 references
decisions/re/                 Re stage-one comparison and blocked decision
decisions/im/                 Im stage comparisons and formal acceptance decision
```

Folders ending in `_v` under `re_screen` and `im_screen` are validation-only
screening evaluations. They contain only `global_complexity_vs_rmse.png`
because test rows were intentionally inaccessible during screening. The formal
Im folder contains the complete scatter, frequency-error, and curve-comparison
figures.

Start with `FINAL_ACCEPTANCE.md`, then use `evaluation_index.csv` for all branch
metrics. For visual comparison, inspect the four figures in
`eval/im_formal/20260816_f2_im_f2_formal_v/figures`.
