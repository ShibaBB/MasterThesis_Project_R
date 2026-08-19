"""Compare all-base and Sobol-modulated Global SR using validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from compare_global_sobol_phase1_results import (
    MATCHED_TRAINING_ARGS,
    load_run,
    metric_row,
    validate_matched,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full-range all-base vs smooth Sobol-modulated Global SR."
    )
    parser.add_argument("--all-eval-dir", type=Path, required=True)
    parser.add_argument("--modulated-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_run = load_run(args.all_eval_dir, "all")
    modulated_run = load_run(args.modulated_eval_dir, "sobol_modulated")
    validate_matched(all_run, modulated_run)

    rows = pd.DataFrame(
        [metric_row("all", all_run), metric_row("sobol_modulated", modulated_run)]
    )
    all_validation_rmse = float(
        rows.loc[rows["branch"] == "all", "validation_rmse"].iloc[0]
    )
    modulated_validation_rmse = float(
        rows.loc[rows["branch"] == "sobol_modulated", "validation_rmse"].iloc[0]
    )
    winner = (
        "sobol_modulated"
        if modulated_validation_rmse <= all_validation_rmse
        else "all"
    )
    rows["validation_rmse_delta_vs_all"] = rows["validation_rmse"] - all_validation_rmse
    rows["validation_rmse_relative_delta_vs_all"] = (
        rows["validation_rmse"] / all_validation_rmse - 1.0
    )
    rows["selected_by_validation_rmse"] = rows["branch"] == winner

    if args.output_dir.exists():
        raise FileExistsError(f"Comparison directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    comparison_path = args.output_dir / "validation_branch_comparison.csv"
    rows.to_csv(comparison_path, index=False)
    winner_run = all_run if winner == "all" else modulated_run
    decision = {
        "schema_version": 1,
        "experiment": "smooth teacher-Sobol parameter frequency modulation",
        "target": all_run["summary"]["target"],
        "dataset_run": all_run["summary"]["dataset_run"],
        "shared_split_hash": all_run["summary"]["shared_split_hash"],
        "frequency_range_hz": all_run["summary"]["global_bounds_hz"],
        "decision_data": "full-range validation rows only",
        "test_metrics_used_for_decision": False,
        "test_metrics_reported_after_validation_decision": True,
        "selected_branch": winner,
        "selected_eval_dir": winner_run["eval_dir"],
        "selected_validation_metrics": winner_run["summary"]["validation_selection_metrics"],
        "selected_test_metrics": winner_run["summary"]["overall_metrics"],
        "matched_training_args": {
            key: all_run["training"]["training_args"].get(key)
            for key in MATCHED_TRAINING_ARGS
        },
        "comparison_csv": str(comparison_path),
    }
    with (args.output_dir / "validation_branch_decision.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(decision, file, indent=2)
    print(
        rows[
            ["branch", "validation_rmse", "test_rmse", "selected_by_validation_rmse"]
        ].to_string(index=False)
    )
    print(f"Validation-selected whole-range branch: {winner}")


if __name__ == "__main__":
    main()
