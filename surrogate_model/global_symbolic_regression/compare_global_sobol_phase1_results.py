"""Choose one matched full-range Global SR branch using validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


MATCHED_TRAINING_ARGS = (
    "dataset_run", "random_seed", "max_samples", "niterations", "populations",
    "population_size", "maxsize", "model_selection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare whole-range all vs Sobol-subset Global SR branches."
    )
    parser.add_argument("--all-eval-dir", type=Path, required=True)
    parser.add_argument("--sobol-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_run(eval_dir: Path, expected_branch: str) -> dict[str, Any]:
    summary_path = eval_dir / "selected_global_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Incomplete Global evaluation artifacts: {eval_dir}")
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    branch = summary.get("feature_policy", {}).get("branch")
    if branch != expected_branch:
        raise ValueError(
            f"Expected branch {expected_branch!r}, found {branch!r}: {eval_dir}"
        )
    training_dir = Path(summary["training_dir"])
    with (training_dir / "training_metadata.json").open("r", encoding="utf-8") as file:
        training = json.load(file)
    return {"summary": summary, "training": training, "eval_dir": str(eval_dir)}


def validate_matched(all_run: dict[str, Any], sobol_run: dict[str, Any]) -> None:
    for key in ("dataset_run", "target", "shared_split_hash", "global_bounds_hz"):
        if all_run["summary"].get(key) != sobol_run["summary"].get(key):
            raise ValueError(f"A/B evaluation mismatch for {key}.")
    for key in MATCHED_TRAINING_ARGS:
        left = all_run["training"]["training_args"].get(key)
        right = sobol_run["training"]["training_args"].get(key)
        if left != right:
            raise ValueError(f"A/B training setting mismatch for {key}: {left!r} != {right!r}")
    if all_run["training"].get("search_contract") != sobol_run["training"].get("search_contract"):
        raise ValueError("A/B PySR operator/search contracts do not match.")


def metric_row(branch: str, run: dict[str, Any]) -> dict[str, Any]:
    summary = run["summary"]
    validation = summary["validation_selection_metrics"]
    test = summary["overall_metrics"]
    return {
        "branch": branch,
        "validation_rmse": validation["rmse"],
        "validation_mae": validation["mae"],
        "validation_r2": validation["r2"],
        "test_rmse": test["rmse"],
        "test_mae": test["mae"],
        "test_max_abs_error": test["max_abs_error"],
        "test_r2": test["r2"],
        "eval_dir": run["eval_dir"],
    }


def main() -> None:
    args = parse_args()
    all_run = load_run(args.all_eval_dir, "all")
    sobol_run = load_run(args.sobol_eval_dir, "sobol_subset")
    validate_matched(all_run, sobol_run)
    rows = pd.DataFrame([
        metric_row("all", all_run), metric_row("sobol_subset", sobol_run)
    ])
    all_validation_rmse = float(
        rows.loc[rows["branch"] == "all", "validation_rmse"].iloc[0]
    )
    subset_validation_rmse = float(
        rows.loc[rows["branch"] == "sobol_subset", "validation_rmse"].iloc[0]
    )
    winner = (
        "sobol_subset"
        if subset_validation_rmse <= all_validation_rmse
        else "all"
    )
    rows["selected_by_validation_rmse"] = rows["branch"] == winner
    if args.output_dir.exists():
        raise FileExistsError(f"Comparison directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    comparison_path = args.output_dir / "validation_branch_comparison.csv"
    rows.to_csv(comparison_path, index=False)
    winner_run = all_run if winner == "all" else sobol_run
    decision = {
        "schema_version": 1,
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
    with (args.output_dir / "validation_branch_decision.json").open("w", encoding="utf-8") as file:
        json.dump(decision, file, indent=2)
    print(rows[["branch", "validation_rmse", "test_rmse", "selected_by_validation_rmse"]].to_string(index=False))
    print(f"Validation-selected whole-range branch: {winner}")


if __name__ == "__main__":
    main()
