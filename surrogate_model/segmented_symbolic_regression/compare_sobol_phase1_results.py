"""Compare matched phase-one branches using validation results only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


MATCHED_TRAINING_ARGS = (
    "dataset_run",
    "random_seed",
    "max_samples_per_segment",
    "niterations",
    "populations",
    "population_size",
    "maxsize",
    "model_selection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare all vs Sobol-subset branches without using test metrics."
    )
    parser.add_argument("--all-eval-dir", type=Path, required=True)
    parser.add_argument("--sobol-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update known comparison files in an existing comparison directory.",
    )
    return parser.parse_args()


def load_run(eval_dir: Path, expected_branch: str) -> dict[str, Any]:
    summary_path = eval_dir / "selected_combination_summary.json"
    segment_path = eval_dir / "selected_segment_metrics.csv"
    if not summary_path.exists() or not segment_path.exists():
        raise FileNotFoundError(f"Incomplete evaluation artifacts: {eval_dir}")
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
    all_metrics = pd.read_csv(segment_path)
    metrics = all_metrics[all_metrics["split"] == "validation"].copy()
    if len(metrics) != 8 or len(all_metrics[all_metrics["split"] == "test"]) != 8:
        raise ValueError(f"Expected eight validation segment rows: {segment_path}")
    return {
        "summary": summary,
        "training": training,
        "metrics": metrics,
        "all_segment_metrics": all_metrics,
    }


def validate_matched(all_run: dict[str, Any], sobol_run: dict[str, Any]) -> None:
    for key in ("dataset_run", "target", "shared_split_hash"):
        if all_run["summary"].get(key) != sobol_run["summary"].get(key):
            raise ValueError(f"A/B evaluation mismatch for {key}.")
    for key in MATCHED_TRAINING_ARGS:
        left = all_run["training"]["training_args"].get(key)
        right = sobol_run["training"]["training_args"].get(key)
        if left != right:
            raise ValueError(f"A/B training setting mismatch for {key}: {left!r} != {right!r}")
    if all_run["training"].get("search_contract") != sobol_run["training"].get(
        "search_contract"
    ):
        raise ValueError("A/B PySR operator/search contracts do not match.")


def aggregate_selected_metrics(
    selected_rows: pd.DataFrame, reference_overall: dict[str, float]
) -> dict[str, float | int]:
    n_rows = selected_rows["n_rows"].astype(int)
    total_rows = int(n_rows.sum())
    sse = float(((selected_rows["rmse"] ** 2) * n_rows).sum())
    reference_sse = float(reference_overall["rmse"] ** 2 * total_rows)
    if reference_overall["r2"] == 1.0:
        r2 = 1.0 if sse == 0.0 else float("-inf")
    else:
        total_sum_squares = reference_sse / (1.0 - reference_overall["r2"])
        r2 = 1.0 - sse / total_sum_squares
    return {
        "n_rows": total_rows,
        "rmse": float((sse / total_rows) ** 0.5),
        "mae": float((selected_rows["mae"] * n_rows).sum() / total_rows),
        "max_abs_error": float(selected_rows["max_abs_error"].max()),
        "r2": float(r2),
        "true_min": float(selected_rows["true_min"].min()),
        "true_max": float(selected_rows["true_max"].max()),
        "prediction_min": float(selected_rows["prediction_min"].min()),
        "prediction_max": float(selected_rows["prediction_max"].max()),
    }


def build_hybrid_metrics(
    all_run: dict[str, Any],
    sobol_run: dict[str, Any],
    winners: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    for split in ("validation", "test"):
        selected_rows: list[pd.Series] = []
        for segment_name, branch in winners.items():
            source = all_run if branch == "all" else sobol_run
            candidates = source["all_segment_metrics"]
            row = candidates[
                (candidates["split"] == split)
                & (candidates["segment_name"] == segment_name)
            ].iloc[0]
            selected_rows.append(row)
            rows.append(
                {
                    "split": split,
                    "segment_id": int(row["segment_id"]),
                    "segment_name": segment_name,
                    "selected_branch": branch,
                    "n_rows": int(row["n_rows"]),
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "max_abs_error": float(row["max_abs_error"]),
                    "r2_within_segment": float(row["r2"]),
                }
            )
        selected_df = pd.DataFrame(selected_rows)
        reference = (
            all_run["summary"]["validation_selection_metrics"]
            if split == "validation"
            else all_run["summary"]["overall_metrics"]
        )
        aggregates[split] = aggregate_selected_metrics(selected_df, reference)
    return pd.DataFrame(rows), aggregates


def main() -> None:
    args = parse_args()
    all_run = load_run(args.all_eval_dir, "all")
    sobol_run = load_run(args.sobol_eval_dir, "sobol_subset")
    validate_matched(all_run, sobol_run)

    columns = ["segment_id", "segment_name", "rmse", "mae", "max_abs_error", "r2"]
    comparison = all_run["metrics"][columns].merge(
        sobol_run["metrics"][columns],
        on=["segment_id", "segment_name"],
        suffixes=("_all", "_sobol_subset"),
        validate="one_to_one",
    )
    comparison["rmse_delta_subset_minus_all"] = (
        comparison["rmse_sobol_subset"] - comparison["rmse_all"]
    )
    comparison["rmse_relative_delta"] = (
        comparison["rmse_delta_subset_minus_all"] / comparison["rmse_all"]
    )
    comparison["validation_rmse_winner"] = comparison.apply(
        lambda row: "sobol_subset"
        if row["rmse_sobol_subset"] <= row["rmse_all"]
        else "all",
        axis=1,
    )

    if args.output_dir.exists() and not args.update_existing:
        raise FileExistsError(f"Comparison directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "validation_branch_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    winners = dict(zip(comparison["segment_name"], comparison["validation_rmse_winner"]))
    hybrid_rows, hybrid_metrics = build_hybrid_metrics(all_run, sobol_run, winners)
    hybrid_rows.to_csv(args.output_dir / "hybrid_selected_segment_metrics.csv", index=False)
    decision = {
        "schema_version": 1,
        "target": all_run["summary"]["target"],
        "dataset_run": all_run["summary"]["dataset_run"],
        "shared_split_hash": all_run["summary"]["shared_split_hash"],
        "decision_data": "validation only",
        "test_metrics_used_for_decision": False,
        "test_metrics_reported_after_validation_decision": True,
        "matched_training_args": {
            key: all_run["training"]["training_args"].get(key)
            for key in MATCHED_TRAINING_ARGS
        },
        "per_segment_validation_rmse_winner": winners,
        "hybrid_metrics": hybrid_metrics,
        "hybrid_segment_metrics_csv": str(
            args.output_dir / "hybrid_selected_segment_metrics.csv"
        ),
        "comparison_csv": str(comparison_path),
    }
    with (args.output_dir / "validation_branch_decision.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(decision, file, indent=2)
    print(comparison[["segment_name", "rmse_all", "rmse_sobol_subset", "validation_rmse_winner"]].to_string(index=False))
    print(f"Validation-only comparison: {comparison_path}")


if __name__ == "__main__":
    main()
