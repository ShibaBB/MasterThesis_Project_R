"""Compare frozen phase-one F0 segments with local-frequency F1 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare existing F0 hybrid against F1.")
    parser.add_argument("--target", required=True, choices=("re", "im"))
    parser.add_argument("--f0-all-eval-dir", type=Path, required=True)
    parser.add_argument("--f0-subset-eval-dir", type=Path, required=True)
    parser.add_argument("--phase1-decision", type=Path, required=True)
    parser.add_argument("--f1-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Replace comparison files in an existing output directory.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def selected_f0_rows(
    all_frame: pd.DataFrame,
    subset_frame: pd.DataFrame,
    winners: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for segment_name, branch in winners.items():
        source = all_frame if branch == "all" else subset_frame
        selected = source[source["segment_name"] == segment_name].copy()
        selected["f0_source_branch"] = branch
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def aggregate_value_metrics(
    rows: pd.DataFrame, reference: dict[str, float]
) -> dict[str, float | int]:
    n_rows = rows["n_rows_selected"].astype(int)
    total_rows = int(n_rows.sum())
    sse = float(((rows["rmse_selected"] ** 2) * n_rows).sum())
    reference_sse = float(reference["rmse"] ** 2 * total_rows)
    total_sum_squares = reference_sse / (1.0 - reference["r2"])
    return {
        "n_rows": total_rows,
        "rmse": float((sse / total_rows) ** 0.5),
        "mae": float((rows["mae_selected"] * n_rows).sum() / total_rows),
        "max_abs_error": float(rows["max_abs_error_selected"].max()),
        "r2": float(1.0 - sse / total_sum_squares),
        "true_min": float(rows["true_min_f0"].min()),
        "true_max": float(rows["true_max_f0"].max()),
        "prediction_min": float(rows["prediction_min_selected"].min()),
        "prediction_max": float(rows["prediction_max_selected"].max()),
    }


def main() -> None:
    args = parse_args()
    decision = read_json(args.phase1_decision)
    if decision["target"] != args.target:
        raise ValueError("Phase-one decision target mismatch.")
    winners = decision["per_segment_validation_rmse_winner"]

    f0_all_value = pd.read_csv(args.f0_all_eval_dir / "selected_segment_metrics.csv")
    f0_subset_value = pd.read_csv(
        args.f0_subset_eval_dir / "selected_segment_metrics.csv"
    )
    f0_all_shape = pd.read_csv(
        args.f0_all_eval_dir / "selected_segment_shape_metrics.csv"
    )
    f0_subset_shape = pd.read_csv(
        args.f0_subset_eval_dir / "selected_segment_shape_metrics.csv"
    )
    f1_value = pd.read_csv(args.f1_eval_dir / "selected_segment_metrics.csv")
    f1_shape = pd.read_csv(args.f1_eval_dir / "selected_segment_shape_metrics.csv")

    f0_value = selected_f0_rows(f0_all_value, f0_subset_value, winners)
    f0_shape = selected_f0_rows(f0_all_shape, f0_subset_shape, winners)
    comparison = f0_value.merge(
        f1_value,
        on=["split", "segment_id", "segment_name"],
        suffixes=("_f0", "_f1"),
        validate="one_to_one",
    ).merge(
        f0_shape,
        on=["split", "segment_id", "segment_name", "f0_source_branch"],
        validate="one_to_one",
    ).merge(
        f1_shape,
        on=["split", "segment_id", "segment_name"],
        suffixes=("_shape_f0", "_shape_f1"),
        validate="one_to_one",
    )
    comparison["rmse_relative_delta_f1_minus_f0"] = (
        comparison["rmse_f1"] - comparison["rmse_f0"]
    ) / comparison["rmse_f0"]
    comparison["first_difference_relative_delta"] = (
        comparison["first_difference_rmse_shape_f1"]
        - comparison["first_difference_rmse_shape_f0"]
    ) / comparison["first_difference_rmse_shape_f0"]
    comparison["second_difference_relative_delta"] = (
        comparison["second_difference_rmse_shape_f1"]
        - comparison["second_difference_rmse_shape_f0"]
    ) / comparison["second_difference_rmse_shape_f0"]
    validation_rows = comparison[comparison["split"] == "validation"]
    value_winners = {
        row.segment_name: ("f1" if row.rmse_f1 <= row.rmse_f0 else "f0")
        for row in validation_rows.itertuples()
    }
    comparison["validation_rmse_winner"] = comparison["segment_name"].map(
        value_winners
    )
    for metric in (
        "n_rows",
        "rmse",
        "mae",
        "max_abs_error",
        "prediction_min",
        "prediction_max",
    ):
        comparison[f"{metric}_selected"] = comparison.apply(
            lambda row, name=metric: row[f"{name}_{row['validation_rmse_winner']}"],
            axis=1,
        )

    selected_candidates = pd.read_csv(args.f1_eval_dir / "selected_candidates.csv")
    selected_candidates["f1_uses_local_frequency"] = selected_candidates[
        "equation"
    ].str.contains("local_t", regex=False)
    selected_candidates["f1_uses_any_frequency"] = selected_candidates[
        "equation"
    ].str.contains(r"local_t|log10_f", regex=True)
    comparison = comparison.merge(
        selected_candidates[
            ["segment_name", "f1_uses_local_frequency", "f1_uses_any_frequency"]
        ],
        on="segment_name",
        validate="many_to_one",
    )

    f1_summary = read_json(args.f1_eval_dir / "selected_combination_summary.json")
    selected_hybrid_metrics = {}
    for split in ("validation", "test"):
        reference = (
            decision["hybrid_metrics"]["validation"]
            if split == "validation"
            else decision["hybrid_metrics"]["test"]
        )
        selected_hybrid_metrics[split] = aggregate_value_metrics(
            comparison[comparison["split"] == split], reference
        )
    summary = {
        "schema_version": 1,
        "target": args.target,
        "dataset_run": f1_summary["dataset_run"],
        "shared_split_hash": f1_summary["shared_split_hash"],
        "f0_retrained": False,
        "f0_hybrid_validation_metrics": decision["hybrid_metrics"]["validation"],
        "f0_hybrid_test_metrics": decision["hybrid_metrics"]["test"],
        "f1_validation_metrics": f1_summary["validation_selection_metrics"],
        "f1_test_metrics": f1_summary["overall_metrics"],
        "candidate_selection": "validation RMSE only",
        "shape_metrics_used_for_selection": False,
        "frequency_dependency_forced": False,
        "per_segment_validation_rmse_winner": value_winners,
        "validation_rmse_selected_hybrid_metrics": selected_hybrid_metrics,
        "shape_acceptance_decision": "not yet made",
    }
    args.output_dir.mkdir(parents=True, exist_ok=args.update_existing)
    comparison.to_csv(args.output_dir / "f0_vs_f1_segment_comparison.csv", index=False)
    with (args.output_dir / "f0_vs_f1_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(
        comparison[comparison["split"] == "validation"]
        [[
            "segment_name",
            "f0_source_branch",
            "rmse_f0",
            "rmse_f1",
            "rmse_relative_delta_f1_minus_f0",
            "first_difference_relative_delta",
            "second_difference_relative_delta",
            "f1_uses_local_frequency",
            "f1_uses_any_frequency",
            "validation_rmse_winner",
        ]]
        .to_string(index=False)
    )
    print(f"F0/F1 comparison: {args.output_dir}")


if __name__ == "__main__":
    main()
