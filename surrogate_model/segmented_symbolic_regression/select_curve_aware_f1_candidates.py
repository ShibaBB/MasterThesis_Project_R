"""Select existing F1 equations using validation-only curve-shape gates.

This script never trains a model.  For each segment it compares every existing
F1 Hall-of-Fame equation with the phase-one F0 winner.  F0 is retained unless
an F1 candidate passes all value, per-curve, derivative, range, and turning-
point gates.  The test split is evaluated only after all choices are frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_segmented_symbolic_candidates import (
    boundary_transition_diagnostics,
    evaluate_segment_candidates,
    make_pysr_variable_names,
    plot_curve_comparisons,
    plot_error_vs_frequency,
    plot_selected_scatter,
    predict_expression,
    read_equations,
    read_symbolic_dataset,
    regression_metrics,
    selected_segment_metrics,
    selected_segment_shape_metrics,
)
from frequency_feature_engineering import apply_local_frequency_features
from segmented_run_paths import default_dataset_run, resolve_dataset_file
from shared_split_utils import (
    resolve_and_load_shared_split,
    source_curve_mask,
    subset_symbolic_data,
    validate_training_split,
)
from sobol_feature_policy import build_feature_policy, validate_stored_feature_policy
from symbolic_feature_transform import (
    apply_feature_transform,
    identity_feature_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only curve-aware reselection of existing F1 candidates."
    )
    parser.add_argument("--target", required=True, choices=("re", "im"))
    parser.add_argument("--dataset-run", default=default_dataset_run())
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--f1-training-dir", type=Path, required=True)
    parser.add_argument("--f0-all-training-dir", type=Path, required=True)
    parser.add_argument("--f0-subset-training-dir", type=Path, required=True)
    parser.add_argument("--f0-all-eval-dir", type=Path, required=True)
    parser.add_argument("--f0-subset-eval-dir", type=Path, required=True)
    parser.add_argument("--phase1-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rmse-tolerance", type=float, default=0.02)
    parser.add_argument("--median-curve-rmse-tolerance", type=float, default=0.02)
    parser.add_argument("--p95-curve-rmse-tolerance", type=float, default=0.05)
    parser.add_argument("--first-difference-tolerance", type=float, default=0.05)
    parser.add_argument("--second-difference-tolerance", type=float, default=0.10)
    parser.add_argument("--range-margin-fraction", type=float, default=0.10)
    parser.add_argument("--mean-range-ratio-limit", type=float, default=1.25)
    parser.add_argument("--turning-excess-tolerance", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--max-plot-points", type=int, default=6000)
    parser.add_argument("--num-curves", type=int, default=6)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_model_data(
    raw_data: dict[str, Any], training_dir: Path, shared_split: dict[str, Any],
    target: str, dataset_run: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_training_split(
        training_dir, shared_split, expected_target=target,
        expected_dataset_run=dataset_run,
    )
    with (training_dir / "training_metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    data = copy.deepcopy(raw_data)
    frequency_index = data["feature_names"].index("f")
    data["frequency_hz"] = data["X"][:, frequency_index].copy()
    original_names = list(data["feature_names"])
    transform = metadata.get("feature_transform") or identity_feature_transform(
        original_names
    )
    data["X"] = apply_feature_transform(data["X"], original_names, transform)
    data["feature_names"] = transform["transformed_feature_names"]
    frequency_spec = metadata.get("frequency_feature_engineering")
    if frequency_spec is not None:
        data["X"], data["feature_names"] = apply_local_frequency_features(
            data["X"], data["feature_names"], data["frequency_hz"],
            data["segment_index"], data["segment_names"], frequency_spec,
        )
    stored_policy = metadata.get("feature_policy")
    if stored_policy is None:
        policy = build_feature_policy(
            target, "all", data["feature_names"], data["segment_names"]
        )
    else:
        policy = validate_stored_feature_policy(
            stored_policy, target, data["feature_names"], data["segment_names"]
        )
    return data, policy


def selected_indices(eval_dir: Path) -> dict[str, int]:
    frame = pd.read_csv(eval_dir / "selected_candidates.csv")
    return {
        str(row.segment_name): int(row.candidate_index)
        for row in frame.itertuples()
    }


def candidate_prediction(
    data: dict[str, Any], training_dir: Path, policy: dict[str, Any],
    segment_id: int, candidate_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    segment_name = data["segment_names"][segment_id - 1]
    equations = read_equations(training_dir, segment_name)
    equation_row = equations[equations["candidate_index"] == candidate_index]
    if len(equation_row) != 1:
        raise ValueError(
            f"Candidate {candidate_index} not found for {segment_name}: {training_dir}"
        )
    row = equation_row.iloc[0]
    mask = data["segment_index"] == segment_id
    feature_spec = policy["per_segment"][segment_name]
    indices = feature_spec["feature_indices_0based"]
    prediction = predict_expression(
        str(row["expression_for_eval"]), data["X"][mask][:, indices],
        make_pysr_variable_names(feature_spec["feature_names"]),
    )
    return prediction, {
        "candidate_index": candidate_index,
        "complexity": int(row["complexity"]),
        "equation": str(row["equation"]),
        "expression_for_eval": str(row["expression_for_eval"]),
    }


def count_turning_points(values: np.ndarray, scale: float) -> int:
    differences = np.diff(values)
    tolerance = max(1e-12, 0.02 * scale / max(values.size - 1, 1))
    signs = np.sign(differences[np.abs(differences) > tolerance])
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def curve_metrics(
    data: dict[str, Any], segment_id: int, prediction: np.ndarray,
) -> dict[str, float]:
    segment_mask = data["segment_index"] == segment_id
    y_segment = data["y"][segment_mask]
    frequencies = data["frequency_hz"][segment_mask]
    curve_ids = data["source_curve_index"][segment_mask]
    point_metrics = regression_metrics(y_segment, prediction)
    curve_rmse: list[float] = []
    first_true: list[np.ndarray] = []
    first_pred: list[np.ndarray] = []
    second_true: list[np.ndarray] = []
    second_pred: list[np.ndarray] = []
    true_ranges: list[float] = []
    prediction_ranges: list[float] = []
    turning_excess: list[float] = []

    for curve_id in np.unique(curve_ids):
        local = curve_ids == curve_id
        order = np.argsort(frequencies[local])
        truth = y_segment[local][order]
        pred = prediction[local][order]
        curve_rmse.append(float(np.sqrt(np.mean((pred - truth) ** 2))))
        first_true.append(np.diff(truth))
        first_pred.append(np.diff(pred))
        second_true.append(np.diff(truth, n=2))
        second_pred.append(np.diff(pred, n=2))
        truth_range = float(np.ptp(truth))
        true_ranges.append(truth_range)
        prediction_ranges.append(float(np.ptp(pred)))
        turning_excess.append(
            float(max(0, count_turning_points(pred, truth_range)
                         - count_turning_points(truth, truth_range)))
        )

    first_error = np.concatenate(first_pred) - np.concatenate(first_true)
    second_error = np.concatenate(second_pred) - np.concatenate(second_true)
    mean_true_range = float(np.mean(true_ranges))
    return {
        **point_metrics,
        "median_curve_rmse": float(np.median(curve_rmse)),
        "p95_curve_rmse": float(np.quantile(curve_rmse, 0.95)),
        "first_difference_rmse": float(np.sqrt(np.mean(first_error ** 2))),
        "second_difference_rmse": float(np.sqrt(np.mean(second_error ** 2))),
        "mean_true_curve_range": mean_true_range,
        "mean_prediction_curve_range": float(np.mean(prediction_ranges)),
        "mean_prediction_to_true_range_ratio": (
            float(np.mean(prediction_ranges)) / mean_true_range
            if mean_true_range > 0 else math.inf
        ),
        "mean_excess_turning_points": float(np.mean(turning_excess)),
    }


def apply_gates(
    candidate: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace,
) -> dict[str, Any]:
    teacher_span = baseline["true_max"] - baseline["true_min"]
    margin = args.range_margin_fraction * teacher_span
    gates = {
        "gate_rmse": candidate["rmse"] <= baseline["rmse"] * (1 + args.rmse_tolerance),
        "gate_median_curve_rmse": candidate["median_curve_rmse"]
        <= baseline["median_curve_rmse"] * (1 + args.median_curve_rmse_tolerance),
        "gate_p95_curve_rmse": candidate["p95_curve_rmse"]
        <= baseline["p95_curve_rmse"] * (1 + args.p95_curve_rmse_tolerance),
        "gate_first_difference": candidate["first_difference_rmse"]
        <= baseline["first_difference_rmse"] * (1 + args.first_difference_tolerance),
        "gate_second_difference": candidate["second_difference_rmse"]
        <= baseline["second_difference_rmse"] * (1 + args.second_difference_tolerance),
        "gate_prediction_bounds": candidate["prediction_min"] >= baseline["true_min"] - margin
        and candidate["prediction_max"] <= baseline["true_max"] + margin,
        "gate_mean_curve_range": candidate["mean_prediction_to_true_range_ratio"]
        <= args.mean_range_ratio_limit,
        "gate_excess_turning_points": candidate["mean_excess_turning_points"]
        <= baseline["mean_excess_turning_points"] + args.turning_excess_tolerance,
    }
    gates["passes_all_gates"] = bool(all(gates.values()))
    gates["failed_gates"] = ";".join(
        name.removeprefix("gate_") for name, passed in gates.items()
        if name.startswith("gate_") and not passed
    )
    return gates


def plot_gate_tradeoff(frame: pd.DataFrame, output_dir: Path, target: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (segment_name, group) in zip(axes.flat, frame.groupby("segment_name", sort=False)):
        f1 = group[group["source"] == "f1"]
        colors = np.where(f1["passes_all_gates"], "tab:green", "tab:red")
        ax.scatter(f1["rmse"], f1["second_difference_rmse"], c=colors, alpha=0.75)
        baseline = group[group["source"] == "f0"].iloc[0]
        ax.scatter([baseline["rmse"]], [baseline["second_difference_rmse"]],
                   marker="*", s=130, color="black", label="F0")
        ax.set_title(segment_name)
        ax.set_xlabel("validation RMSE")
        ax.set_ylabel("2nd-difference RMSE")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{target.upper()} curve-aware gates: green=pass, red=reject")
    fig.savefig(output_dir / "candidate_gate_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir()

    dataset_file, dataset_run = resolve_dataset_file(args.dataset_run, args.dataset_file)
    raw_data = read_symbolic_dataset(dataset_file, args.target, dataset_run)
    shared_split = resolve_and_load_shared_split(
        dataset_run, args.split_file,
        int(np.unique(raw_data["source_curve_index"]).size),
    )
    decision = read_json(args.phase1_decision)
    if decision["target"] != args.target:
        raise ValueError("Phase-one target mismatch.")
    branch_winners = decision["per_segment_validation_rmse_winner"]

    specifications = {
        "f1": (args.f1_training_dir, None),
        "all": (args.f0_all_training_dir, args.f0_all_eval_dir),
        "sobol_subset": (args.f0_subset_training_dir, args.f0_subset_eval_dir),
    }
    contexts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    selected_f0_indices: dict[str, dict[str, int]] = {}
    for name, (training_dir, eval_dir) in specifications.items():
        contexts[name] = prepare_model_data(
            raw_data, training_dir, shared_split, args.target, dataset_run
        )
        if eval_dir is not None:
            selected_f0_indices[name] = selected_indices(eval_dir)

    validation_mask = source_curve_mask(
        raw_data["source_curve_index"], shared_split, "validation"
    )
    validation_contexts = {
        name: (subset_symbolic_data(data, validation_mask), policy)
        for name, (data, policy) in contexts.items()
    }

    rows: list[dict[str, Any]] = []
    choices: dict[str, dict[str, Any]] = {}
    segment_names = raw_data["segment_names"]
    for segment_id, segment_name in enumerate(segment_names, start=1):
        f0_branch = branch_winners[segment_name]
        f0_data, f0_policy = validation_contexts[f0_branch]
        f0_index = selected_f0_indices[f0_branch][segment_name]
        f0_prediction, f0_equation = candidate_prediction(
            f0_data, specifications[f0_branch][0], f0_policy, segment_id, f0_index
        )
        baseline = curve_metrics(f0_data, segment_id, f0_prediction)
        baseline_row = {
            "segment_id": segment_id, "segment_name": segment_name,
            "source": "f0", "source_branch": f0_branch,
            **f0_equation, **baseline,
        }
        baseline_row.update({
            "passes_all_gates": True, "failed_gates": "",
        })
        rows.append(baseline_row)

        f1_data, f1_policy = validation_contexts["f1"]
        f1_frame, f1_predictions = evaluate_segment_candidates(
            f1_data, args.f1_training_dir, f1_policy, segment_id
        )
        passing: list[dict[str, Any]] = []
        for candidate in f1_frame.itertuples(index=False):
            candidate_dict = candidate._asdict()
            if candidate.eval_status != "ok":
                row = {
                    "segment_id": segment_id, "segment_name": segment_name,
                    "source": "f1", "source_branch": "frequency_enhanced_f1",
                    **candidate_dict, "passes_all_gates": False,
                    "failed_gates": "evaluation",
                }
                rows.append(row)
                continue
            diagnostics = curve_metrics(
                f1_data, segment_id, f1_predictions[int(candidate.candidate_index)]
            )
            gates = apply_gates(diagnostics, baseline, args)
            row = {
                "segment_id": segment_id, "segment_name": segment_name,
                "source": "f1", "source_branch": "frequency_enhanced_f1",
                "candidate_index": int(candidate.candidate_index),
                "complexity": int(candidate.complexity),
                "equation": candidate.equation,
                "expression_for_eval": candidate.expression_for_eval,
                **diagnostics, **gates,
            }
            rows.append(row)
            if gates["passes_all_gates"]:
                passing.append(row)

        if passing:
            winner = min(
                passing,
                key=lambda row: (row["rmse"], row["second_difference_rmse"], row["complexity"]),
            )
            choices[segment_name] = {
                "source": "f1", "source_branch": "frequency_enhanced_f1",
                "candidate_index": winner["candidate_index"],
                "complexity": winner["complexity"], "equation": winner["equation"],
                "num_passing_f1_candidates": len(passing),
            }
        else:
            choices[segment_name] = {
                "source": "f0", "source_branch": f0_branch,
                **f0_equation, "num_passing_f1_candidates": 0,
            }

    candidate_frame = pd.DataFrame(rows)
    candidate_frame.to_csv(args.output_dir / "candidate_curve_metrics.csv", index=False)
    plot_gate_tradeoff(candidate_frame, figures_dir, args.target)
    pd.DataFrame([
        {"segment_name": name, **choice} for name, choice in choices.items()
    ]).to_csv(args.output_dir / "selected_candidates.csv", index=False)

    def assembled_prediction(split: str) -> tuple[dict[str, Any], np.ndarray]:
        split_mask = source_curve_mask(raw_data["source_curve_index"], shared_split, split)
        split_contexts = {
            name: (subset_symbolic_data(data, split_mask), policy)
            for name, (data, policy) in contexts.items()
        }
        reference_data = split_contexts["f1"][0]
        prediction = np.full(reference_data["y"].shape, np.nan)
        for segment_id, segment_name in enumerate(segment_names, start=1):
            choice = choices[segment_name]
            context_name = "f1" if choice["source"] == "f1" else choice["source_branch"]
            model_data, model_policy = split_contexts[context_name]
            values, _ = candidate_prediction(
                model_data, specifications[context_name][0], model_policy,
                segment_id, int(choice["candidate_index"]),
            )
            mask = reference_data["segment_index"] == segment_id
            prediction[mask] = values
        if not np.all(np.isfinite(prediction)):
            raise ValueError(f"Non-finite assembled {split} predictions.")
        return reference_data, prediction

    validation_data, validation_prediction = assembled_prediction("validation")
    # Choices are now frozen.  Only at this point is the test split evaluated.
    test_data, test_prediction = assembled_prediction("test")
    segment_metrics = pd.concat([
        selected_segment_metrics(validation_data, validation_prediction, "validation"),
        selected_segment_metrics(test_data, test_prediction, "test"),
    ], ignore_index=True)
    shape_metrics = pd.concat([
        selected_segment_shape_metrics(validation_data, validation_prediction, "validation"),
        selected_segment_shape_metrics(test_data, test_prediction, "test"),
    ], ignore_index=True)
    segment_metrics.to_csv(args.output_dir / "selected_segment_metrics.csv", index=False)
    shape_metrics.to_csv(args.output_dir / "selected_segment_shape_metrics.csv", index=False)
    boundary = boundary_transition_diagnostics(test_data, test_prediction)
    boundary.to_csv(args.output_dir / "boundary_transition_metrics.csv", index=False)

    summary = {
        "schema_version": 1,
        "purpose": "validation-only curve-aware reselection of existing F1 candidates",
        "target": args.target,
        "dataset_run": dataset_run,
        "dataset_file": str(dataset_file),
        "shared_split_file": shared_split["split_file"],
        "shared_split_hash": shared_split["split_hash"],
        "f0_retrained": False,
        "f1_retrained": False,
        "candidate_selection_split": "validation",
        "final_evaluation_split": "test",
        "gate_tolerances": {
            "rmse": args.rmse_tolerance,
            "median_curve_rmse": args.median_curve_rmse_tolerance,
            "p95_curve_rmse": args.p95_curve_rmse_tolerance,
            "first_difference": args.first_difference_tolerance,
            "second_difference": args.second_difference_tolerance,
            "prediction_range_margin_fraction": args.range_margin_fraction,
            "mean_curve_range_ratio_limit": args.mean_range_ratio_limit,
            "mean_excess_turning_points": args.turning_excess_tolerance,
        },
        "fallback": "retain phase-one F0 segment when no F1 candidate passes every gate",
        "selected_candidates": choices,
        "validation_metrics": regression_metrics(validation_data["y"], validation_prediction),
        "test_metrics": regression_metrics(test_data["y"], test_prediction),
        "selected_segment_metrics": segment_metrics.to_dict(orient="records"),
        "selected_segment_shape_metrics": shape_metrics.to_dict(orient="records"),
        "boundary_transition_diagnostics": boundary.to_dict(orient="records"),
    }
    with (args.output_dir / "curve_aware_selection_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    plot_selected_scatter(
        test_data["y"], test_prediction, figures_dir, args.max_plot_points,
        args.random_seed, test_data["target_name"],
    )
    plot_error_vs_frequency(
        test_data["frequency_hz"], test_prediction - test_data["y"], figures_dir,
        args.max_plot_points, args.random_seed, test_data["target_name"],
    )
    plot_curve_comparisons(
        test_data, test_prediction, figures_dir, args.num_curves, args.random_seed
    )
    print(json.dumps({
        "target": args.target,
        "selected_sources": {name: choice["source"] for name, choice in choices.items()},
        "validation_metrics": summary["validation_metrics"],
        "test_metrics": summary["test_metrics"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
