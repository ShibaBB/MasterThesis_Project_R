"""Evaluate and plot PySR candidate formulas for segmented symbolic models.

This script is for model inspection, not training. It reads the symbolic
dataset and PySR equations CSV files, scores candidate formulas, and creates
plots that make formula trade-offs easier to compare.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from segmented_run_paths import (
    default_dataset_run,
    resolve_dataset_file,
    training_dataset_run,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_ROOT = SCRIPT_DIR.parent
if str(SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURROGATE_ROOT))

from shared_split_utils import (  # noqa: E402
    resolve_and_load_shared_split,
    source_curve_mask,
    subset_symbolic_data,
    validate_training_split,
)

DEFAULT_TRAINING_ROOT = SCRIPT_DIR / "artifacts" / "wool_segmented_symbolic_pysr_runs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "artifacts" / "wool_segmented_symbolic_candidate_evaluation_runs"
OUTPUT_LOWER_BOUND = 0.0
OUTPUT_UPPER_BOUND = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PySR candidate equations and generate diagnostic plots."
    )
    parser.add_argument(
        "--dataset-run",
        default=default_dataset_run(),
        help="Dataset run under surrogate_model/datasets (default comes from the central dataset_run_config.json).",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Optional explicit dataset path. Standard datasets/run*/ paths must match --dataset-run.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Shared source-curve split JSON. Defaults to datasets/<run>/shared_curve_split.json.",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
        help="Parent directory used to find the latest training run when --training-dir is omitted.",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=None,
        help="Specific training run directory to evaluate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for automatically named evaluation runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory. If omitted, a unique evaluation directory is created under --output-root.",
    )
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--max-plot-points", type=int, default=6000)
    parser.add_argument("--num-curves", type=int, default=6)
    parser.add_argument(
        "--selection-rule",
        choices=["best_loss", "best_score", "max_complexity"],
        default="best_loss",
        help="Default rule for selecting one candidate per segment for full-curve plots.",
    )
    parser.add_argument(
        "--max-complexity",
        type=int,
        default=12,
        help="Complexity ceiling used when --selection-rule=max_complexity.",
    )
    parser.add_argument(
        "--selected-candidates",
        default="",
        help=(
            "Optional manual segment formula choice, e.g. "
            "'low_100_700:3,mid_700_1300:1,high_1300_2000:4'. "
            "Numbers are 1-based row numbers in each equations CSV."
        ),
    )
    return parser.parse_args()


def unique_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    for suffix in range(2, 1000):
        candidate = base_path.with_name(f"{base_path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create a unique output path for base path: {base_path}")


def resolve_training_dir(args: argparse.Namespace, dataset_run: str) -> Path:
    if args.training_dir is not None:
        candidate_run = training_dataset_run(args.training_dir)
        if dataset_run != "custom" and candidate_run != dataset_run:
            raise ValueError(
                f"Training/evaluation run mismatch: dataset uses {dataset_run!r}, but "
                f"training metadata reports {candidate_run!r}: {args.training_dir}"
            )
        return args.training_dir

    if not args.training_root.exists():
        raise FileNotFoundError(
            "No --training-dir was provided and the default training root does not exist: "
            f"{args.training_root}"
        )

    candidates = [
        path
        for path in args.training_root.iterdir()
        if path.is_dir()
        and (path / "equations").exists()
        and (dataset_run == "custom" or training_dataset_run(path) == dataset_run)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No training runs for dataset run {dataset_run!r} with an equations "
            f"directory were found under: {args.training_root}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_name = sanitize_name(args.training_dir.name)
    selection_tag = sanitize_name(args.selection_rule)
    run_tag = sanitize_name(args.dataset_run)
    return unique_path(args.output_root / f"{timestamp}_{run_tag}_{training_name}_{selection_tag}")


def decode_matlab_string(file: h5py.File, value: Any) -> str:
    arr = np.array(value)
    if arr.dtype == object:
        if arr.size == 0:
            return ""
        return decode_matlab_string(file, file[arr.flat[0]])
    if h5py.check_dtype(ref=arr.dtype) is not None:
        if arr.size == 0:
            return ""
        return decode_matlab_string(file, file[arr.flat[0]])
    if np.issubdtype(arr.dtype, np.integer):
        return "".join(chr(int(code)) for code in arr.flatten() if int(code) != 0)
    if arr.dtype.kind in {"S", "U"}:
        return str(arr.flatten()[0])
    return str(arr)


def decode_matlab_string_array(file: h5py.File, dataset: h5py.Dataset) -> list[str]:
    values: list[str] = []
    refs = np.array(dataset)
    for ref in refs.flatten(order="F"):
        values.append(decode_matlab_string(file, file[ref]))
    return values


def read_symbolic_dataset(dataset_file: Path) -> dict[str, Any]:
    if not dataset_file.exists():
        raise FileNotFoundError(f"Symbolic dataset file not found: {dataset_file}")

    with h5py.File(dataset_file, "r") as file:
        info_group = file["symbolic_dataset_info"]
        return {
            "X": np.array(file["X_symbolic"]).T,
            "y": np.array(file["y_symbolic"]).reshape(-1),
            "segment_index": np.array(file["segment_index"]).reshape(-1).astype(int),
            "source_curve_index": np.array(file["source_curve_index"]).reshape(-1).astype(int),
            "freq_grid": np.array(file["freq_grid"]).reshape(-1),
            "feature_names": decode_matlab_string_array(file, info_group["feature_names"]),
            "segment_names": decode_matlab_string_array(file, info_group["segment_names"]),
            "segment_bounds": np.array(info_group["segment_bounds_hz"]).T,
        }


def make_pysr_variable_names(feature_names: list[str]) -> list[str]:
    reserved = {"lambda", "Lambda", "I", "E", "pi", "oo", "nan"}
    names: list[str] = []
    used: set[str] = set()
    for feature_name in feature_names:
        candidate = "".join(ch if ch.isalnum() else "_" for ch in feature_name).strip("_")
        if not candidate:
            candidate = "x"
        if candidate[0].isdigit():
            candidate = f"x_{candidate}"
        if candidate in reserved:
            candidate = f"{candidate}_var"
        base = candidate
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        names.append(candidate)
    return names


def sanitize_name(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in safe.split("_") if part)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(y_pred)
    if not np.all(finite):
        y_true = y_true[finite]
        y_pred = y_pred[finite]
    if y_true.size == 0:
        return {"rmse": math.inf, "mae": math.inf, "max_abs_error": math.inf, "r2": -math.inf}
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "max_abs_error": float(np.max(np.abs(y_true - y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def clip_predictions(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(y_pred, OUTPUT_LOWER_BOUND, OUTPUT_UPPER_BOUND)


def count_values_clipped(y_pred: np.ndarray) -> int:
    return int(
        np.sum(
            np.isfinite(y_pred)
            & ((y_pred < OUTPUT_LOWER_BOUND) | (y_pred > OUTPUT_UPPER_BOUND))
        )
    )


def build_predictor(expression: str, variable_names: list[str]) -> Any:
    symbols = sp.symbols(variable_names)
    local_dict = {name: symbol for name, symbol in zip(variable_names, symbols)}
    expr = sp.sympify(expression, locals=local_dict)
    return sp.lambdify(symbols, expr, modules=["numpy"])


def predict_expression(expression: str, X: np.ndarray, variable_names: list[str]) -> np.ndarray:
    predictor = build_predictor(expression, variable_names)
    values = predictor(*[X[:, idx] for idx in range(X.shape[1])])
    if np.isscalar(values):
        return np.full(X.shape[0], float(values))
    return np.asarray(values, dtype=float).reshape(-1)


def read_equations(training_dir: Path, segment_name: str) -> pd.DataFrame:
    equations_file = training_dir / "equations" / f"{sanitize_name(segment_name)}_equations.csv"
    if not equations_file.exists():
        raise FileNotFoundError(f"Equations CSV not found: {equations_file}")
    equations = pd.read_csv(equations_file)
    expression_column = "sympy_format" if "sympy_format" in equations.columns else "equation"
    equations["expression_for_eval"] = equations[expression_column].astype(str)
    equations["candidate_index"] = np.arange(1, len(equations) + 1)
    return equations


def evaluate_segment_candidates(
    data: dict[str, Any],
    training_dir: Path,
    variable_names: list[str],
    segment_id: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    segment_name = data["segment_names"][segment_id - 1]
    equations = read_equations(training_dir, segment_name)
    mask = data["segment_index"] == segment_id
    X_seg = data["X"][mask]
    y_seg = data["y"][mask]

    rows: list[dict[str, Any]] = []
    predictions: dict[int, np.ndarray] = {}

    for _, equation_row in equations.iterrows():
        candidate_index = int(equation_row["candidate_index"])
        expression = str(equation_row["expression_for_eval"])
        try:
            raw_y_pred = predict_expression(expression, X_seg, variable_names)
            y_pred = clip_predictions(raw_y_pred)
            metrics = regression_metrics(y_seg, y_pred)
            status = "ok"
            predictions[candidate_index] = y_pred
            raw_prediction_min = float(np.min(raw_y_pred))
            raw_prediction_max = float(np.max(raw_y_pred))
            num_clipped = count_values_clipped(raw_y_pred)
        except Exception as exc:  # Keep evaluating other candidates.
            metrics = {"rmse": math.inf, "mae": math.inf, "max_abs_error": math.inf, "r2": -math.inf}
            status = f"failed: {exc}"
            raw_prediction_min = math.nan
            raw_prediction_max = math.nan
            num_clipped = 0

        rows.append(
            {
                "segment_id": segment_id,
                "segment_name": segment_name,
                "candidate_index": candidate_index,
                "complexity": int(equation_row["complexity"]),
                "pysr_loss": float(equation_row["loss"]),
                "pysr_score": float(equation_row["score"]) if "score" in equation_row and pd.notna(equation_row["score"]) else np.nan,
                "equation": str(equation_row["equation"]),
                "expression_for_eval": expression,
                "eval_status": status,
                "raw_prediction_min": raw_prediction_min,
                "raw_prediction_max": raw_prediction_max,
                "num_clipped": num_clipped,
                **metrics,
            }
        )

    return pd.DataFrame(rows), predictions


def parse_manual_selection(value: str) -> dict[str, int]:
    if not value.strip():
        return {}
    selections: dict[str, int] = {}
    for item in value.split(","):
        name, index = item.split(":", maxsplit=1)
        selections[name.strip()] = int(index)
    return selections


def choose_candidates(metrics: pd.DataFrame, args: argparse.Namespace) -> dict[str, int]:
    manual = parse_manual_selection(args.selected_candidates)
    selected: dict[str, int] = {}

    for segment_name, group in metrics.groupby("segment_name", sort=False):
        if segment_name in manual:
            selected[segment_name] = manual[segment_name]
            continue

        valid = group[group["eval_status"] == "ok"].copy()
        if args.selection_rule == "best_score":
            valid = valid.sort_values(["r2", "rmse"], ascending=[False, True])
        elif args.selection_rule == "max_complexity":
            within = valid[valid["complexity"] <= args.max_complexity]
            if not within.empty:
                valid = within
            valid = valid.sort_values(["rmse", "complexity"], ascending=[True, True])
        else:
            valid = valid.sort_values(["rmse", "complexity"], ascending=[True, True])

        if valid.empty:
            raise RuntimeError(f"No valid candidates available for segment: {segment_name}")
        selected[segment_name] = int(valid.iloc[0]["candidate_index"])

    return selected


def plot_complexity_vs_error(metrics: pd.DataFrame, figures_dir: Path) -> None:
    for segment_name, group in metrics.groupby("segment_name", sort=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        valid = group[group["eval_status"] == "ok"]
        ax.scatter(valid["complexity"], valid["rmse"], s=48)
        for _, row in valid.iterrows():
            ax.annotate(str(int(row["candidate_index"])), (row["complexity"], row["rmse"]), xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Formula complexity")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{segment_name}: complexity vs error")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{sanitize_name(segment_name)}_complexity_vs_rmse.png", dpi=180)
        plt.close(fig)


def plot_selected_scatter(y_true: np.ndarray, y_pred: np.ndarray, figures_dir: Path, max_points: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    indices = np.arange(y_true.size)
    if indices.size > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true[indices], y_pred[indices], s=8, alpha=0.35)
    lower = min(float(np.min(y_true[indices])), float(np.min(y_pred[indices])))
    upper = max(float(np.max(y_true[indices])), float(np.max(y_pred[indices])))
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    ax.set_xlabel("Teacher alpha")
    ax.set_ylabel("Symbolic alpha")
    ax.set_title("Selected formulas: predicted vs true")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "selected_predicted_vs_true.png", dpi=180)
    plt.close(fig)


def plot_error_vs_frequency(freq: np.ndarray, error: np.ndarray, figures_dir: Path, max_points: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    indices = np.arange(freq.size)
    if indices.size > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(freq[indices], error[indices], s=8, alpha=0.35)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Prediction error")
    ax.set_title("Selected formulas: error vs frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "selected_error_vs_frequency.png", dpi=180)
    plt.close(fig)


def plot_curve_comparisons(
    data: dict[str, Any],
    y_pred_all: np.ndarray,
    figures_dir: Path,
    num_curves: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    curve_ids = np.unique(data["source_curve_index"])
    if curve_ids.size > num_curves:
        curve_ids = rng.choice(curve_ids, size=num_curves, replace=False)

    n_cols = 2
    n_rows = int(math.ceil(len(curve_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows), squeeze=False)

    for ax, curve_id in zip(axes.flatten(), curve_ids):
        mask = data["source_curve_index"] == curve_id
        order = np.argsort(data["X"][mask, -1])
        freq = data["X"][mask, -1][order]
        y_true = data["y"][mask][order]
        y_pred = y_pred_all[mask][order]
        ax.plot(freq, y_true, label="Teacher", linewidth=1.8)
        ax.plot(freq, y_pred, label="Symbolic", linewidth=1.5, linestyle="--")
        ax.set_title(f"Curve {int(curve_id)}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("alpha")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    for ax in axes.flatten()[len(curve_ids):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figures_dir / "selected_curve_comparisons.png", dpi=180)
    plt.close(fig)


def evaluate_selected_combination(
    data: dict[str, Any],
    metrics: pd.DataFrame,
    selected: dict[str, int],
    variable_names: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    raw_y_pred_all = np.full_like(data["y"], np.nan, dtype=float)
    y_pred_all = np.full_like(data["y"], np.nan, dtype=float)
    selected_rows: list[dict[str, Any]] = []

    for segment_id, segment_name in enumerate(data["segment_names"], start=1):
        candidate_index = selected[segment_name]
        row = metrics[
            (metrics["segment_name"] == segment_name)
            & (metrics["candidate_index"] == candidate_index)
        ].iloc[0]
        mask = data["segment_index"] == segment_id
        raw_prediction = predict_expression(
            row["expression_for_eval"], data["X"][mask], variable_names
        )
        raw_y_pred_all[mask] = raw_prediction
        y_pred_all[mask] = clip_predictions(raw_prediction)
        selected_rows.append(row.to_dict())

    selected_df = pd.DataFrame(selected_rows)
    return raw_y_pred_all, y_pred_all, selected_df


def prediction_diagnostics(y_pred: np.ndarray) -> dict[str, float | int]:
    below_zero = y_pred < 0.0
    above_one = y_pred > 1.0
    return {
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "count_below_zero": int(np.sum(below_zero)),
        "count_above_one": int(np.sum(above_one)),
        "fraction_outside_0_1": float(np.mean(below_zero | above_one)),
    }


def boundary_transition_diagnostics(
    data: dict[str, Any], y_pred: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    curve_ids = np.unique(data["source_curve_index"])
    frequency = data["X"][:, -1]

    for left_segment_id in range(1, len(data["segment_names"])):
        right_segment_id = left_segment_id + 1
        prediction_changes: list[float] = []
        teacher_changes: list[float] = []

        for curve_id in curve_ids:
            curve_mask = data["source_curve_index"] == curve_id
            left_indices = np.flatnonzero(
                curve_mask & (data["segment_index"] == left_segment_id)
            )
            right_indices = np.flatnonzero(
                curve_mask & (data["segment_index"] == right_segment_id)
            )
            if left_indices.size == 0 or right_indices.size == 0:
                continue

            left_index = left_indices[np.argmax(frequency[left_indices])]
            right_index = right_indices[np.argmin(frequency[right_indices])]
            prediction_changes.append(float(y_pred[right_index] - y_pred[left_index]))
            teacher_changes.append(float(data["y"][right_index] - data["y"][left_index]))

        prediction_array = np.asarray(prediction_changes)
        teacher_array = np.asarray(teacher_changes)
        excess_change = prediction_array - teacher_array
        rows.append(
            {
                "left_segment": data["segment_names"][left_segment_id - 1],
                "right_segment": data["segment_names"][right_segment_id - 1],
                "boundary_hz": float(data["segment_bounds"][left_segment_id - 1][1]),
                "num_curves": int(prediction_array.size),
                "mean_abs_prediction_change": float(np.mean(np.abs(prediction_array))),
                "max_abs_prediction_change": float(np.max(np.abs(prediction_array))),
                "mean_abs_teacher_change": float(np.mean(np.abs(teacher_array))),
                "mean_abs_excess_change": float(np.mean(np.abs(excess_change))),
                "max_abs_excess_change": float(np.max(np.abs(excess_change))),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.dataset_file, resolved_dataset_run = resolve_dataset_file(
        args.dataset_run, args.dataset_file
    )
    args.dataset_run = resolved_dataset_run
    args.training_dir = resolve_training_dir(args, resolved_dataset_run)
    args.output_dir = resolve_output_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = read_symbolic_dataset(args.dataset_file)
    num_curve_samples = int(np.unique(data["source_curve_index"]).size)
    shared_split = resolve_and_load_shared_split(
        resolved_dataset_run, args.split_file, num_curve_samples
    )
    validate_training_split(args.training_dir, shared_split)
    validation_data = subset_symbolic_data(
        data,
        source_curve_mask(data["source_curve_index"], shared_split, "validation"),
    )
    test_data = subset_symbolic_data(
        data,
        source_curve_mask(data["source_curve_index"], shared_split, "test"),
    )
    variable_names = make_pysr_variable_names(data["feature_names"])

    all_metrics: list[pd.DataFrame] = []
    for segment_id in range(1, len(data["segment_names"]) + 1):
        segment_metrics, _ = evaluate_segment_candidates(
            validation_data,
            args.training_dir,
            variable_names,
            segment_id,
        )
        all_metrics.append(segment_metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics["evaluation_split"] = "validation"
    metrics.to_csv(args.output_dir / "candidate_metrics.csv", index=False)
    plot_complexity_vs_error(metrics, figures_dir)

    selected = choose_candidates(metrics, args)
    _, validation_y_pred, _ = evaluate_selected_combination(
        validation_data, metrics, selected, variable_names
    )
    validation_metrics = regression_metrics(validation_data["y"], validation_y_pred)
    raw_y_pred_all, y_pred_all, selected_df = evaluate_selected_combination(
        test_data, metrics, selected, variable_names
    )
    selected_df.to_csv(args.output_dir / "selected_candidates.csv", index=False)

    selected_metrics = regression_metrics(test_data["y"], y_pred_all)
    raw_physical_diagnostics = prediction_diagnostics(raw_y_pred_all)
    physical_diagnostics = prediction_diagnostics(y_pred_all)
    boundary_diagnostics = boundary_transition_diagnostics(test_data, y_pred_all)
    boundary_diagnostics.to_csv(
        args.output_dir / "boundary_transition_metrics.csv", index=False
    )
    selected_summary = {
        "dataset_file": str(args.dataset_file),
        "dataset_run": resolved_dataset_run,
        "shared_split_file": shared_split["split_file"],
        "shared_split_hash": shared_split["split_hash"],
        "training_dir": str(args.training_dir),
        "candidate_selection_split": "validation",
        "final_evaluation_split": "test",
        "selection_rule": args.selection_rule,
        "output_postprocessing": {
            "method": "clip",
            "lower_bound": OUTPUT_LOWER_BOUND,
            "upper_bound": OUTPUT_UPPER_BOUND,
            "num_values_clipped": count_values_clipped(raw_y_pred_all),
        },
        "selected_candidates": selected,
        "feature_names": data["feature_names"],
        "pysr_variable_names": variable_names,
        "overall_metrics": selected_metrics,
        "validation_selection_metrics": validation_metrics,
        "raw_prediction_diagnostics": raw_physical_diagnostics,
        "prediction_diagnostics": physical_diagnostics,
        "boundary_transition_diagnostics": boundary_diagnostics.to_dict(orient="records"),
    }
    with (args.output_dir / "selected_combination_summary.json").open("w", encoding="utf-8") as file:
        json.dump(selected_summary, file, indent=2)

    freq_values = test_data["X"][:, -1]
    plot_selected_scatter(test_data["y"], y_pred_all, figures_dir, args.max_plot_points, args.random_seed)
    plot_error_vs_frequency(freq_values, y_pred_all - test_data["y"], figures_dir, args.max_plot_points, args.random_seed)
    plot_curve_comparisons(test_data, y_pred_all, figures_dir, args.num_curves, args.random_seed)

    print("Symbolic candidate evaluation complete.")
    print(f"Candidate metrics: {args.output_dir / 'candidate_metrics.csv'}")
    print(f"Selected candidates: {selected}")
    print(
        "Overall selected-formula metrics: "
        f"RMSE={selected_metrics['rmse']:.6f}, "
        f"MAE={selected_metrics['mae']:.6f}, "
        f"R2={selected_metrics['r2']:.6f}"
    )
    print(
        "Prediction diagnostics: "
        f"raw_range=[{raw_physical_diagnostics['prediction_min']:.6f}, "
        f"{raw_physical_diagnostics['prediction_max']:.6f}], "
        f"clipped_values={count_values_clipped(raw_y_pred_all)}, "
        f"final_range=[{physical_diagnostics['prediction_min']:.6f}, "
        f"{physical_diagnostics['prediction_max']:.6f}]"
    )
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
