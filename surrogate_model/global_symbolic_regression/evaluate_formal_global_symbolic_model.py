"""Evaluate the frozen formal Global SR pair without selecting or tuning models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_ROOT = SCRIPT_DIR.parent
for import_path in (SCRIPT_DIR, SURROGATE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from evaluate_global_symbolic_candidates import (  # noqa: E402
    read_symbolic_dataset,
    regression_metrics,
)
from formal_global_symbolic_model import (  # noqa: E402
    DEFAULT_POINTER,
    FormalGlobalSymbolicModel,
)
from plot_formal_global_symbolic_evaluation import (  # noqa: E402
    generate_evaluation_figures,
)
from shared_split_utils import load_shared_split, source_curve_mask  # noqa: E402


DEFAULT_DATASET = SURROGATE_ROOT / "datasets" / "run1" / "global_SR" / "Wool_R_global.mat"
DEFAULT_SPLIT = SURROGATE_ROOT / "datasets" / "run1" / "shared_curve_split.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "artifacts" / "formal_integration"
DEFAULT_RE_REFERENCE = (
    SCRIPT_DIR
    / "artifacts"
    / "F3"
    / "F3-F2"
    / "20260818T213528"
    / "test_predictions.npz"
)


def load_paired_symbolic_dataset(dataset_file: Path) -> dict[str, Any]:
    re_data = read_symbolic_dataset(dataset_file, "re", "run1")
    im_data = read_symbolic_dataset(dataset_file, "im", "run1")
    for key in ("feature_names", "segment_names"):
        if re_data[key] != im_data[key]:
            raise ValueError(f"Paired symbolic dataset differs between targets: {key}")
    for key in ("X", "source_curve_index", "freq_grid", "segment_bounds"):
        if not np.array_equal(re_data[key], im_data[key]):
            raise ValueError(f"Paired symbolic dataset differs between targets: {key}")
    return {
        "X": re_data["X"],
        "y_re": re_data["y"],
        "y_im": im_data["y"],
        "source_curve_index": re_data["source_curve_index"],
        "frequency_hz": re_data["X"][:, re_data["feature_names"].index("f")],
        "feature_names": re_data["feature_names"],
        "freq_grid": re_data["freq_grid"],
    }


def complex_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.abs(y_pred - y_true)
    return {
        "complex_rmse": float(math.sqrt(np.mean(error**2))),
        "complex_mae": float(np.mean(error)),
        "complex_max_abs_error": float(np.max(error)),
    }


def compare_metrics(
    actual: dict[str, float], expected: dict[str, float], tolerance: float = 1e-12
) -> dict[str, Any]:
    compared: dict[str, dict[str, float | bool]] = {}
    passed = True
    for key in ("rmse", "mae", "max_abs_error", "r2", "prediction_min", "prediction_max"):
        difference = abs(float(actual[key]) - float(expected[key]))
        item_passed = difference <= tolerance
        passed = passed and item_passed
        compared[key] = {
            "actual": float(actual[key]),
            "expected": float(expected[key]),
            "absolute_difference": difference,
            "passed": item_passed,
        }
    return {"tolerance": tolerance, "passed": passed, "metrics": compared}


def evaluate_partition(
    model: FormalGlobalSymbolicModel,
    paired: dict[str, Any],
    split: dict[str, Any],
    partition: str,
    re_reference: Path | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if split["split_hash"] != model.split_hash:
        raise ValueError("Runtime split hash differs from the formal model manifest.")
    if partition == "all":
        mask = np.ones(paired["source_curve_index"].shape, dtype=bool)
    else:
        mask = source_curve_mask(paired["source_curve_index"], split, partition)
    X = paired["X"][mask]
    source_ids = paired["source_curve_index"][mask]
    frequency = paired["frequency_hz"][mask]
    y_re = paired["y_re"][mask]
    y_im = paired["y_im"][mask]

    prediction = model.predict_with_diagnostics(X, paired["feature_names"])
    re_metrics = regression_metrics(y_re, prediction["R_real"])
    im_metrics = regression_metrics(y_im, prediction["R_imag"])
    paired_metrics = complex_metrics(y_re + 1j * y_im, prediction["Reflect"])
    summary: dict[str, Any] = {
        "model_id": model.manifest["model_id"],
        "dataset_run": "run1",
        "partition": partition,
        "shared_split_hash": split["split_hash"],
        "curve_count": int(np.unique(source_ids).size),
        "row_count": int(X.shape[0]),
        "active_models": {"re": "Re_F3", "im": "Im_F2"},
        "candidate_selection_performed": False,
        "output_postprocessing": {"method": "none"},
        "metrics": {"re": re_metrics, "im": im_metrics, "complex": paired_metrics},
        "reference_metric_regression": {
            "applicable": partition == "test",
            "re": None,
            "im": None,
        },
        "exact_re_prediction_regression": {"applicable": False},
    }

    if partition == "test":
        summary["reference_metric_regression"]["re"] = compare_metrics(
            re_metrics, model.manifest["targets"]["re"]["reference_test_metrics"]
        )
        summary["reference_metric_regression"]["im"] = compare_metrics(
            im_metrics, model.manifest["targets"]["im"]["reference_test_metrics"]
        )
        if re_reference is not None:
            with np.load(re_reference, allow_pickle=False) as archive:
                ids_match = np.array_equal(source_ids, archive["source_curve_index"])
                frequency_match = np.array_equal(frequency, archive["frequency_hz"])
                reference_prediction = np.asarray(archive["frozen_f3_prediction"], dtype=float)
                prediction_match = np.array_equal(
                    prediction["R_real"], reference_prediction
                )
                terminal_match = np.array_equal(
                    prediction["z_sigma__f3_mid"], archive["z_sigma__f3_mid"]
                )
                max_difference = float(
                    np.max(np.abs(prediction["R_real"] - reference_prediction))
                )
            summary["exact_re_prediction_regression"] = {
                "applicable": True,
                "reference_file": str(re_reference.resolve()),
                "source_curve_order_exact": ids_match,
                "frequency_order_exact": frequency_match,
                "prediction_exact": prediction_match,
                "f3_terminal_exact": terminal_match,
                "maximum_absolute_prediction_difference": max_difference,
                "passed": ids_match and frequency_match and prediction_match and terminal_match,
            }

    arrays = {
        "source_curve_index": source_ids,
        "frequency_hz": frequency,
        "teacher_re": y_re,
        "teacher_im": y_im,
        "predicted_re": prediction["R_real"],
        "predicted_im": prediction["R_imag"],
        "predicted_reflect": prediction["Reflect"],
        "re_f1_prediction": prediction["Re_F1"],
        "re_f3_residual": prediction["Re_F3_residual"],
        "z_sigma__f3_mid": prediction["z_sigma__f3_mid"],
    }
    return summary, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--partition", choices=("train", "validation", "test", "all"), default="test"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--re-reference",
        type=Path,
        default=DEFAULT_RE_REFERENCE,
        help="Saved one-time F3-F2 prediction artifact used only for exact replay checking.",
    )
    parser.add_argument("--require-exact-regression", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%dT%H%M%S_run1_f3_f2")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = FormalGlobalSymbolicModel.load(args.model)
    paired = load_paired_symbolic_dataset(args.dataset_file.resolve())
    split = load_shared_split(
        args.split_file.resolve(), expected_dataset_run="run1", expected_curve_count=1000
    )
    reference = args.re_reference.resolve() if args.partition == "test" else None
    summary, arrays = evaluate_partition(model, paired, split, args.partition, reference)
    np.savez_compressed(output_dir / "predictions.npz", **arrays)
    figure_files = generate_evaluation_figures(arrays, output_dir / "figures")
    summary["figure_files"] = [str(path.resolve()) for path in figure_files]
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    metric_regression = summary["reference_metric_regression"]
    passed = True
    if metric_regression["applicable"]:
        passed = bool(metric_regression["re"]["passed"] and metric_regression["im"]["passed"])
    exact = summary["exact_re_prediction_regression"]
    if exact["applicable"]:
        passed = passed and bool(exact["passed"])
    if args.require_exact_regression and not passed:
        raise RuntimeError(f"Formal Global SR exact regression failed; see {output_dir}")

    print(
        "Formal Global SR evaluation complete: "
        f"partition={args.partition}, rows={summary['row_count']}, regression_passed={passed}"
    )
    print(f"Evaluation artifacts: {output_dir}")


if __name__ == "__main__":
    main()
