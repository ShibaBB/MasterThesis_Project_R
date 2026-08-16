"""Run the validation-only Re Global SR F3-A ridge residual diagnosis.

The stage reconstructs the frozen Re F1 equation, fits predeclared local
linear corrections on training residuals, and evaluates them on validation
curves.  It deliberately reads neither test features nor test targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_ROOT = SCRIPT_DIR.parent
if str(SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURROGATE_ROOT))

from shared_split_utils import load_shared_split  # noqa: E402
from symbolic_feature_transform import apply_feature_transform  # noqa: E402
from global_frequency_modulation import (  # noqa: E402
    apply_frequency_modulation,
    validate_frequency_modulation,
)
from global_sobol_feature_policy import validate_stored_feature_policy  # noqa: E402
from evaluate_global_symbolic_candidates import (  # noqa: E402
    make_pysr_variable_names,
    predict_expression,
)


DATASET_RUN = "run1"
TARGET = "re"
EXPECTED_TARGET_NAME = "R_real"
DEFAULT_DATASET_FILE = (
    SURROGATE_ROOT / "datasets" / DATASET_RUN / "global_SR" / "Wool_R_global.mat"
)
DEFAULT_SPLIT_FILE = SURROGATE_ROOT / "datasets" / DATASET_RUN / "shared_curve_split.json"
DEFAULT_F1_TRAINING_DIR = (
    SCRIPT_DIR / "artifacts" / "re" / "train" / "20260816_run1_sobol_modulated"
)
DEFAULT_F1_SELECTION_FILE = (
    SCRIPT_DIR
    / "artifacts"
    / "re"
    / "eval"
    / "20260816_run1_sobol_modulated_best_loss"
    / "selected_candidate.csv"
)
F3_ARTIFACT_ROOT = SCRIPT_DIR / "artifacts" / "F3"

RIDGE_ALPHA = 1.0
GAUSSIAN_WIDTH_LOG10 = 0.08
BANDPASS_LOW_HZ = 1250.0
BANDPASS_HIGH_HZ = 2100.0
BANDPASS_TRANSITION_LOG10 = 0.06
F1_VALIDATION_RMSE = 0.06997865488067417
F1_RECONSTRUCTION_TOLERANCE = 1.0e-10
GUARDRAIL_MAX_RELATIVE_WORSENING = 0.10

REPORTING_REGIONS = (
    (100.0, 700.0),
    (700.0, 1000.0),
    (1000.0, 1300.0),
    (1300.0, 1650.0),
    (1650.0, 2000.0),
    (2000.0, 3000.0),
    (3000.0, 4000.0),
    (4000.0, 4950.0),
)
GUARDRAIL_REGIONS = (
    (700.0, 1000.0),
    (1300.0, 1650.0),
    (1650.0, 2000.0),
)
TARGET_REGION = (1300.0, 2000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Re Global SR F3-A without reading test rows."
    )
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--f1-training-dir", type=Path, default=DEFAULT_F1_TRAINING_DIR)
    parser.add_argument("--f1-selection-file", type=Path, default=DEFAULT_F1_SELECTION_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact run directory; it must remain under artifacts/F3.",
    )
    parser.add_argument("--num-overlay-curves", type=int, default=6)
    return parser.parse_args()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for suffix in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique output directory near {path}.")


def resolve_output_dir(requested: Path | None) -> Path:
    root = F3_ARTIFACT_ROOT.resolve()
    if requested is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = unique_path(root / "F3-A" / timestamp)
    else:
        output_dir = requested.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError(f"F3-A artifacts must stay under {root}: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_matlab_string(file: h5py.File, value: Any) -> str:
    array = np.asarray(value)
    if array.dtype == object or h5py.check_dtype(ref=array.dtype) is not None:
        if array.size == 0:
            return ""
        return decode_matlab_string(file, file[array.flat[0]])
    if np.issubdtype(array.dtype, np.integer):
        return "".join(chr(int(code)) for code in array.flatten() if int(code) != 0)
    if array.dtype.kind in {"S", "U"}:
        return str(array.flatten()[0])
    return str(array)


def decode_matlab_string_array(file: h5py.File, dataset: h5py.Dataset) -> list[str]:
    return [
        decode_matlab_string(file, file[reference])
        for reference in np.asarray(dataset).flatten(order="F")
    ]


def load_partition_rows(
    dataset_file: Path,
    curve_ids: np.ndarray,
    expected_curve_count: int,
) -> dict[str, Any]:
    """Read only explicitly allowed curve blocks from the HDF5 MAT file."""
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    ordered_curve_ids = np.sort(np.asarray(curve_ids, dtype=int).reshape(-1))
    if ordered_curve_ids.size == 0:
        raise ValueError("A requested partition contains no curves.")

    with h5py.File(dataset_file, "r") as file:
        required = (
            "X_symbolic",
            "y_re_symbolic",
            "y_im_symbolic",
            "source_curve_index",
            "freq_grid",
            "symbolic_dataset_info",
        )
        missing = [name for name in required if name not in file]
        if missing:
            raise ValueError(f"Global symbolic dataset is missing fields: {missing}")
        info = file["symbolic_dataset_info"]
        feature_names = decode_matlab_string_array(file, info["feature_names"])
        target_names = decode_matlab_string_array(file, info["target_names"])
        complex_source = decode_matlab_string(file, info["complex_source"])
        recorded_run = decode_matlab_string(file, info["dataset_run"])
        if target_names != ["R_real", "R_imag"] or complex_source != "Reflect":
            raise ValueError("Dataset metadata does not describe paired Reflect targets.")
        if recorded_run != DATASET_RUN:
            raise ValueError(f"Expected dataset run {DATASET_RUN!r}, found {recorded_run!r}.")

        frequency_grid = np.asarray(file["freq_grid"]).reshape(-1).astype(float)
        rows_per_curve = int(frequency_grid.size)
        expected_rows = expected_curve_count * rows_per_curve
        if file["X_symbolic"].shape != (len(feature_names), expected_rows):
            raise ValueError("X_symbolic shape does not match curve count and frequency grid.")
        if file["y_re_symbolic"].shape != (1, expected_rows):
            raise ValueError("y_re_symbolic shape does not match the expected scalar rows.")

        x_blocks: list[np.ndarray] = []
        y_blocks: list[np.ndarray] = []
        source_blocks: list[np.ndarray] = []
        for curve_id in ordered_curve_ids:
            if curve_id < 1 or curve_id > expected_curve_count:
                raise ValueError(f"Curve id is outside the dataset: {curve_id}")
            start = (int(curve_id) - 1) * rows_per_curve
            stop = start + rows_per_curve
            stored_ids = np.asarray(file["source_curve_index"][0, start:stop]).astype(int)
            if stored_ids.size != rows_per_curve or not np.all(stored_ids == curve_id):
                raise ValueError(
                    "Dataset is not in the required contiguous curve-major row order."
                )
            x_blocks.append(np.asarray(file["X_symbolic"][:, start:stop]).T)
            y_blocks.append(np.asarray(file["y_re_symbolic"][0, start:stop]))
            source_blocks.append(stored_ids)

    X = np.vstack(x_blocks).astype(float)
    y = np.concatenate(y_blocks).astype(float)
    source_curve_index = np.concatenate(source_blocks).astype(int)
    frequency_index = feature_names.index("f")
    frequency_hz = X[:, frequency_index].copy()
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Loaded partition contains non-finite data.")
    return {
        "X": X,
        "y": y,
        "source_curve_index": source_curve_index,
        "frequency_hz": frequency_hz,
        "feature_names": feature_names,
        "frequency_grid": frequency_grid,
        "curve_ids": ordered_curve_ids,
        "rows_per_curve": rows_per_curve,
    }


def prepare_f1_features(
    partition: dict[str, Any], training_metadata: dict[str, Any]
) -> dict[str, Any]:
    original_names = list(partition["feature_names"])
    transform = training_metadata["feature_transform"]
    X = apply_feature_transform(partition["X"], original_names, transform)
    transformed_names = list(transform["transformed_feature_names"])
    policy = validate_stored_feature_policy(
        training_metadata["feature_policy"], TARGET, transformed_names
    )
    X = X[:, [int(index) for index in policy["feature_indices_0based"]]]
    base_names = list(policy["feature_names"])
    modulation = validate_frequency_modulation(
        training_metadata["frequency_modulation"], TARGET, base_names
    )
    X_f1, f1_names = apply_frequency_modulation(X, base_names, modulation)
    expected_names = list(training_metadata["feature_names"])
    if f1_names != expected_names:
        raise ValueError("Reconstructed F1 feature order differs from training metadata.")
    result = partition.copy()
    result["X_base"] = X
    result["base_feature_names"] = base_names
    result["X_f1"] = X_f1
    result["f1_feature_names"] = f1_names
    return result


def standardization(values: np.ndarray, curve_ids: np.ndarray) -> dict[str, float | str]:
    _, first_indices = np.unique(curve_ids, return_index=True)
    unique_curve_values = np.asarray(values[first_indices], dtype=float)
    center = float(np.mean(unique_curve_values))
    scale = float(np.std(unique_curve_values, ddof=0))
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cannot standardize a non-finite or constant material feature.")
    return {
        "center": center,
        "scale": scale,
        "method": "training_unique_curve_population_mean_std_ddof0",
        "training_curve_count": int(unique_curve_values.size),
    }


def apply_standardization(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    standardized = (np.asarray(values, dtype=float) - float(stats["center"])) / float(
        stats["scale"]
    )
    if not np.all(np.isfinite(standardized)):
        raise ValueError("Standardized feature contains non-finite values.")
    return standardized


def gaussian_envelope(frequency_hz: np.ndarray, center_hz: float) -> np.ndarray:
    log_frequency = np.log10(np.asarray(frequency_hz, dtype=float))
    return np.exp(
        -0.5
        * ((log_frequency - math.log10(center_hz)) / GAUSSIAN_WIDTH_LOG10) ** 2
    )


def logistic_highpass(
    log_frequency: np.ndarray, center_hz: float, transition_log10: float
) -> np.ndarray:
    signed = (log_frequency - math.log10(center_hz)) / transition_log10
    return 1.0 / (1.0 + np.exp(-np.clip(signed, -60.0, 60.0)))


def bandpass_envelope(frequency_hz: np.ndarray) -> np.ndarray:
    log_frequency = np.log10(np.asarray(frequency_hz, dtype=float))
    highpass = logistic_highpass(
        log_frequency, BANDPASS_LOW_HZ, BANDPASS_TRANSITION_LOG10
    )
    lowpass = 1.0 - logistic_highpass(
        log_frequency, BANDPASS_HIGH_HZ, BANDPASS_TRANSITION_LOG10
    )
    values = highpass * lowpass
    reference = np.linspace(math.log10(100.0), math.log10(4950.0), 4096)
    normalizer = float(
        np.max(
            logistic_highpass(reference, BANDPASS_LOW_HZ, BANDPASS_TRANSITION_LOG10)
            * (
                1.0
                - logistic_highpass(
                    reference, BANDPASS_HIGH_HZ, BANDPASS_TRANSITION_LOG10
                )
            )
        )
    )
    return values / normalizer


def build_envelopes(frequency_hz: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "g1450": gaussian_envelope(frequency_hz, 1450.0),
        "g1750": gaussian_envelope(frequency_hz, 1750.0),
        "g_bandpass_1250_2100": bandpass_envelope(frequency_hz),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "max_abs_error": float(np.max(np.abs(error))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(error)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
    }


def curve_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, curve_ids: np.ndarray
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    for curve_id in np.unique(curve_ids):
        mask = curve_ids == curve_id
        rmse = float(math.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
        rows.append({"curve_id": int(curve_id), "rmse": rmse})
    values = np.asarray([row["rmse"] for row in rows], dtype=float)
    worst_index = int(np.argmax(values))
    summary: dict[str, float | int] = {
        "curve_mean_rmse": float(np.mean(values)),
        "curve_median_rmse": float(np.median(values)),
        "curve_p95_rmse": float(np.quantile(values, 0.95)),
        "worst_curve_rmse": float(values[worst_index]),
        "worst_curve_id": int(rows[worst_index]["curve_id"]),
    }
    return summary, rows


def region_mask(frequency_hz: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = bounds
    return (frequency_hz >= lower) & (frequency_hz <= upper)


def regional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    frequency_hz: np.ndarray,
    regions: tuple[tuple[float, float], ...],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for bounds in regions:
        mask = region_mask(frequency_hz, bounds)
        if not np.any(mask):
            raise ValueError(f"No rows occur in requested region {bounds}.")
        error = y_pred[mask] - y_true[mask]
        rows.append(
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "row_count": int(np.sum(mask)),
                "rmse": float(math.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
            }
        )
    return rows


def frequency_wise_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, frequency_hz: np.ndarray
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for frequency in np.unique(frequency_hz):
        mask = frequency_hz == frequency
        error = y_pred[mask] - y_true[mask]
        rows.append(
            {
                "frequency_hz": float(frequency),
                "rmse": float(math.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
            }
        )
    return rows


def fit_ridge(
    term_names: list[str],
    train_terms: dict[str, np.ndarray],
    train_residual: np.ndarray,
) -> tuple[Ridge | None, np.ndarray]:
    if not term_names:
        return None, np.zeros_like(train_residual)
    design = np.column_stack([train_terms[name] for name in term_names])
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, solver="cholesky")
    model.fit(design, train_residual)
    return model, np.asarray(model.predict(design), dtype=float)


def predict_ridge(
    model: Ridge | None, term_names: list[str], terms: dict[str, np.ndarray], row_count: int
) -> np.ndarray:
    if model is None:
        return np.zeros(row_count, dtype=float)
    design = np.column_stack([terms[name] for name in term_names])
    return np.asarray(model.predict(design), dtype=float)


def region_lookup(rows: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (float(row["lower_hz"]), float(row["upper_hz"])): row for row in rows
    }


def candidate_passes(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    reasons: list[str] = []
    if candidate["validation_metrics"]["rmse"] >= baseline["validation_metrics"]["rmse"]:
        reasons.append("full_validation_rmse_not_improved")
    candidate_regions = region_lookup(candidate["validation_regions"])
    baseline_regions = region_lookup(baseline["validation_regions"])
    if candidate_regions[TARGET_REGION]["rmse"] >= baseline_regions[TARGET_REGION]["rmse"]:
        reasons.append("target_1300_2000_rmse_not_improved")
    guardrails: list[dict[str, Any]] = []
    for bounds in GUARDRAIL_REGIONS:
        control_rmse = float(baseline_regions[bounds]["rmse"])
        candidate_rmse = float(candidate_regions[bounds]["rmse"])
        relative_change = candidate_rmse / control_rmse - 1.0
        passed = relative_change <= GUARDRAIL_MAX_RELATIVE_WORSENING
        if not passed:
            reasons.append(f"guardrail_{int(bounds[0])}_{int(bounds[1])}_failed")
        guardrails.append(
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "control_rmse": control_rmse,
                "candidate_rmse": candidate_rmse,
                "relative_change": relative_change,
                "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                "passed": passed,
            }
        )
    return not reasons, reasons, guardrails


def plot_envelopes(frequency_hz: np.ndarray, envelopes: dict[str, np.ndarray], path: Path) -> None:
    order = np.argsort(frequency_hz)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in envelopes.items():
        ax.plot(frequency_hz[order], values[order], linewidth=2, label=name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Envelope value")
    ax.set_title("F3-A predeclared local frequency envelopes")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_candidate_rmse(candidate_records: list[dict[str, Any]], path: Path) -> None:
    records = [record for record in candidate_records if record["family"] != "optional_material_diagnostic"]
    names = [record["name"] for record in records]
    values = [record["validation_metrics"]["rmse"] for record in records]
    colors = ["#777777" if record["family"] == "control" else "#2878b5" for record in records]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(names)), values, color=colors)
    ax.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("F3-A validation-only ridge residual comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_frequency_rmse(
    records: list[dict[str, Any]], selected_name: str, path: Path
) -> None:
    chosen_names = ["zero_correction", selected_name]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in chosen_names:
        record = next(record for record in records if record["name"] == name)
        table = pd.DataFrame(record["validation_frequency_metrics"])
        label = "Re F1" if name == "zero_correction" else name
        ax.plot(table["frequency_hz"], table["rmse"], linewidth=2, label=label)
    for lower, upper in GUARDRAIL_REGIONS:
        ax.axvspan(lower, upper, color="#bbbbbb", alpha=0.08)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Frequency-wise validation RMSE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def select_overlay_curve_ids(
    baseline_curve_rows: list[dict[str, Any]], count: int
) -> list[int]:
    ordered = sorted(baseline_curve_rows, key=lambda row: float(row["rmse"]))
    count = max(1, min(int(count), len(ordered)))
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [int(ordered[position]["curve_id"]) for position in positions]


def plot_curve_overlays(
    validation: dict[str, Any],
    f1_prediction: np.ndarray,
    selected_prediction: np.ndarray,
    selected_name: str,
    curve_ids: list[int],
    path: Path,
) -> None:
    columns = 2
    rows = int(math.ceil(len(curve_ids) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.8 * rows), squeeze=False)
    for ax, curve_id in zip(axes.flatten(), curve_ids):
        mask = validation["source_curve_index"] == curve_id
        order = np.argsort(validation["frequency_hz"][mask])
        frequency = validation["frequency_hz"][mask][order]
        ax.plot(frequency, validation["y"][mask][order], color="black", linewidth=2, label="Teacher")
        ax.plot(frequency, f1_prediction[mask][order], color="#777777", linestyle="--", label="Re F1")
        ax.plot(frequency, selected_prediction[mask][order], color="#2878b5", label=selected_name)
        combined = np.concatenate(
            [validation["y"][mask], f1_prediction[mask], selected_prediction[mask]]
        )
        span = float(np.max(combined) - np.min(combined))
        padding = max(0.05 * span, 0.02)
        ax.set_ylim(float(np.min(combined) - padding), float(np.max(combined) + padding))
        ax.set_title(f"Validation curve {curve_id}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(EXPECTED_TARGET_NAME)
        ax.grid(alpha=0.2)
    for ax in axes.flatten()[len(curve_ids):]:
        ax.axis("off")
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("F3-A full-curve validation overlays", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mean_residual_by_frequency(
    validation: dict[str, Any],
    f1_prediction: np.ndarray,
    selected_prediction: np.ndarray,
    selected_name: str,
    path: Path,
) -> None:
    frequency = np.unique(validation["frequency_hz"])
    f1_bias: list[float] = []
    selected_bias: list[float] = []
    for value in frequency:
        mask = validation["frequency_hz"] == value
        f1_bias.append(float(np.mean(validation["y"][mask] - f1_prediction[mask])))
        selected_bias.append(float(np.mean(validation["y"][mask] - selected_prediction[mask])))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(frequency, f1_bias, color="#777777", linestyle="--", linewidth=2, label="Re F1 residual")
    ax.plot(frequency, selected_bias, color="#2878b5", linewidth=2, label=f"{selected_name} residual")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean teacher - prediction")
    ax.set_title("Validation mean residual by frequency")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_decision_markdown(path: Path, decision: dict[str, Any]) -> None:
    selected = decision.get("selected_sigma_candidate")
    status = "PASS" if decision["stage_passed"] else "STOP / NO PASS"
    lines = [
        "# Re Global SR F3-A Decision Report",
        "",
        f"- Stage decision: **{status}**",
        f"- Selected sigma candidate: `{selected}`" if selected else "- Selected sigma candidate: none",
        f"- Test rows accessed: `{decision['test_rows_accessed']}`",
        f"- Next stage automatically authorized: `{decision['next_stage_automatically_authorized']}`",
        "",
        "## Decision basis",
        "",
    ]
    for reason in decision["decision_reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "Full metrics, regional guardrails, coefficients, figures, and the exact",
            "training-only standardization are stored beside this report.",
            "F3-B requires a new explicit user command even when F3-A passes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    execution_log = output_dir / "execution.log"

    def log(message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with execution_log.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    log("Starting validation-only F3-A ridge residual diagnosis.")
    dataset_file = args.dataset_file.resolve()
    split_file = args.split_file.resolve()
    f1_training_dir = args.f1_training_dir.resolve()
    f1_selection_file = args.f1_selection_file.resolve()
    training_metadata_file = f1_training_dir / "training_metadata.json"
    for required_path in (split_file, training_metadata_file, f1_selection_file):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    split = load_shared_split(
        split_file,
        expected_dataset_run=DATASET_RUN,
        expected_curve_count=1000,
    )
    train_curve_ids = np.asarray(split["train_curve_indices"], dtype=int)
    validation_curve_ids = np.asarray(split["validation_curve_indices"], dtype=int)
    test_curve_ids = np.asarray(split["test_curve_indices"], dtype=int)
    if np.intersect1d(np.concatenate([train_curve_ids, validation_curve_ids]), test_curve_ids).size:
        raise ValueError("Train/validation ids overlap the forbidden test partition.")

    with training_metadata_file.open("r", encoding="utf-8") as file:
        training_metadata = json.load(file)
    if training_metadata.get("target") != TARGET or training_metadata.get("dataset_run") != DATASET_RUN:
        raise ValueError("F1 training metadata target or dataset run is incompatible with F3-A.")
    if training_metadata.get("shared_split_hash") != split["split_hash"]:
        raise ValueError("F1 training metadata and shared split hashes differ.")

    selection = pd.read_csv(f1_selection_file)
    if len(selection) != 1:
        raise ValueError("Frozen F1 selection file must contain exactly one selected equation.")
    selected_candidate = int(selection.iloc[0]["candidate_index"])
    if selected_candidate != 15:
        raise ValueError(f"Expected frozen Re F1 candidate 15, found {selected_candidate}.")
    expression = str(selection.iloc[0]["expression_for_eval"])

    log("Reading only train and validation curve blocks from the symbolic dataset.")
    train = load_partition_rows(dataset_file, train_curve_ids, 1000)
    validation = load_partition_rows(dataset_file, validation_curve_ids, 1000)
    train = prepare_f1_features(train, training_metadata)
    validation = prepare_f1_features(validation, training_metadata)
    variable_names = make_pysr_variable_names(train["f1_feature_names"])
    if variable_names != list(training_metadata["pysr_variable_names"]):
        raise ValueError("F1 PySR variable names do not replay exactly.")

    train_f1 = predict_expression(expression, train["X_f1"], variable_names)
    validation_f1 = predict_expression(expression, validation["X_f1"], variable_names)
    if not np.all(np.isfinite(train_f1)) or not np.all(np.isfinite(validation_f1)):
        raise ValueError("Frozen F1 reconstruction produced non-finite predictions.")
    reconstructed_rmse = regression_metrics(validation["y"], validation_f1)["rmse"]
    if abs(reconstructed_rmse - F1_VALIDATION_RMSE) > F1_RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            "Frozen F1 validation reconstruction mismatch: "
            f"expected {F1_VALIDATION_RMSE:.15f}, got {reconstructed_rmse:.15f}."
        )
    log(f"Reconstructed frozen Re F1 validation RMSE={reconstructed_rmse:.12f}.")

    sigma_index = train["base_feature_names"].index("log10_sigma")
    sigma_stats = standardization(
        train["X_base"][:, sigma_index], train["source_curve_index"]
    )
    train_z_sigma = apply_standardization(train["X_base"][:, sigma_index], sigma_stats)
    validation_z_sigma = apply_standardization(
        validation["X_base"][:, sigma_index], sigma_stats
    )
    train_envelopes = build_envelopes(train["frequency_hz"])
    validation_envelopes = build_envelopes(validation["frequency_hz"])
    train_terms = {
        f"z_sigma_x_{name}": train_z_sigma * envelope
        for name, envelope in train_envelopes.items()
    }
    validation_terms = {
        f"z_sigma_x_{name}": validation_z_sigma * envelope
        for name, envelope in validation_envelopes.items()
    }
    sigma_candidates = {
        "zero_correction": [],
        "sigma_g1450": ["z_sigma_x_g1450"],
        "sigma_g1750": ["z_sigma_x_g1750"],
        "sigma_g1450_g1750": ["z_sigma_x_g1450", "z_sigma_x_g1750"],
        "sigma_bandpass_1250_2100": ["z_sigma_x_g_bandpass_1250_2100"],
    }

    train_residual = train["y"] - train_f1
    candidate_records: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}

    def evaluate_candidate(
        name: str,
        family: str,
        term_names: list[str],
        train_term_map: dict[str, np.ndarray],
        validation_term_map: dict[str, np.ndarray],
        eligible_for_stage_decision: bool,
    ) -> dict[str, Any]:
        model, train_correction = fit_ridge(term_names, train_term_map, train_residual)
        validation_correction = predict_ridge(
            model, term_names, validation_term_map, validation["y"].size
        )
        train_prediction = train_f1 + train_correction
        validation_prediction = validation_f1 + validation_correction
        train_curve_summary, train_curve_rows = curve_metrics(
            train["y"], train_prediction, train["source_curve_index"]
        )
        validation_curve_summary, validation_curve_rows = curve_metrics(
            validation["y"], validation_prediction, validation["source_curve_index"]
        )
        all_regions = (*REPORTING_REGIONS, TARGET_REGION)
        record = {
            "name": name,
            "family": family,
            "eligible_for_stage_decision": eligible_for_stage_decision,
            "term_names": term_names,
            "ridge_alpha": RIDGE_ALPHA,
            "fit_intercept": False,
            "coefficients": (
                []
                if model is None
                else [float(value) for value in np.asarray(model.coef_).reshape(-1)]
            ),
            "train_metrics": regression_metrics(train["y"], train_prediction),
            "validation_metrics": regression_metrics(validation["y"], validation_prediction),
            "train_curve_summary": train_curve_summary,
            "validation_curve_summary": validation_curve_summary,
            "train_regions": regional_metrics(
                train["y"], train_prediction, train["frequency_hz"], all_regions
            ),
            "validation_regions": regional_metrics(
                validation["y"], validation_prediction, validation["frequency_hz"], all_regions
            ),
            "validation_frequency_metrics": frequency_wise_metrics(
                validation["y"], validation_prediction, validation["frequency_hz"]
            ),
            "train_curve_rows": train_curve_rows,
            "validation_curve_rows": validation_curve_rows,
        }
        predictions[name] = {
            "train": train_prediction,
            "validation": validation_prediction,
        }
        candidate_records.append(record)
        return record

    log("Fitting the zero control and four predeclared sigma-local ridge candidates.")
    for candidate_name, term_names in sigma_candidates.items():
        evaluate_candidate(
            candidate_name,
            "control" if candidate_name == "zero_correction" else "sigma_local",
            term_names,
            train_terms,
            validation_terms,
            candidate_name != "zero_correction",
        )

    baseline = next(record for record in candidate_records if record["name"] == "zero_correction")
    sigma_records = [record for record in candidate_records if record["family"] == "sigma_local"]
    for record in sigma_records:
        passed, reasons, guardrails = candidate_passes(record, baseline)
        record["passes_stage_rules"] = passed
        record["failure_reasons"] = reasons
        record["guardrails"] = guardrails
    raw_best_sigma = min(sigma_records, key=lambda record: record["validation_metrics"]["rmse"])
    feasible_sigma = [record for record in sigma_records if record["passes_stage_rules"]]
    selected_sigma = (
        min(feasible_sigma, key=lambda record: record["validation_metrics"]["rmse"])
        if feasible_sigma
        else None
    )
    sigma_shows_gain = (
        raw_best_sigma["validation_metrics"]["rmse"]
        < baseline["validation_metrics"]["rmse"]
    )

    standardization_stats: dict[str, Any] = {"log10_sigma": sigma_stats}
    optional_diagnostic_names: list[str] = []
    if sigma_shows_gain:
        log(
            f"Sigma shows a validation gain with {raw_best_sigma['name']}; "
            "running the conditional lambda/alpha ridge diagnostic."
        )
        best_envelope_suffixes = [
            name.removeprefix("z_sigma_x_") for name in raw_best_sigma["term_names"]
        ]
        for base_name, short_name in (
            ("log10_lambda", "lambda"),
            ("alpha_infinity", "alpha"),
        ):
            feature_index = train["base_feature_names"].index(base_name)
            stats = standardization(
                train["X_base"][:, feature_index], train["source_curve_index"]
            )
            standardization_stats[base_name] = stats
            train_z = apply_standardization(train["X_base"][:, feature_index], stats)
            validation_z = apply_standardization(
                validation["X_base"][:, feature_index], stats
            )
            for suffix in best_envelope_suffixes:
                train_terms[f"z_{short_name}_x_{suffix}"] = train_z * train_envelopes[suffix]
                validation_terms[f"z_{short_name}_x_{suffix}"] = (
                    validation_z * validation_envelopes[suffix]
                )
        sigma_term_names = list(raw_best_sigma["term_names"])
        lambda_terms = [f"z_lambda_x_{suffix}" for suffix in best_envelope_suffixes]
        alpha_terms = [f"z_alpha_x_{suffix}" for suffix in best_envelope_suffixes]
        optional_definitions = {
            f"diagnostic_{raw_best_sigma['name']}_plus_lambda": sigma_term_names + lambda_terms,
            f"diagnostic_{raw_best_sigma['name']}_plus_alpha": sigma_term_names + alpha_terms,
            f"diagnostic_{raw_best_sigma['name']}_plus_lambda_alpha": (
                sigma_term_names + lambda_terms + alpha_terms
            ),
        }
        for candidate_name, term_names in optional_definitions.items():
            optional_diagnostic_names.append(candidate_name)
            evaluate_candidate(
                candidate_name,
                "optional_material_diagnostic",
                term_names,
                train_terms,
                validation_terms,
                False,
            )
    else:
        log("No sigma-local candidate improves full validation RMSE; skipping lambda/alpha diagnostic.")

    plot_record = selected_sigma or raw_best_sigma
    stage_passed = selected_sigma is not None
    decision_reasons: list[str]
    if stage_passed:
        decision_reasons = [
            f"{selected_sigma['name']} improves full validation RMSE.",
            f"{selected_sigma['name']} improves the combined 1300-2000 Hz validation RMSE.",
            "All declared regional RMSE guardrails remain within the 10% limit.",
        ]
    else:
        decision_reasons = [
            "No predeclared sigma-local candidate satisfies every F3-A pass condition.",
            *[f"Raw best {raw_best_sigma['name']}: {reason}" for reason in raw_best_sigma["failure_reasons"]],
        ]

    optional_records = [
        record for record in candidate_records if record["name"] in optional_diagnostic_names
    ]
    best_optional = (
        min(optional_records, key=lambda record: record["validation_metrics"]["rmse"])
        if optional_records
        else None
    )
    decision = {
        "stage": "F3-A",
        "stage_passed": stage_passed,
        "selected_sigma_candidate": None if selected_sigma is None else selected_sigma["name"],
        "raw_best_sigma_candidate": raw_best_sigma["name"],
        "plot_and_diagnostic_candidate": plot_record["name"],
        "sigma_shows_full_validation_gain": sigma_shows_gain,
        "conditional_material_diagnostic_performed": bool(optional_records),
        "best_optional_material_diagnostic": None if best_optional is None else best_optional["name"],
        "decision_reasons": decision_reasons,
        "test_rows_accessed": False,
        "next_stage_automatically_authorized": False,
        "next_command_if_user_accepts": "执行 F3-B",
    }

    overall_rows: list[dict[str, Any]] = []
    regional_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    guardrail_rows: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for record in candidate_records:
        models.append(
            {
                key: record[key]
                for key in (
                    "name",
                    "family",
                    "eligible_for_stage_decision",
                    "term_names",
                    "ridge_alpha",
                    "fit_intercept",
                    "coefficients",
                )
            }
        )
        for split_name in ("train", "validation"):
            overall_rows.append(
                {
                    "candidate": record["name"],
                    "family": record["family"],
                    "split": split_name,
                    **record[f"{split_name}_metrics"],
                    **record[f"{split_name}_curve_summary"],
                }
            )
            for row in record[f"{split_name}_regions"]:
                regional_rows.append(
                    {"candidate": record["name"], "split": split_name, **row}
                )
        for row in record["validation_frequency_metrics"]:
            frequency_rows.append({"candidate": record["name"], **row})
        for split_name in ("train", "validation"):
            for row in record[f"{split_name}_curve_rows"]:
                curve_rows.append(
                    {"candidate": record["name"], "split": split_name, **row}
                )
        for row in record.get("guardrails", []):
            guardrail_rows.append({"candidate": record["name"], **row})

    pd.DataFrame(overall_rows).to_csv(output_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(regional_rows).to_csv(output_dir / "regional_metrics.csv", index=False)
    pd.DataFrame(frequency_rows).to_csv(
        output_dir / "validation_frequency_metrics.csv", index=False
    )
    pd.DataFrame(curve_rows).to_csv(output_dir / "curve_metrics.csv", index=False)
    pd.DataFrame(guardrail_rows).to_csv(
        output_dir / "sigma_guardrail_comparison.csv", index=False
    )
    dump_json(output_dir / "ridge_models.json", models)
    dump_json(output_dir / "training_only_standardization.json", standardization_stats)
    dump_json(output_dir / "decision_report.json", decision)
    write_decision_markdown(output_dir / "decision_report.md", decision)

    validation_plot_ids = select_overlay_curve_ids(
        baseline["validation_curve_rows"], args.num_overlay_curves
    )
    plot_envelopes(
        validation["frequency_grid"],
        build_envelopes(validation["frequency_grid"]),
        figures_dir / "predeclared_envelopes.png",
    )
    plot_candidate_rmse(candidate_records, figures_dir / "candidate_validation_rmse.png")
    plot_frequency_rmse(
        candidate_records, plot_record["name"], figures_dir / "validation_frequency_rmse.png"
    )
    plot_curve_overlays(
        validation,
        validation_f1,
        predictions[plot_record["name"]]["validation"],
        plot_record["name"],
        validation_plot_ids,
        figures_dir / "validation_full_curve_overlays.png",
    )
    plot_mean_residual_by_frequency(
        validation,
        validation_f1,
        predictions[plot_record["name"]]["validation"],
        plot_record["name"],
        figures_dir / "validation_mean_residual_by_frequency.png",
    )

    np.savez_compressed(
        output_dir / "f1_and_selected_sigma_predictions.npz",
        train_curve_ids=train["source_curve_index"],
        train_frequency_hz=train["frequency_hz"],
        train_teacher=train["y"],
        train_f1_prediction=train_f1,
        train_selected_sigma_prediction=predictions[plot_record["name"]]["train"],
        validation_curve_ids=validation["source_curve_index"],
        validation_frequency_hz=validation["frequency_hz"],
        validation_teacher=validation["y"],
        validation_f1_prediction=validation_f1,
        validation_selected_sigma_prediction=predictions[plot_record["name"]]["validation"],
        selected_sigma_candidate=np.asarray(plot_record["name"]),
    )

    manifest = {
        "schema_version": 1,
        "stage": "F3-A",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Re Global SR cheap linear residual diagnosis; no PySR search",
        "dataset_run": DATASET_RUN,
        "target": TARGET,
        "target_name": EXPECTED_TARGET_NAME,
        "dataset_file": str(dataset_file),
        "dataset_sha256": sha256_file(dataset_file),
        "shared_split_file": str(split_file),
        "shared_split_hash": split["split_hash"],
        "f1_training_dir": str(f1_training_dir),
        "f1_training_metadata_file": str(training_metadata_file),
        "f1_training_metadata_sha256": sha256_file(training_metadata_file),
        "f1_selection_file": str(f1_selection_file),
        "f1_selection_sha256": sha256_file(f1_selection_file),
        "f1_selected_candidate": selected_candidate,
        "f1_expression": expression,
        "f1_expected_validation_rmse": F1_VALIDATION_RMSE,
        "f1_reconstructed_validation_rmse": reconstructed_rmse,
        "output_dir": str(output_dir),
        "data_access": {
            "partitions_loaded": ["train", "validation"],
            "train_curve_count": int(train["curve_ids"].size),
            "validation_curve_count": int(validation["curve_ids"].size),
            "train_row_count": int(train["y"].size),
            "validation_row_count": int(validation["y"].size),
            "test_curve_count_declared_but_not_loaded": int(test_curve_ids.size),
            "test_features_read": False,
            "test_targets_read": False,
            "test_predictions_computed": False,
        },
        "standardization": standardization_stats,
        "predeclared_envelopes": {
            "frequency_coordinate": "log10_hz",
            "gaussians": [
                {"center_hz": 1450.0, "width_log10": GAUSSIAN_WIDTH_LOG10},
                {"center_hz": 1750.0, "width_log10": GAUSSIAN_WIDTH_LOG10},
            ],
            "bandpass": {
                "definition": "logistic_highpass(1250) * logistic_lowpass(2100), normalized to unit maximum",
                "low_hz": BANDPASS_LOW_HZ,
                "high_hz": BANDPASS_HIGH_HZ,
                "transition_log10": BANDPASS_TRANSITION_LOG10,
            },
        },
        "ridge": {"alpha": RIDGE_ALPHA, "fit_intercept": False},
        "decision_contract": {
            "selection_split": "complete validation curves",
            "full_validation_rmse_must_improve": True,
            "target_region_hz": list(TARGET_REGION),
            "target_region_rmse_must_improve": True,
            "guardrail_regions_hz": [list(bounds) for bounds in GUARDRAIL_REGIONS],
            "maximum_relative_guardrail_rmse_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
            "test_rows_inaccessible": True,
        },
        "decision": decision,
        "artifact_files": sorted(
            str(path.relative_to(output_dir)).replace("\\", "/")
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    dump_json(output_dir / "stage_manifest.json", manifest)
    log(
        f"F3-A complete: stage_passed={stage_passed}, "
        f"selected_sigma={decision['selected_sigma_candidate']}, "
        f"raw_best_sigma={raw_best_sigma['name']}."
    )
    log(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
