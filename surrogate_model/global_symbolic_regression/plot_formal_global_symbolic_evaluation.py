"""Create teacher-comparison figures from saved formal Global SR predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_ARRAYS = (
    "source_curve_index",
    "frequency_hz",
    "teacher_re",
    "teacher_im",
    "predicted_re",
    "predicted_im",
    "predicted_reflect",
)


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"Formal prediction artifact is missing arrays: {missing}")
    validated = {name: np.asarray(arrays[name]).reshape(-1) for name in REQUIRED_ARRAYS}
    row_counts = {values.size for values in validated.values()}
    if len(row_counts) != 1 or not row_counts or next(iter(row_counts)) == 0:
        raise ValueError("Formal prediction arrays do not share one non-empty row order.")
    if not all(np.all(np.isfinite(values)) for values in validated.values()):
        raise ValueError("Formal prediction arrays contain non-finite values.")
    reconstructed = validated["predicted_re"] + 1j * validated["predicted_im"]
    if not np.array_equal(reconstructed, validated["predicted_reflect"]):
        raise ValueError("Saved complex output does not exactly match saved Re/Im rows.")
    return validated


def _scatter(
    teacher: np.ndarray,
    predicted: np.ndarray,
    target_label: str,
    output_file: Path,
) -> None:
    lower = float(min(np.min(teacher), np.min(predicted)))
    upper = float(max(np.max(teacher), np.max(predicted)))
    padding = 0.04 * max(upper - lower, 1e-12)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(teacher, predicted, s=7, alpha=0.22, linewidths=0)
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.2, label="ideal")
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xlabel(f"Teacher {target_label}")
    ax.set_ylabel(f"Formal Global SR {target_label}")
    ax.set_title(f"{target_label}: integrated formal prediction vs teacher")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def _error_vs_frequency(
    frequency: np.ndarray,
    teacher: np.ndarray,
    predicted: np.ndarray,
    target_label: str,
    output_file: Path,
) -> None:
    error = predicted - teacher
    unique_frequency = np.unique(frequency)
    bias = np.asarray([np.mean(error[frequency == value]) for value in unique_frequency])
    rmse = np.asarray(
        [np.sqrt(np.mean(error[frequency == value] ** 2)) for value in unique_frequency]
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.scatter(frequency, error, s=4, alpha=0.08, color="tab:blue", label="row error")
    ax.plot(unique_frequency, bias, color="tab:orange", linewidth=1.8, label="mean error")
    ax.plot(unique_frequency, rmse, color="tab:red", linewidth=1.5, label="frequency RMSE")
    ax.plot(unique_frequency, -rmse, color="tab:red", linewidth=1.0, alpha=0.65)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Prediction - teacher")
    ax.set_title(f"{target_label}: integrated formal error vs frequency")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def _paired_curves(arrays: dict[str, np.ndarray], output_file: Path) -> None:
    curve_ids = np.unique(arrays["source_curve_index"])
    selected_indices = np.linspace(0, curve_ids.size - 1, 4, dtype=int)
    selected_ids = curve_ids[selected_indices]
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 13.0), sharex=True)
    for row, curve_id in enumerate(selected_ids):
        mask = arrays["source_curve_index"] == curve_id
        order = np.argsort(arrays["frequency_hz"][mask])
        frequency = arrays["frequency_hz"][mask][order]
        for column, suffix in enumerate(("re", "im")):
            ax = axes[row, column]
            teacher = arrays[f"teacher_{suffix}"][mask][order]
            predicted = arrays[f"predicted_{suffix}"][mask][order]
            ax.plot(frequency, teacher, color="black", linewidth=1.8, label="teacher")
            ax.plot(
                frequency,
                predicted,
                color="tab:blue" if suffix == "re" else "tab:orange",
                linewidth=1.5,
                label="formal prediction",
            )
            ax.set_ylabel("R_real" if suffix == "re" else "R_imag")
            ax.set_title(f"Curve {int(curve_id)}: {'Re F3' if suffix == 're' else 'Im F2'}")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.legend()
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Integrated formal Global SR curves vs teacher", y=1.002)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _complex_plane(arrays: dict[str, np.ndarray], output_file: Path) -> None:
    teacher = arrays["teacher_re"] + 1j * arrays["teacher_im"]
    predicted = arrays["predicted_reflect"]
    indices = np.linspace(0, teacher.size - 1, min(6000, teacher.size), dtype=int)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(
        teacher.real[indices],
        teacher.imag[indices],
        s=8,
        alpha=0.22,
        color="black",
        label="teacher Reflect",
    )
    ax.scatter(
        predicted.real[indices],
        predicted.imag[indices],
        s=8,
        alpha=0.22,
        color="tab:purple",
        label="formal Re F3 + Im F2",
    )
    ax.set_xlabel("Real component")
    ax.set_ylabel("Imaginary component")
    ax.set_title("Integrated complex reflection: prediction vs teacher")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def _complex_error_vs_frequency(
    arrays: dict[str, np.ndarray], output_file: Path
) -> None:
    teacher = arrays["teacher_re"] + 1j * arrays["teacher_im"]
    error = np.abs(arrays["predicted_reflect"] - teacher)
    frequency = arrays["frequency_hz"]
    unique_frequency = np.unique(frequency)
    mean_error = np.asarray([np.mean(error[frequency == value]) for value in unique_frequency])
    rmse = np.asarray(
        [np.sqrt(np.mean(error[frequency == value] ** 2)) for value in unique_frequency]
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.scatter(frequency, error, s=4, alpha=0.08, label="row |complex error|")
    ax.plot(unique_frequency, mean_error, linewidth=1.8, label="mean |complex error|")
    ax.plot(unique_frequency, rmse, linewidth=1.8, label="complex RMSE")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Complex reflection error magnitude")
    ax.set_title("Integrated complex reflection error vs frequency")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def generate_evaluation_figures(
    arrays: Mapping[str, np.ndarray], output_dir: Path
) -> list[Path]:
    values = _validate_arrays(arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        output_dir / "re_predicted_vs_teacher.png",
        output_dir / "im_predicted_vs_teacher.png",
        output_dir / "re_error_vs_frequency.png",
        output_dir / "im_error_vs_frequency.png",
        output_dir / "paired_curve_comparisons.png",
        output_dir / "complex_plane_comparison.png",
        output_dir / "complex_error_vs_frequency.png",
    ]
    if any(path.exists() for path in files):
        raise FileExistsError(f"Refusing to overwrite existing evaluation figures in {output_dir}")
    _scatter(values["teacher_re"], values["predicted_re"], "R_real", files[0])
    _scatter(values["teacher_im"], values["predicted_im"], "R_imag", files[1])
    _error_vs_frequency(
        values["frequency_hz"], values["teacher_re"], values["predicted_re"], "R_real", files[2]
    )
    _error_vs_frequency(
        values["frequency_hz"], values["teacher_im"], values["predicted_im"], "R_imag", files[3]
    )
    _paired_curves(values, files[4])
    _complex_plane(values, files[5])
    _complex_error_vs_frequency(values, files[6])
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = args.predictions.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else predictions.parent / "figures"
    )
    with np.load(predictions, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    files = generate_evaluation_figures(arrays, output_dir)
    print(f"Generated {len(files)} formal teacher-comparison figures in: {output_dir}")


if __name__ == "__main__":
    main()
