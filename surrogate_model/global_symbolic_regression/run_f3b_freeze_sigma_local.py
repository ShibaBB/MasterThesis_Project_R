"""Freeze and replay-verify the Re Global SR F3-B sigma-local terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_ROOT = SCRIPT_DIR.parent
if str(SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURROGATE_ROOT))

from shared_split_utils import load_shared_split  # noqa: E402
from f3_local_terminals import (  # noqa: E402
    apply_f3_local_terminals,
    build_re_f3b_sigma_spec,
    validate_f3_local_terminal_spec,
)
from run_f3a_residual_diagnosis import (  # noqa: E402
    DEFAULT_DATASET_FILE,
    DEFAULT_F1_TRAINING_DIR,
    DEFAULT_SPLIT_FILE,
    F3_ARTIFACT_ROOT,
    bandpass_envelope,
    load_partition_rows,
    prepare_f1_features,
    sha256_file,
    standardization,
)


DEFAULT_F3A_DIR = F3_ARTIFACT_ROOT / "F3-A" / "20260816T153654"
EXPECTED_F3A_SELECTION = "sigma_bandpass_1250_2100"
TERMINAL_NAME = "z_sigma__f3_mid"
F3A_RIDGE_COEFFICIENT_TOLERANCE = 0.0
REPLAY_TOLERANCE = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the F3-B Re sigma-local terminal and verify exact replay."
    )
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--f1-training-dir", type=Path, default=DEFAULT_F1_TRAINING_DIR)
    parser.add_argument("--f3a-dir", type=Path, default=DEFAULT_F3A_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory; it must remain under artifacts/F3.",
    )
    return parser.parse_args()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for suffix in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique output path near {path}.")


def resolve_output_dir(requested: Path | None) -> Path:
    root = F3_ARTIFACT_ROOT.resolve()
    if requested is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = unique_path(root / "F3-B" / timestamp)
    else:
        output_dir = requested.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError(f"F3-B artifacts must stay under {root}: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty F3-B directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def terminal_summary(
    split_name: str,
    values: np.ndarray,
    curve_ids: np.ndarray,
    frequency_hz: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    return {
        "split": split_name,
        "row_count": int(values.size),
        "curve_count": int(np.unique(curve_ids).size),
        "frequency_count": int(np.unique(frequency_hz).size),
        "finite_count": int(np.sum(finite)),
        "nonfinite_count": int(np.sum(~finite)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=0)),
        "sha256_float64_bytes": array_sha256(values),
    }


def sampled_replay_rows(
    split_name: str,
    curve_ids: np.ndarray,
    frequency_hz: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> list[dict[str, Any]]:
    indices = sorted({0, len(before) // 2, len(before) - 1})
    return [
        {
            "split": split_name,
            "row_index_0based": index,
            "curve_id": int(curve_ids[index]),
            "frequency_hz": float(frequency_hz[index]),
            "terminal_before_serialization": float(before[index]),
            "terminal_after_reload": float(after[index]),
            "absolute_difference": float(abs(before[index] - after[index])),
        }
        for index in indices
    ]


def plot_terminal_by_frequency(
    validation_frequency_hz: np.ndarray,
    validation_terminal: np.ndarray,
    path: Path,
) -> None:
    frequencies = np.unique(validation_frequency_hz)
    means: list[float] = []
    p05: list[float] = []
    p95: list[float] = []
    for frequency in frequencies:
        values = validation_terminal[validation_frequency_hz == frequency]
        means.append(float(np.mean(values)))
        p05.append(float(np.quantile(values, 0.05)))
        p95.append(float(np.quantile(values, 0.95)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(frequencies, p05, p95, alpha=0.25, color="#2878b5", label="5th–95th percentile")
    ax.plot(frequencies, means, linewidth=2, color="#2878b5", label="validation mean")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(TERMINAL_NAME)
    ax.set_title("Frozen F3-B sigma-local terminal across validation curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_standardized_sigma(
    train_z_sigma: np.ndarray,
    train_curve_ids: np.ndarray,
    validation_z_sigma: np.ndarray,
    validation_curve_ids: np.ndarray,
    path: Path,
) -> None:
    _, train_first = np.unique(train_curve_ids, return_index=True)
    _, validation_first = np.unique(validation_curve_ids, return_index=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(float(np.min(train_z_sigma[train_first])), float(np.min(validation_z_sigma[validation_first]))),
        max(float(np.max(train_z_sigma[train_first])), float(np.max(validation_z_sigma[validation_first]))),
        25,
    )
    ax.hist(train_z_sigma[train_first], bins=bins, alpha=0.6, label="train (fit source)")
    ax.hist(validation_z_sigma[validation_first], bins=bins, alpha=0.6, label="validation (replay only)")
    ax.set_xlabel("z_sigma using frozen training statistics")
    ax.set_ylabel("Unique material curves")
    ax.set_title("Training-only z_sigma normalization check")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_decision_markdown(path: Path, report: dict[str, Any]) -> None:
    status = "PASS" if report["stage_passed"] else "FAIL"
    replay = report["exact_reload_replay"]
    f3a = report["f3a_equivalence"]
    lines = [
        "# Re Global SR F3-B Decision Report",
        "",
        f"- Stage decision: **{status}**",
        f"- Frozen terminal: `{report['frozen_terminal_name']}`",
        f"- Frozen representation: `{report['frozen_representation']}`",
        f"- Exact JSON reload/replay: `{replay['passed']}`",
        f"- Maximum reload difference: `{replay['maximum_absolute_difference']:.17g}`",
        f"- Exact F3-A terminal equivalence: `{f3a['terminal_values_exact']}`",
        f"- Exact F3-A corrected-prediction equivalence: `{f3a['corrected_predictions_exact']}`",
        f"- All output values finite: `{report['all_values_finite']}`",
        f"- Training-only normalization verified: `{report['training_only_normalization_verified']}`",
        f"- Test rows accessed: `{report['test_rows_accessed']}`",
        "",
        "The terminal specification, feature order, normalization constants,",
        "transform diagnostics, and replay samples are stored beside this report.",
        "F3-C requires a new explicit user command.",
        "",
    ]
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    execution_log = output_dir / "execution.log"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with execution_log.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    log("Starting F3-B sigma-local terminal freeze and transform-level validation.")
    dataset_file = args.dataset_file.resolve()
    split_file = args.split_file.resolve()
    f1_training_dir = args.f1_training_dir.resolve()
    f3a_dir = args.f3a_dir.resolve()
    f1_metadata_file = f1_training_dir / "training_metadata.json"
    f3a_decision_file = f3a_dir / "decision_report.json"
    f3a_standardization_file = f3a_dir / "training_only_standardization.json"
    f3a_models_file = f3a_dir / "ridge_models.json"
    f3a_predictions_file = f3a_dir / "f1_and_selected_sigma_predictions.npz"
    f3a_manifest_file = f3a_dir / "stage_manifest.json"
    required = (
        dataset_file,
        split_file,
        f1_metadata_file,
        f3a_decision_file,
        f3a_standardization_file,
        f3a_models_file,
        f3a_predictions_file,
        f3a_manifest_file,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"F3-B required inputs are missing: {missing}")

    with f3a_decision_file.open("r", encoding="utf-8") as file:
        f3a_decision = json.load(file)
    with f3a_standardization_file.open("r", encoding="utf-8") as file:
        f3a_standardizations = json.load(file)
    with f3a_models_file.open("r", encoding="utf-8") as file:
        f3a_models = json.load(file)
    with f3a_manifest_file.open("r", encoding="utf-8") as file:
        f3a_manifest = json.load(file)
    with f1_metadata_file.open("r", encoding="utf-8") as file:
        f1_metadata = json.load(file)

    if not f3a_decision.get("stage_passed"):
        raise ValueError("F3-A did not pass, so F3-B cannot freeze a representation.")
    if f3a_decision.get("selected_sigma_candidate") != EXPECTED_F3A_SELECTION:
        raise ValueError("F3-A selected representation does not match the F3-B contract.")
    if f3a_decision.get("test_rows_accessed") is not False:
        raise ValueError("F3-A does not provide a clean validation-only provenance.")
    if f3a_manifest.get("shared_split_hash") != f1_metadata.get("shared_split_hash"):
        raise ValueError("F3-A and Re F1 use different shared splits.")
    if f3a_manifest["predeclared_envelopes"]["bandpass"] != {
        "definition": "logistic_highpass(1250) * logistic_lowpass(2100), normalized to unit maximum",
        "low_hz": 1250.0,
        "high_hz": 2100.0,
        "transition_log10": 0.06,
    }:
        raise ValueError("F3-A selected band-pass constants differ from the expected freeze.")

    split = load_shared_split(
        split_file, expected_dataset_run="run1", expected_curve_count=1000
    )
    train_curve_ids = np.asarray(split["train_curve_indices"], dtype=int)
    validation_curve_ids = np.asarray(split["validation_curve_indices"], dtype=int)
    test_curve_ids = np.asarray(split["test_curve_indices"], dtype=int)
    if np.intersect1d(np.concatenate([train_curve_ids, validation_curve_ids]), test_curve_ids).size:
        raise ValueError("Allowed F3-B partitions overlap the forbidden test curves.")

    log("Reading only train and validation curve blocks for transform verification.")
    train = prepare_f1_features(
        load_partition_rows(dataset_file, train_curve_ids, 1000), f1_metadata
    )
    validation = prepare_f1_features(
        load_partition_rows(dataset_file, validation_curve_ids, 1000), f1_metadata
    )
    base_feature_names = list(train["base_feature_names"])
    if base_feature_names != list(validation["base_feature_names"]):
        raise ValueError("Training and validation base feature orders differ.")

    sigma_stats = f3a_standardizations["log10_sigma"]
    specification = build_re_f3b_sigma_spec(
        base_feature_names,
        sigma_center=float(sigma_stats["center"]),
        sigma_scale=float(sigma_stats["scale"]),
        split_hash=split["split_hash"],
        source_f3a_dir=str(f3a_dir),
    )
    specification["f3a_selection_evidence"] = {
        "selected_candidate": EXPECTED_F3A_SELECTION,
        "ridge_coefficient_reference_only": next(
            model["coefficients"][0]
            for model in f3a_models
            if model["name"] == EXPECTED_F3A_SELECTION
        ),
        "ridge_coefficient_is_part_of_terminal": False,
        "decision_file": str(f3a_decision_file),
        "decision_file_sha256": sha256_file(f3a_decision_file),
    }
    specification = validate_f3_local_terminal_spec(specification, base_feature_names)

    train_before, output_feature_names = apply_f3_local_terminals(
        train["X_base"], base_feature_names, specification
    )
    validation_before, validation_output_names = apply_f3_local_terminals(
        validation["X_base"], base_feature_names, specification
    )
    if validation_output_names != output_feature_names:
        raise ValueError("F3-B output feature order differs by partition.")

    specification_file = output_dir / "frozen_sigma_local_spec.json"
    dump_json(specification_file, specification)
    with specification_file.open("r", encoding="utf-8") as file:
        reloaded_specification = json.load(file)
    reloaded_specification = validate_f3_local_terminal_spec(
        reloaded_specification, base_feature_names
    )
    train_after, train_after_names = apply_f3_local_terminals(
        train["X_base"], base_feature_names, reloaded_specification
    )
    validation_after, validation_after_names = apply_f3_local_terminals(
        validation["X_base"], base_feature_names, reloaded_specification
    )

    train_terminal_before = train_before[:, -1]
    validation_terminal_before = validation_before[:, -1]
    train_terminal_after = train_after[:, -1]
    validation_terminal_after = validation_after[:, -1]
    replay_max_difference = max(
        float(np.max(np.abs(train_before - train_after))),
        float(np.max(np.abs(validation_before - validation_after))),
    )
    replay_exact = (
        np.array_equal(train_before, train_after)
        and np.array_equal(validation_before, validation_after)
        and output_feature_names == train_after_names == validation_after_names
    )

    sigma_index = base_feature_names.index("log10_sigma")
    recomputed_training_stats = standardization(
        train["X_base"][:, sigma_index], train["source_curve_index"]
    )
    recomputed_validation_stats = standardization(
        validation["X_base"][:, sigma_index], validation["source_curve_index"]
    )
    recomputed_validation_stats["method"] = (
        "descriptive_validation_unique_curve_population_mean_std_ddof0_not_used_for_transform"
    )
    recomputed_validation_stats["validation_curve_count"] = (
        recomputed_validation_stats.pop("training_curve_count")
    )
    training_center_exact = float(recomputed_training_stats["center"]) == float(
        sigma_stats["center"]
    )
    training_scale_exact = float(recomputed_training_stats["scale"]) == float(
        sigma_stats["scale"]
    )
    normalization_verified = training_center_exact and training_scale_exact
    train_z_sigma = (
        train["X_base"][:, sigma_index] - float(sigma_stats["center"])
    ) / float(sigma_stats["scale"])
    validation_z_sigma = (
        validation["X_base"][:, sigma_index] - float(sigma_stats["center"])
    ) / float(sigma_stats["scale"])

    legacy_train_terminal = train_z_sigma * bandpass_envelope(train["frequency_hz"])
    legacy_validation_terminal = validation_z_sigma * bandpass_envelope(
        validation["frequency_hz"]
    )
    legacy_terminal_max_difference = max(
        float(np.max(np.abs(legacy_train_terminal - train_terminal_after))),
        float(np.max(np.abs(legacy_validation_terminal - validation_terminal_after))),
    )
    terminal_values_exact = (
        np.array_equal(legacy_train_terminal, train_terminal_after)
        and np.array_equal(legacy_validation_terminal, validation_terminal_after)
    )

    f3a_ridge_model = next(
        model for model in f3a_models if model["name"] == EXPECTED_F3A_SELECTION
    )
    ridge_coefficient = float(f3a_ridge_model["coefficients"][0])
    with np.load(f3a_predictions_file) as f3a_predictions:
        if not np.array_equal(
            f3a_predictions["train_curve_ids"], train["source_curve_index"]
        ) or not np.array_equal(
            f3a_predictions["validation_curve_ids"], validation["source_curve_index"]
        ):
            raise ValueError("F3-A prediction rows do not align with F3-B partitions.")
        replayed_train_prediction = (
            f3a_predictions["train_f1_prediction"] + ridge_coefficient * train_terminal_after
        )
        replayed_validation_prediction = (
            f3a_predictions["validation_f1_prediction"]
            + ridge_coefficient * validation_terminal_after
        )
        train_prediction_difference = float(
            np.max(
                np.abs(
                    replayed_train_prediction
                    - f3a_predictions["train_selected_sigma_prediction"]
                )
            )
        )
        validation_prediction_difference = float(
            np.max(
                np.abs(
                    replayed_validation_prediction
                    - f3a_predictions["validation_selected_sigma_prediction"]
                )
            )
        )
        corrected_predictions_exact = (
            np.array_equal(
                replayed_train_prediction,
                f3a_predictions["train_selected_sigma_prediction"],
            )
            and np.array_equal(
                replayed_validation_prediction,
                f3a_predictions["validation_selected_sigma_prediction"],
            )
        )

    summaries = [
        terminal_summary(
            "train",
            train_terminal_after,
            train["source_curve_index"],
            train["frequency_hz"],
        ),
        terminal_summary(
            "validation",
            validation_terminal_after,
            validation["source_curve_index"],
            validation["frequency_hz"],
        ),
    ]
    all_values_finite = all(summary["nonfinite_count"] == 0 for summary in summaries)
    feature_order_exact = (
        output_feature_names == specification["output_feature_names"]
        and train_after_names == validation_after_names == output_feature_names
    )
    checks = {
        "f3a_stage_passed": bool(f3a_decision["stage_passed"]),
        "selected_representation_matches": (
            f3a_decision["selected_sigma_candidate"] == EXPECTED_F3A_SELECTION
        ),
        "training_center_exact": training_center_exact,
        "training_scale_exact": training_scale_exact,
        "normalization_source_is_train": (
            specification["standardizations"]["z_sigma"]["fitted_partition"] == "train"
        ),
        "all_values_finite": all_values_finite,
        "feature_order_exact": feature_order_exact,
        "reload_replay_exact": replay_exact,
        "f3a_terminal_values_exact": terminal_values_exact,
        "f3a_corrected_predictions_exact": corrected_predictions_exact,
        "test_rows_not_accessed": True,
    }
    stage_passed = all(value is True for value in checks.values())
    report = {
        "stage": "F3-B",
        "stage_passed": stage_passed,
        "frozen_terminal_name": TERMINAL_NAME,
        "frozen_representation": "z_sigma * smooth_log10_bandpass(1250 Hz, 2100 Hz)",
        "input_feature_names": base_feature_names,
        "output_feature_names": output_feature_names,
        "all_values_finite": all_values_finite,
        "training_only_normalization_verified": normalization_verified,
        "normalization_check": {
            "stored_training_stats": sigma_stats,
            "recomputed_training_stats": recomputed_training_stats,
            "descriptive_validation_stats_not_used_for_transform": recomputed_validation_stats,
            "training_center_exact": training_center_exact,
            "training_scale_exact": training_scale_exact,
        },
        "exact_reload_replay": {
            "passed": replay_exact,
            "tolerance": REPLAY_TOLERANCE,
            "maximum_absolute_difference": replay_max_difference,
            "feature_order_exact": feature_order_exact,
            "train_output_sha256": array_sha256(train_after),
            "validation_output_sha256": array_sha256(validation_after),
        },
        "f3a_equivalence": {
            "terminal_values_exact": terminal_values_exact,
            "maximum_terminal_absolute_difference": legacy_terminal_max_difference,
            "corrected_predictions_exact": corrected_predictions_exact,
            "maximum_train_prediction_absolute_difference": train_prediction_difference,
            "maximum_validation_prediction_absolute_difference": validation_prediction_difference,
            "ridge_coefficient_reference_only": ridge_coefficient,
            "ridge_coefficient_tolerance": F3A_RIDGE_COEFFICIENT_TOLERANCE,
        },
        "terminal_summaries": summaries,
        "checks": checks,
        "test_rows_accessed": False,
        "next_stage_automatically_authorized": False,
        "next_command_if_user_accepts": "执行 F3-C",
    }

    replay_rows = [
        *sampled_replay_rows(
            "train",
            train["source_curve_index"],
            train["frequency_hz"],
            train_terminal_before,
            train_terminal_after,
        ),
        *sampled_replay_rows(
            "validation",
            validation["source_curve_index"],
            validation["frequency_hz"],
            validation_terminal_before,
            validation_terminal_after,
        ),
    ]
    pd.DataFrame(summaries).to_csv(output_dir / "terminal_summary.csv", index=False)
    pd.DataFrame(replay_rows).to_csv(output_dir / "replay_samples.csv", index=False)
    dump_json(output_dir / "transform_validation_report.json", report)
    dump_json(output_dir / "decision_report.json", report)
    write_decision_markdown(output_dir / "decision_report.md", report)
    np.savez_compressed(
        output_dir / "terminal_values_train_validation.npz",
        train_curve_ids=train["source_curve_index"],
        train_frequency_hz=train["frequency_hz"],
        train_terminal=train_terminal_after,
        validation_curve_ids=validation["source_curve_index"],
        validation_frequency_hz=validation["frequency_hz"],
        validation_terminal=validation_terminal_after,
        output_feature_names=np.asarray(output_feature_names),
    )
    plot_terminal_by_frequency(
        validation["frequency_hz"],
        validation_terminal_after,
        figures_dir / "frozen_terminal_by_frequency.png",
    )
    plot_standardized_sigma(
        train_z_sigma,
        train["source_curve_index"],
        validation_z_sigma,
        validation["source_curve_index"],
        figures_dir / "training_only_z_sigma_normalization.png",
    )

    manifest = {
        "schema_version": 1,
        "stage": "F3-B",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Freeze, serialize, reload, and replay the selected Re sigma-local terminal",
        "dataset_run": "run1",
        "target": "re",
        "dataset_file": str(dataset_file),
        "dataset_sha256": sha256_file(dataset_file),
        "shared_split_file": str(split_file),
        "shared_split_hash": split["split_hash"],
        "f1_training_dir": str(f1_training_dir),
        "source_f3a_dir": str(f3a_dir),
        "source_f3a_files": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in (
                f3a_decision_file,
                f3a_standardization_file,
                f3a_models_file,
                f3a_predictions_file,
                f3a_manifest_file,
            )
        },
        "frozen_spec_file": str(specification_file),
        "frozen_spec_sha256": sha256_file(specification_file),
        "data_access": {
            "partitions_loaded": ["train", "validation"],
            "train_curve_count": int(np.unique(train["source_curve_index"]).size),
            "validation_curve_count": int(np.unique(validation["source_curve_index"]).size),
            "train_row_count": int(train["X_base"].shape[0]),
            "validation_row_count": int(validation["X_base"].shape[0]),
            "test_curve_count_declared_but_not_loaded": int(test_curve_ids.size),
            "test_features_read": False,
            "test_targets_read": False,
            "test_predictions_computed": False,
        },
        "decision": report,
        "next_stage_automatically_authorized": False,
        "artifact_files": sorted(
            [
                *(
                    str(path.relative_to(output_dir)).replace("\\", "/")
                    for path in output_dir.rglob("*")
                    if path.is_file()
                ),
                "stage_manifest.json",
            ]
        ),
    }
    dump_json(output_dir / "stage_manifest.json", manifest)
    log(
        f"F3-B complete: stage_passed={stage_passed}, replay_exact={replay_exact}, "
        f"f3a_terminal_exact={terminal_values_exact}, test_rows_accessed=False."
    )
    log(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
