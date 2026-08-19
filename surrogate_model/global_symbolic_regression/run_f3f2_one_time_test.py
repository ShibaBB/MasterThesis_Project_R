"""Run the separately authorized one-time Re F3-F2 test evaluation.

This runner is prepared by F3-F1S but must not be executed until the user gives
an explicit F3-F2 command.  It enforces a per-freeze access receipt before it
loads any test rows, so an interrupted run cannot silently reopen the test set.
"""

from __future__ import annotations

import argparse
import json
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
for import_path in (SCRIPT_DIR, SURROGATE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from evaluate_global_symbolic_candidates import (  # noqa: E402
    make_pysr_variable_names,
    predict_expression,
)
from f3_local_terminals import (  # noqa: E402
    apply_f3_local_terminals,
    validate_f3_local_terminal_spec,
)
from run_f3a_residual_diagnosis import (  # noqa: E402
    F3_ARTIFACT_ROOT,
    GUARDRAIL_MAX_RELATIVE_WORSENING,
    GUARDRAIL_REGIONS,
    REPORTING_REGIONS,
    TARGET_REGION,
    curve_metrics,
    frequency_wise_metrics,
    load_partition_rows,
    prepare_f1_features,
    regional_metrics,
    regression_metrics,
    sha256_file,
)
from shared_split_utils import load_shared_split  # noqa: E402


STAGE = "F3-F2"
EXPECTED_TEST_CURVES = 150
EXPECTED_ROWS_PER_CURVE = 128
EXPECTED_F1_TEST_RMSE_ROUNDED = 0.069470


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        required=True,
        help="Exact passing F3-F1S session directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory; it must remain under artifacts/F3.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


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
        output_dir = unique_path(root / STAGE / datetime.now().strftime("%Y%m%dT%H%M%S"))
    else:
        output_dir = requested.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError(f"{STAGE} artifacts must stay under {root}: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {STAGE} directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def region_map(rows: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (float(row["lower_hz"]), float(row["upper_hz"])): row for row in rows
    }


def verify_frozen_inputs(
    freeze_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    root = F3_ARTIFACT_ROOT.resolve()
    freeze_dir = freeze_dir.resolve()
    if not freeze_dir.is_relative_to(root / "F3-F1S"):
        raise ValueError("--freeze-dir must be an F3-F1S directory under artifacts/F3.")
    spec_path = freeze_dir / "frozen_candidate_spec.json"
    protocol_path = freeze_dir / "f3f2_test_protocol.json"
    decision_path = freeze_dir / "comparison_decision.json"
    manifest_path = freeze_dir / "stage_manifest.json"
    for path in (spec_path, protocol_path, decision_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen F3-F1S input: {path}")
    spec = load_json(spec_path)
    protocol = load_json(protocol_path)
    decision = load_json(decision_path)
    manifest = load_json(manifest_path)
    if decision.get("f3f2_readiness") is not True or decision.get("stage_passed") is not True:
        raise ValueError("F3-F1S did not freeze a test-ready candidate.")
    if decision.get("test_rows_accessed") is not False:
        raise ValueError("F3-F1S does not report a sealed test partition.")
    if protocol.get("explicit_user_command_required") is not True:
        raise ValueError("Frozen protocol does not preserve separate F3-F2 authorization.")
    if protocol.get("frozen_candidate_spec_sha256") != sha256_file(spec_path):
        raise ValueError("Frozen candidate specification hash changed.")
    if protocol.get("f3f2_runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("F3-F2 runner changed after the protocol was frozen.")
    if manifest.get("shared_split_hash") != spec.get("shared_split_hash"):
        raise ValueError("F3-F1S manifest/spec split hashes differ.")
    paths = {
        name: Path(record["path"]).resolve()
        for name, record in protocol["frozen_input_files"].items()
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Frozen F3-F2 input is missing ({name}): {path}")
        expected = protocol["frozen_input_files"][name]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen F3-F2 input hash changed: {name}")
    return spec, protocol, decision, paths


def create_access_receipt(
    freeze_dir: Path, spec_hash: str, output_dir: Path
) -> Path:
    receipt_dir = F3_ARTIFACT_ROOT.resolve() / STAGE / "access_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{spec_hash}.json"
    receipt = {
        "stage": STAGE,
        "status": "test_access_claimed_before_row_load",
        "claimed_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_dir": str(freeze_dir),
        "frozen_candidate_spec_sha256": spec_hash,
        "output_dir": str(output_dir),
        "test_rows_loaded": False,
        "rerun_permitted": False,
    }
    try:
        with receipt_path.open("x", encoding="utf-8") as file:
            json.dump(receipt, file, indent=2, ensure_ascii=False)
    except FileExistsError as error:
        raise RuntimeError(
            "This frozen candidate already has an F3-F2 access receipt; refusing to reopen test. "
            f"Receipt: {receipt_path}"
        ) from error
    return receipt_path


def plot_frequency_metrics(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["frequency_hz"], frame["formal_Re_F1_rmse"], label="formal Re F1")
    ax.plot(frame["frequency_hz"], frame["frozen_Re_F3_rmse"], label="frozen Re F3")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Test RMSE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_curve_metrics(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(frame["formal_Re_F1_rmse"], frame["frozen_Re_F3_rmse"], s=18, alpha=0.7)
    upper = float(max(frame[["formal_Re_F1_rmse", "frozen_Re_F3_rmse"]].max()))
    ax.plot([0.0, upper], [0.0, upper], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Formal Re F1 curve RMSE")
    ax.set_ylabel("Frozen Re F3 curve RMSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    freeze_dir = args.freeze_dir.resolve()
    output_dir = resolve_output_dir(args.output_dir)
    spec, protocol, _, paths = verify_frozen_inputs(freeze_dir)
    spec_path = freeze_dir / "frozen_candidate_spec.json"
    spec_hash = sha256_file(spec_path)

    pre_access = {
        "stage": STAGE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_dir": str(freeze_dir),
        "frozen_candidate_spec_sha256": spec_hash,
        "acceptance_rules": protocol["acceptance_rules"],
        "no_post_test_tuning": True,
        "test_features_read": False,
        "test_targets_read": False,
        "test_predictions_computed": False,
    }
    dump_json(output_dir / "pre_access_manifest.json", pre_access)
    receipt_path = create_access_receipt(freeze_dir, spec_hash, output_dir)

    # The test partition is intentionally loaded only after the irreversible receipt.
    split = load_shared_split(
        paths["split"], expected_dataset_run="run1", expected_curve_count=1000
    )
    if split["split_hash"] != spec["shared_split_hash"]:
        raise ValueError("Runtime test split differs from the frozen F3-F1S split.")
    test_ids = np.asarray(split["test_curve_indices"], dtype=int)
    if test_ids.size != EXPECTED_TEST_CURVES:
        raise ValueError(f"Expected {EXPECTED_TEST_CURVES} test curves, found {test_ids.size}.")
    f1_metadata = load_json(paths["f1_metadata"])
    test = prepare_f1_features(
        load_partition_rows(paths["dataset"], test_ids, 1000), f1_metadata
    )
    if test["y"].size != EXPECTED_TEST_CURVES * EXPECTED_ROWS_PER_CURVE:
        raise ValueError("Unexpected F3-F2 test row count.")

    f1_selection = pd.read_csv(paths["f1_selection"])
    if len(f1_selection) != 1:
        raise ValueError("Frozen formal F1 selection must contain exactly one equation.")
    f1_expression = str(f1_selection.iloc[0]["expression_for_eval"])
    f1_variable_names = make_pysr_variable_names(test["f1_feature_names"])
    f1_prediction = predict_expression(f1_expression, test["X_f1"], f1_variable_names)

    f3b_spec = validate_f3_local_terminal_spec(
        load_json(paths["f3b_spec"]), list(test["base_feature_names"])
    )
    test_f3, test_f3_names = apply_f3_local_terminals(
        test["X_base"], list(test["base_feature_names"]), f3b_spec
    )
    if test_f3_names != spec["f3_output_feature_names"]:
        raise ValueError("Frozen F3 feature order changed on test replay.")
    terminal_index = test_f3_names.index(spec["terminal_name"])
    terminal = test_f3[:, terminal_index].reshape(-1, 1)
    residual = predict_expression(
        spec["canonical_expression_for_eval"], terminal, [spec["terminal_name"]]
    )
    f3_prediction = f1_prediction + residual
    all_finite = bool(
        np.all(np.isfinite(f1_prediction))
        and np.all(np.isfinite(residual))
        and np.all(np.isfinite(f3_prediction))
    )
    if not all_finite:
        raise ValueError("F3-F2 produced non-finite frozen predictions.")

    f1_metrics = regression_metrics(test["y"], f1_prediction)
    f3_metrics = regression_metrics(test["y"], f3_prediction)
    f1_curve_summary, f1_curve_rows = curve_metrics(
        test["y"], f1_prediction, test["source_curve_index"]
    )
    f3_curve_summary, f3_curve_rows = curve_metrics(
        test["y"], f3_prediction, test["source_curve_index"]
    )
    all_regions = (*REPORTING_REGIONS, TARGET_REGION)
    f1_regions = regional_metrics(test["y"], f1_prediction, test["frequency_hz"], all_regions)
    f3_regions = regional_metrics(test["y"], f3_prediction, test["frequency_hz"], all_regions)
    f1_region_map = region_map(f1_regions)
    f3_region_map = region_map(f3_regions)
    guardrails = []
    for bounds in GUARDRAIL_REGIONS:
        relative_change = float(
            f3_region_map[bounds]["rmse"] / f1_region_map[bounds]["rmse"] - 1.0
        )
        guardrails.append(
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "formal_f1_rmse": float(f1_region_map[bounds]["rmse"]),
                "frozen_f3_rmse": float(f3_region_map[bounds]["rmse"]),
                "relative_change": relative_change,
                "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                "passed": relative_change <= GUARDRAIL_MAX_RELATIVE_WORSENING,
            }
        )
    checks = {
        "frozen_inputs_replayed_exactly": True,
        "all_predictions_finite": all_finite,
        "candidate_beats_f1_full_test_rmse": f3_metrics["rmse"] < f1_metrics["rmse"],
        "candidate_improves_curve_mean_rmse": (
            f3_curve_summary["curve_mean_rmse"] < f1_curve_summary["curve_mean_rmse"]
        ),
        "candidate_improves_worst_curve_rmse": (
            f3_curve_summary["worst_curve_rmse"] < f1_curve_summary["worst_curve_rmse"]
        ),
        "candidate_improves_target_region": (
            f3_region_map[TARGET_REGION]["rmse"] < f1_region_map[TARGET_REGION]["rmse"]
        ),
        "all_regional_guardrails_pass": all(row["passed"] for row in guardrails),
    }
    stage_passed = all(checks.values())
    comparison = {
        "stage": STAGE,
        "stage_passed": stage_passed,
        "decision": (
            "accept_frozen_Re_F3_as_formal_Re_model"
            if stage_passed
            else "retain_formal_Re_F1"
        ),
        "formal_model_replacement_decision": stage_passed,
        "frozen_candidate_spec_sha256": spec_hash,
        "formal_f1_test_metrics": f1_metrics,
        "frozen_f3_test_metrics": f3_metrics,
        "formal_f1_test_curve_summary": f1_curve_summary,
        "frozen_f3_test_curve_summary": f3_curve_summary,
        "target_region_comparison": {
            "region_hz": list(TARGET_REGION),
            "formal_f1_rmse": float(f1_region_map[TARGET_REGION]["rmse"]),
            "frozen_f3_rmse": float(f3_region_map[TARGET_REGION]["rmse"]),
        },
        "guardrails": guardrails,
        "checks": checks,
        "reference_only_f1_test_rmse_rounded_from_plan": EXPECTED_F1_TEST_RMSE_ROUNDED,
        "no_post_test_tuning": True,
        "test_rows_accessed": True,
    }
    dump_json(output_dir / "comparison_decision.json", comparison)

    pd.DataFrame(
        [
            {"model": "formal_Re_F1", **f1_metrics, **f1_curve_summary},
            {"model": "frozen_Re_F3", **f3_metrics, **f3_curve_summary},
        ]
    ).to_csv(output_dir / "test_overall_comparison.csv", index=False)
    pd.DataFrame(
        [{"model": "formal_Re_F1", **row} for row in f1_regions]
        + [{"model": "frozen_Re_F3", **row} for row in f3_regions]
    ).to_csv(output_dir / "test_regional_comparison.csv", index=False)
    pd.DataFrame(guardrails).to_csv(output_dir / "test_guardrails.csv", index=False)

    f1_curve_frame = pd.DataFrame(f1_curve_rows).rename(columns={"rmse": "formal_Re_F1_rmse"})
    f3_curve_frame = pd.DataFrame(f3_curve_rows).rename(columns={"rmse": "frozen_Re_F3_rmse"})
    curve_frame = f1_curve_frame.merge(f3_curve_frame, on="curve_id", validate="one_to_one")
    curve_frame.to_csv(output_dir / "test_curve_metrics.csv", index=False)

    f1_frequency = pd.DataFrame(
        frequency_wise_metrics(test["y"], f1_prediction, test["frequency_hz"])
    ).rename(columns={"rmse": "formal_Re_F1_rmse", "mae": "formal_Re_F1_mae", "bias": "formal_Re_F1_bias"})
    f3_frequency = pd.DataFrame(
        frequency_wise_metrics(test["y"], f3_prediction, test["frequency_hz"])
    ).rename(columns={"rmse": "frozen_Re_F3_rmse", "mae": "frozen_Re_F3_mae", "bias": "frozen_Re_F3_bias"})
    frequency_frame = f1_frequency.merge(f3_frequency, on="frequency_hz", validate="one_to_one")
    frequency_frame.to_csv(output_dir / "test_frequency_metrics.csv", index=False)

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        source_curve_index=test["source_curve_index"],
        frequency_hz=test["frequency_hz"],
        teacher=test["y"],
        formal_f1_prediction=f1_prediction,
        frozen_f3_residual=residual,
        frozen_f3_prediction=f3_prediction,
        z_sigma__f3_mid=terminal[:, 0],
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_frequency_metrics(frequency_frame, figures / "test_frequency_rmse.png")
    plot_curve_metrics(curve_frame, figures / "test_curve_rmse_scatter.png")

    report_lines = [
        "# F3-F2 One-Time Test Decision",
        "",
        f"- Decision: `{comparison['decision']}`",
        f"- Formal Re F1 test RMSE: `{f1_metrics['rmse']:.9f}`",
        f"- Frozen Re F3 test RMSE: `{f3_metrics['rmse']:.9f}`",
        f"- Relative RMSE change: `{f3_metrics['rmse'] / f1_metrics['rmse'] - 1.0:.3%}`",
        f"- F1 curve-mean / worst RMSE: `{f1_curve_summary['curve_mean_rmse']:.9f}` / `{f1_curve_summary['worst_curve_rmse']:.9f}`",
        f"- F3 curve-mean / worst RMSE: `{f3_curve_summary['curve_mean_rmse']:.9f}` / `{f3_curve_summary['worst_curve_rmse']:.9f}`",
        f"- Test rows accessed: `True` (one-time receipt `{receipt_path.name}`)",
        "- Post-test formula or threshold changes are forbidden by the frozen protocol.",
    ]
    (output_dir / "decision_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    receipt = load_json(receipt_path)
    receipt.update(
        {
            "status": "test_evaluation_completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "test_rows_loaded": True,
            "test_row_count": int(test["y"].size),
            "decision": comparison["decision"],
        }
    )
    dump_json(receipt_path, receipt)
    manifest = {
        **pre_access,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "shared_split_hash": split["split_hash"],
        "test_curve_count": int(test_ids.size),
        "test_row_count": int(test["y"].size),
        "test_features_read": True,
        "test_targets_read": True,
        "test_predictions_computed": True,
        "access_receipt": str(receipt_path),
        "decision": comparison,
        "artifact_files": sorted(
            [
                *(str(path.relative_to(output_dir)).replace("\\", "/") for path in output_dir.rglob("*") if path.is_file()),
                "stage_manifest.json",
            ]
        ),
    }
    dump_json(output_dir / "stage_manifest.json", manifest)
    print(
        f"F3-F2 complete: decision={comparison['decision']}, "
        f"F1_RMSE={f1_metrics['rmse']:.9f}, F3_RMSE={f3_metrics['rmse']:.9f}"
    )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
