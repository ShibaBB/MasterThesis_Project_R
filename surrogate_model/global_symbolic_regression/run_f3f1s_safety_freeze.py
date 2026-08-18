"""Freeze the domain-safe F3-F1 Hall-of-Fame candidate for F3-F2.

F3-F1S performs no search and cannot load test rows.  It applies the newly
authorized safety-first eligibility rule to the existing formal F3-F1 Hall of
Fame, replays candidate 5 on train/validation, and freezes the later one-time
F3-F2 protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from sympy.calculus.util import continuous_domain


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_ROOT = SCRIPT_DIR.parent
for import_path in (SCRIPT_DIR, SURROGATE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_f3c_residual_pysr_pilot as f3c  # noqa: E402
import run_f3f1_formal_validation as f3f1  # noqa: E402
from run_f3a_residual_diagnosis import (  # noqa: E402
    F3_ARTIFACT_ROOT,
    GUARDRAIL_MAX_RELATIVE_WORSENING,
    GUARDRAIL_REGIONS,
    REPORTING_REGIONS,
    TARGET_REGION,
    curve_metrics,
    regional_metrics,
    regression_metrics,
    sha256_file,
)


STAGE = "F3-F1S"
AUTHORIZED_CANDIDATE_INDEX = 5
DEFAULT_F3F1_DIR = F3_ARTIFACT_ROOT / "F3-F1" / "20260818T205339"
DEFAULT_F1_SELECTION_FILE = (
    SCRIPT_DIR
    / "artifacts"
    / "re"
    / "eval"
    / "20260816_run1_sobol_modulated_best_loss"
    / "selected_candidate.csv"
)
DEFAULT_F3F2_RUNNER = SCRIPT_DIR / "run_f3f2_one_time_test.py"
CANONICAL_EQUATION = "4.196377*z_sigma__f3_mid/(44.352028 + z_sigma__f3_mid**2)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, default=f3c.DEFAULT_DATASET_FILE)
    parser.add_argument("--split-file", type=Path, default=f3c.DEFAULT_SPLIT_FILE)
    parser.add_argument("--f1-training-dir", type=Path, default=f3c.DEFAULT_F1_TRAINING_DIR)
    parser.add_argument("--f1-selection-file", type=Path, default=DEFAULT_F1_SELECTION_FILE)
    parser.add_argument("--f3a-dir", type=Path, default=f3c.DEFAULT_F3A_DIR)
    parser.add_argument("--f3b-dir", type=Path, default=f3c.DEFAULT_F3B_DIR)
    parser.add_argument("--f3c-dir", type=Path, default=f3f1.DEFAULT_F3C_DIR)
    parser.add_argument("--f3d1-dir", type=Path, default=f3f1.DEFAULT_F3D1_DIR)
    parser.add_argument("--f3e1-dir", type=Path, default=f3f1.DEFAULT_F3E1_DIR)
    parser.add_argument("--julia-exe", type=Path, default=f3c.DEFAULT_JULIA_EXE)
    parser.add_argument("--f3f1-dir", type=Path, default=DEFAULT_F3F1_DIR)
    parser.add_argument("--f3f2-runner", type=Path, default=DEFAULT_F3F2_RUNNER)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact session directory; it must remain under artifacts/F3.",
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


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    f3f1_dir = args.f3f1_dir.resolve()
    paths = {
        **f3f1.required_paths(args),
        "f3f1s_runner": Path(__file__).resolve(),
        "f1_selection": args.f1_selection_file.resolve(),
        "f3f1_preflight": f3f1_dir / "preflight_manifest.json",
        "f3f1_decision": f3f1_dir / "comparison_decision.json",
        "f3f1_manifest": f3f1_dir / "stage_manifest.json",
        "f3f1_formula_audit": f3f1_dir / "formula_domain_audit.json",
        "f3f1_equations": f3f1_dir / "runs" / "formal_a" / "equations.csv",
        "f3f1_candidate_metrics": f3f1_dir / "runs" / "formal_a" / "candidate_metrics.csv",
        "f3f1_selected_summary": f3f1_dir / "runs" / "formal_a" / "selected_run_summary.json",
        "f3f2_runner": args.f3f2_runner.resolve(),
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{STAGE} required inputs are missing: {missing}")
    return paths


def symbolic_safety_audit(expression: str) -> dict[str, Any]:
    name = "z_sigma__f3_mid"
    z = sp.Symbol(name, real=True)
    try:
        parsed = sp.sympify(expression, locals={name: z})
        value_at_zero = sp.simplify(parsed.subs(z, 0))
        left = sp.limit(parsed, z, 0, dir="-")
        right = sp.limit(parsed, z, 0, dir="+")
        domain = continuous_domain(parsed, z, sp.S.Reals)
        all_real_continuous = domain == sp.S.Reals
        zero_is_finite = value_at_zero.is_finite is True
        zero_residual = zero_is_finite and sp.simplify(value_at_zero) == 0
        zero_continuous = bool(
            zero_is_finite
            and sp.simplify(left - right) == 0
            and sp.simplify(left - value_at_zero) == 0
        )
        eligible = bool(all_real_continuous and zero_continuous and zero_residual)
        return {
            "parse_status": "ok",
            "continuous_domain": str(domain),
            "all_real_continuous_and_finite": bool(all_real_continuous),
            "value_at_zero": str(value_at_zero),
            "left_limit_at_zero": str(left),
            "right_limit_at_zero": str(right),
            "continuous_at_zero": zero_continuous,
            "zero_terminal_gives_exact_zero_residual": bool(zero_residual),
            "safety_eligible": eligible,
        }
    except Exception as error:
        return {
            "parse_status": f"failed: {error}",
            "continuous_domain": "unknown",
            "all_real_continuous_and_finite": False,
            "value_at_zero": "unknown",
            "left_limit_at_zero": "unknown",
            "right_limit_at_zero": "unknown",
            "continuous_at_zero": False,
            "zero_terminal_gives_exact_zero_residual": False,
            "safety_eligible": False,
        }


def build_selection_table(
    candidate_metrics: pd.DataFrame, view: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, candidate in candidate_metrics.iterrows():
        expression = str(candidate["expression_for_eval"])
        audit = symbolic_safety_audit(expression)
        try:
            train_residual = f3c.predict_expression(
                expression, view["X_train"], view["variable_names"]
            )
            validation_residual = f3c.predict_expression(
                expression, view["X_validation"], view["variable_names"]
            )
            observed_eval_ok = bool(
                str(candidate["eval_status"]) == "ok"
                and math.isfinite(float(candidate["rmse"]))
                and np.all(np.isfinite(train_residual))
                and np.all(np.isfinite(validation_residual))
            )
        except Exception:
            observed_eval_ok = False
        eligible = observed_eval_ok and audit["safety_eligible"]
        rows.append(
            {
                "candidate_index": int(candidate["candidate_index"]),
                "candidate_source": str(candidate["candidate_source"]),
                "complexity": int(candidate["complexity"]),
                "validation_rmse": float(candidate["rmse"]),
                "equation": str(candidate["equation"]),
                "expression_for_eval": expression,
                "observed_train_validation_eval_ok": observed_eval_ok,
                **audit,
                "eligible_for_safety_first_selection": bool(eligible),
            }
        )
    return pd.DataFrame(rows)


def metric_comparison(
    evaluated: dict[str, Any], context: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    f1_metrics = regression_metrics(context["validation"]["y"], context["validation_f1"])
    f1_curve_summary, f1_curve_rows = curve_metrics(
        context["validation"]["y"],
        context["validation_f1"],
        context["validation"]["source_curve_index"],
    )
    f1_regions = regional_metrics(
        context["validation"]["y"],
        context["validation_f1"],
        context["validation"]["frequency_hz"],
        (*REPORTING_REGIONS, TARGET_REGION),
    )
    f1_lookup = f3f1.region_map(f1_regions)
    f3_lookup = f3f1.region_map(evaluated["validation_regional_metrics"])
    guardrails = []
    for bounds in GUARDRAIL_REGIONS:
        relative_change = float(f3_lookup[bounds]["rmse"] / f1_lookup[bounds]["rmse"] - 1.0)
        guardrails.append(
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "formal_f1_control_rmse": float(f1_lookup[bounds]["rmse"]),
                "frozen_safe_candidate_rmse": float(f3_lookup[bounds]["rmse"]),
                "relative_change": relative_change,
                "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                "passed": relative_change <= GUARDRAIL_MAX_RELATIVE_WORSENING,
            }
        )
    checks = {
        "beats_formal_f1_full_validation_rmse": evaluated["validation_metrics"]["rmse"] < f1_metrics["rmse"],
        "improves_curve_mean_rmse": evaluated["validation_curve_summary"]["curve_mean_rmse"] < f1_curve_summary["curve_mean_rmse"],
        "improves_worst_curve_rmse": evaluated["validation_curve_summary"]["worst_curve_rmse"] < f1_curve_summary["worst_curve_rmse"],
        "improves_target_region": f3_lookup[TARGET_REGION]["rmse"] < f1_lookup[TARGET_REGION]["rmse"],
        "all_formal_f1_regional_guardrails_pass": all(row["passed"] for row in guardrails),
    }
    reference = {
        "validation_metrics": f1_metrics,
        "validation_curve_summary": f1_curve_summary,
        "validation_regional_metrics": f1_regions,
        "validation_curve_rows": f1_curve_rows,
    }
    return reference, guardrails, checks


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    paths = source_paths(args)
    f3f1_decision = load_json(paths["f3f1_decision"])
    f3f1_manifest = load_json(paths["f3f1_manifest"])
    formula_audit = load_json(paths["f3f1_formula_audit"])
    if f3f1_decision.get("predeclared_validation_gate_passed") is not True:
        raise ValueError("The source formal F3-F1 validation gate did not pass.")
    if f3f1_decision.get("f3f2_readiness") is not False:
        raise ValueError("F3-F1S expected the source F3-F1 formula-domain hold.")
    if formula_audit.get("best_continuous_hall_of_fame_candidate_diagnostic_only", {}).get("candidate_index") != AUTHORIZED_CANDIDATE_INDEX:
        raise ValueError("F3-F1 did not identify candidate 5 as its best continuous diagnostic.")
    if any(
        f3f1_manifest.get(key) is not False
        for key in ("test_features_read", "test_targets_read", "test_predictions_computed")
    ):
        raise ValueError("The source F3-F1 manifest does not preserve test isolation.")

    preflight = {
        "schema_version": 1,
        "stage": STAGE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "validation-only safety freeze; no training and no search",
        "explicit_user_authorization": "freeze safe candidate 5 and prepare F3-F2",
        "authorized_candidate_index": AUTHORIZED_CANDIDATE_INDEX,
        "selection_policy_frozen_before_replay": {
            "eligibility_first": [
                "expression evaluates successfully on observed train/validation rows",
                "expression is finite and continuous over the entire real terminal domain",
                "residual is continuous at z_sigma__f3_mid = 0",
                "residual equals exactly zero at z_sigma__f3_mid = 0",
            ],
            "ranking_among_eligible": "minimum complete-validation RMSE, then minimum complexity, then candidate index",
            "purpose": "deployment-safety resolution before any test access",
        },
        "validation_acceptance_rules": {
            "full_validation": "candidate RMSE must be strictly below formal Re F1",
            "curve_behavior": "curve-mean and worst-curve RMSE must both be strictly below formal Re F1",
            "target_region": "1300-2000 Hz RMSE must be strictly below formal Re F1",
            "guardrails": "no declared regional RMSE may worsen more than 10% versus formal Re F1",
        },
        "source_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "test_access_contract": "test features, targets, and predictions remain inaccessible",
        "test_features_read": False,
        "test_targets_read": False,
        "test_predictions_computed": False,
    }
    dump_json(output_dir / "preflight_manifest.json", preflight)

    # load_context is intentionally the same train/validation-only loader used by F3-F1.
    context = f3f1.load_context(args)
    view = f3f1.branch_view(context)
    candidate_metrics = pd.read_csv(paths["f3f1_candidate_metrics"])
    selection_table = build_selection_table(candidate_metrics, view)
    eligible = selection_table[selection_table["eligible_for_safety_first_selection"]].copy()
    if eligible.empty:
        raise RuntimeError("No Hall-of-Fame candidate passed the frozen safety eligibility rules.")
    selected_audit = eligible.sort_values(
        ["validation_rmse", "complexity", "candidate_index"], ascending=True
    ).iloc[0]
    selected_index = int(selected_audit["candidate_index"])
    if selected_index != AUTHORIZED_CANDIDATE_INDEX:
        raise ValueError(
            f"Safety-first policy selected candidate {selected_index}, not explicitly authorized candidate 5."
        )
    selection_table["selected_by_safety_first_policy"] = selection_table["candidate_index"] == selected_index
    selection_table.to_csv(output_dir / "candidate_safety_selection.csv", index=False)

    source_row = candidate_metrics[candidate_metrics["candidate_index"] == selected_index]
    if len(source_row) != 1:
        raise ValueError("Candidate 5 is not unique in the source candidate table.")
    source_row = source_row.iloc[0]
    source_expression = str(source_row["expression_for_eval"])
    z = sp.Symbol("z_sigma__f3_mid", real=True)
    source_symbolic = sp.sympify(source_expression, locals={"z_sigma__f3_mid": z})
    canonical_symbolic = sp.sympify(CANONICAL_EQUATION, locals={"z_sigma__f3_mid": z})
    if sp.simplify(source_symbolic - canonical_symbolic) != 0:
        raise ValueError("Canonical candidate-5 formula is not symbolically identical to the source equation.")

    source_validation_residual = f3c.predict_expression(source_expression, view["X_validation"], view["variable_names"])
    canonical_validation_residual = f3c.predict_expression(CANONICAL_EQUATION, view["X_validation"], view["variable_names"])
    canonical_train_residual = f3c.predict_expression(CANONICAL_EQUATION, view["X_train"], view["variable_names"])
    equivalence_max_abs = float(np.max(np.abs(source_validation_residual - canonical_validation_residual)))
    if equivalence_max_abs > 1.0e-14:
        raise ValueError("Canonical candidate-5 replay differs from the source formula.")
    validation_prediction = context["validation_f1"] + canonical_validation_residual
    selected_for_eval = source_row.copy()
    selected_for_eval["expression_for_eval"] = CANONICAL_EQUATION
    evaluated = f3c.evaluate_selected(selected_for_eval, view, context, validation_prediction)

    dense_grid = np.unique(
        np.concatenate(
            [
                np.linspace(-10.0, 10.0, 20001),
                -np.logspace(-12, 6, 4000),
                np.asarray([0.0]),
                np.logspace(-12, 6, 4000),
            ]
        )
    )
    dense_residual = f3c.predict_expression(CANONICAL_EQUATION, dense_grid.reshape(-1, 1), ["z_sigma__f3_mid"])
    denominator_minimum_proof = 44.352028
    domain_audit = {
        "candidate_index": selected_index,
        "source_equation": str(source_row["equation"]),
        "source_expression_for_eval": source_expression,
        "canonical_equation": "delta_Re = 4.196377*z_sigma__f3_mid/(44.352028 + z_sigma__f3_mid^2)",
        "canonical_expression_for_eval": CANONICAL_EQUATION,
        "symbolically_equivalent_to_source": True,
        "analytic_proof": {
            "denominator": "44.352028 + z_sigma__f3_mid^2",
            "denominator_strict_lower_bound_for_all_real_z": denominator_minimum_proof,
            "has_real_singularity": False,
            "continuous_and_finite_domain": "all real z_sigma__f3_mid",
            "residual_at_zero": 0.0,
            "limit_at_positive_and_negative_infinity": 0.0,
        },
        "dense_numeric_audit": {
            "grid_min": float(dense_grid.min()),
            "grid_max": float(dense_grid.max()),
            "grid_size": int(dense_grid.size),
            "all_finite": bool(np.all(np.isfinite(dense_residual))),
            "value_at_zero": float(dense_residual[np.flatnonzero(dense_grid == 0.0)[0]]),
            "maximum_absolute_residual": float(np.max(np.abs(dense_residual))),
        },
        "source_to_canonical_validation_max_abs_difference": equivalence_max_abs,
        "passed": bool(
            denominator_minimum_proof > 0.0
            and np.all(np.isfinite(dense_residual))
            and dense_residual[np.flatnonzero(dense_grid == 0.0)[0]] == 0.0
        ),
    }
    dump_json(output_dir / "formula_domain_safety_audit.json", domain_audit)

    f1_reference, guardrails, numerical_checks = metric_comparison(evaluated, context)
    replay_spec_base = {
        "schema_version": 1,
        "stage_frozen": STAGE,
        "source_formal_stage": "F3-F1",
        "source_formal_seed": 42,
        "source_candidate_index": selected_index,
        "source_candidate_complexity": int(source_row["complexity"]),
        "source_equation": str(source_row["equation"]),
        "source_expression_for_eval": source_expression,
        "canonical_equation": domain_audit["canonical_equation"],
        "canonical_expression_for_eval": CANONICAL_EQUATION,
        "terminal_name": "z_sigma__f3_mid",
        "terminal_coefficient": 4.196377,
        "positive_denominator_constant": 44.352028,
        "model_definition": "Re_F3 = frozen_formal_Re_F1 + delta_Re",
        "f1_fallback_at_terminal_zero": True,
        "input_feature_names": ["z_sigma__f3_mid"],
        "f3_output_feature_names": context["f3_feature_names"],
        "f3b_terminal_spec": context["f3b_spec"],
        "shared_split_hash": context["split"]["split_hash"],
        "selection_policy": preflight["selection_policy_frozen_before_replay"],
        "validation_metrics": evaluated["validation_metrics"],
        "validation_curve_summary": evaluated["validation_curve_summary"],
        "validation_regional_metrics": evaluated["validation_regional_metrics"],
        "train_metrics": evaluated["train_metrics"],
        "train_curve_summary": evaluated["train_curve_summary"],
        "formula_domain_safety_passed": domain_audit["passed"],
        "test_rows_accessed": False,
    }
    dump_json(output_dir / "frozen_candidate_spec.json", replay_spec_base)
    frozen_spec = load_json(output_dir / "frozen_candidate_spec.json")
    replay_train_residual = f3c.predict_expression(
        frozen_spec["canonical_expression_for_eval"], view["X_train"], [frozen_spec["terminal_name"]]
    )
    replay_validation_residual = f3c.predict_expression(
        frozen_spec["canonical_expression_for_eval"], view["X_validation"], [frozen_spec["terminal_name"]]
    )
    replay_checks = {
        "train_residual_exact_reload_replay": bool(np.array_equal(canonical_train_residual, replay_train_residual)),
        "validation_residual_exact_reload_replay": bool(np.array_equal(canonical_validation_residual, replay_validation_residual)),
        "train_prediction_all_finite": bool(np.all(np.isfinite(evaluated["train_prediction"]))),
        "validation_prediction_all_finite": bool(np.all(np.isfinite(evaluated["validation_prediction"]))),
    }
    numerical_checks.update(replay_checks)
    numerical_checks["candidate_5_selected_by_frozen_safety_first_policy"] = selected_index == AUTHORIZED_CANDIDATE_INDEX
    numerical_checks["formula_domain_safety_passed"] = domain_audit["passed"]
    stage_passed = all(numerical_checks.values())

    pd.DataFrame(
        [
            {"model": "formal_Re_F1", **f1_reference["validation_metrics"], **f1_reference["validation_curve_summary"]},
            {"model": "frozen_safe_Re_F3_candidate_5", **evaluated["validation_metrics"], **evaluated["validation_curve_summary"]},
        ]
    ).to_csv(output_dir / "validation_overall_comparison.csv", index=False)
    pd.DataFrame(
        [{"model": "formal_Re_F1", **row} for row in f1_reference["validation_regional_metrics"]]
        + [{"model": "frozen_safe_Re_F3_candidate_5", **row} for row in evaluated["validation_regional_metrics"]]
    ).to_csv(output_dir / "validation_regional_comparison.csv", index=False)
    pd.DataFrame(guardrails).to_csv(output_dir / "validation_guardrails.csv", index=False)
    pd.DataFrame(evaluated["validation_curve_rows"]).to_csv(output_dir / "validation_curve_metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "frozen_predictions_train_validation.npz",
        train_curve_ids=context["train"]["source_curve_index"],
        train_frequency_hz=context["train"]["frequency_hz"],
        train_f1_prediction=context["train_f1"],
        train_residual=canonical_train_residual,
        train_f3_prediction=evaluated["train_prediction"],
        validation_curve_ids=context["validation"]["source_curve_index"],
        validation_frequency_hz=context["validation"]["frequency_hz"],
        validation_f1_prediction=context["validation_f1"],
        validation_residual=canonical_validation_residual,
        validation_f3_prediction=evaluated["validation_prediction"],
    )

    decision = {
        "stage": STAGE,
        "stage_passed": stage_passed,
        "decision": (
            "freeze_domain_safe_formal_Re_F3_candidate_for_separately_authorized_F3_F2"
            if stage_passed
            else "retain_formal_Re_F1_and_do_not_open_test"
        ),
        "selected_candidate_index": selected_index,
        "selected_complexity": int(source_row["complexity"]),
        "frozen_candidate": replay_spec_base,
        "formal_f1_validation_reference": {
            key: value for key, value in f1_reference.items() if key != "validation_curve_rows"
        },
        "formal_f1_guardrails": guardrails,
        "checks": numerical_checks,
        "validation_rmse_relative_improvement": float(
            1.0 - evaluated["validation_metrics"]["rmse"] / f1_reference["validation_metrics"]["rmse"]
        ),
        "formal_model_replacement_decision": False,
        "f3f2_readiness": stage_passed,
        "test_evaluation_authorized": False,
        "explicit_f3f2_command_still_required": True,
        "test_rows_accessed": False,
        "next_command_if_user_accepts": "执行 F3-F2" if stage_passed else None,
    }
    dump_json(output_dir / "comparison_decision.json", decision)

    frozen_input_names = ("dataset", "split", "f1_metadata", "f1_selection", "f3b_spec")
    protocol = {
        "schema_version": 1,
        "stage_prepared_for": "F3-F2",
        "prepared_by": STAGE,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "f3f1s_stage_passed": stage_passed,
        "explicit_user_command_required": True,
        "one_time_access_enforcement": "exclusive receipt keyed by frozen candidate spec SHA256 is written before test row load",
        "frozen_candidate_spec_path": str(output_dir / "frozen_candidate_spec.json"),
        "frozen_candidate_spec_sha256": sha256_file(output_dir / "frozen_candidate_spec.json"),
        "f3f2_runner_path": str(paths["f3f2_runner"]),
        "f3f2_runner_sha256": sha256_file(paths["f3f2_runner"]),
        "frozen_input_files": {
            name: {"path": str(paths[name]), "sha256": sha256_file(paths[name])}
            for name in frozen_input_names
        },
        "expected_partition": {
            "name": "test",
            "curve_count": 150,
            "rows_per_curve": 128,
            "row_count": 19200,
            "shared_split_hash": context["split"]["split_hash"],
        },
        "acceptance_rules": {
            "full_test": "frozen Re F3 RMSE must be strictly below formal Re F1 on the same test rows",
            "curve_behavior": "curve-mean and worst-curve RMSE must both be strictly below formal Re F1",
            "target_region": "1300-2000 Hz RMSE must be strictly below formal Re F1",
            "guardrails": "none of 700-1000, 1300-1650, or 1650-2000 Hz RMSE may worsen more than 10% versus formal Re F1",
            "numerical_safety": "F1, residual, and final predictions must all be finite",
            "all_checks_required": True,
            "failure_action": "retain formal Re F1",
        },
        "reporting_contract": [
            "raw full-range RMSE/MAE/max-absolute-error/R2/bias/prediction range",
            "curve mean/median/p95/worst RMSE and per-curve metrics",
            "frequency-wise RMSE/MAE/bias",
            "all reporting regions, target region, and guardrails",
            "frozen predictions and diagnostic figures",
        ],
        "no_post_test_formula_or_threshold_changes": True,
        "test_rows_accessed_during_protocol_preparation": False,
    }
    dump_json(output_dir / "f3f2_test_protocol.json", protocol)

    report = [
        "# F3-F1S Safety Freeze Decision",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Frozen Hall-of-Fame candidate: `{selected_index}` (complexity `{int(source_row['complexity'])}`)",
        f"- Canonical residual: `{domain_audit['canonical_equation']}`",
        f"- Candidate validation RMSE: `{evaluated['validation_metrics']['rmse']:.9f}`",
        f"- Formal Re F1 validation RMSE: `{f1_reference['validation_metrics']['rmse']:.9f}`",
        f"- Relative validation RMSE improvement: `{decision['validation_rmse_relative_improvement']:.3%}`",
        f"- Formula domain safety: `{'PASS' if domain_audit['passed'] else 'FAIL'}`",
        f"- F3-F2 readiness: `{decision['f3f2_readiness']}`",
        "- Test rows accessed: `False`",
        "- F3-F2 is prepared but still requires the separate explicit command `执行 F3-F2`.",
    ]
    (output_dir / "decision_report.md").write_text("\n".join(report), encoding="utf-8")

    manifest = {
        **preflight,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "shared_split_hash": context["split"]["split_hash"],
        "train_curve_count": 700,
        "validation_curve_count": 150,
        "test_curve_count_declared_but_not_loaded": context["test_curve_count"],
        "train_row_count": int(context["train"]["y"].size),
        "validation_row_count": int(context["validation"]["y"].size),
        "test_features_read": False,
        "test_targets_read": False,
        "test_predictions_computed": False,
        "candidate_selection": selection_table.to_dict(orient="records"),
        "formula_domain_safety_audit": domain_audit,
        "decision": decision,
        "f3f2_protocol": protocol,
        "artifact_files": sorted(
            [
                *(str(path.relative_to(output_dir)).replace("\\", "/") for path in output_dir.rglob("*") if path.is_file()),
                "stage_manifest.json",
            ]
        ),
    }
    dump_json(output_dir / "stage_manifest.json", manifest)
    print(
        f"F3-F1S complete: stage_passed={stage_passed}, candidate={selected_index}, "
        f"validation_RMSE={evaluated['validation_metrics']['rmse']:.9f}, test_rows_accessed=False"
    )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
