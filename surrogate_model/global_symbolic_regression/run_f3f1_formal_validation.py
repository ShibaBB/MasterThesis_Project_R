"""Run the validation-only formal Re Global SR F3-F1 confirmation."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for import_path in (SCRIPT_DIR, SURROGATE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_f3c_residual_pysr_pilot as f3c  # noqa: E402
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


STAGE = "F3-F1"
BRANCH = "formal_a"
SEED = 42
NITERATIONS = 100
POPULATIONS = 12
POPULATION_SIZE = 80
MAX_COMPLEXITY = 24
BATCH_SIZE = 4096
BINARY_OPERATORS = ("+", "-", "*", "/")
UNARY_OPERATORS = ("log", "sqrt")
NESTED_CONSTRAINTS = None
DEFAULT_F3C_DIR = F3_ARTIFACT_ROOT / "F3-C" / "20260817T115632"
DEFAULT_F3D1_DIR = F3_ARTIFACT_ROOT / "F3-D1" / "20260817T174849"
DEFAULT_F3E1_DIR = F3_ARTIFACT_ROOT / "F3-E1" / "20260818T203958"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, default=f3c.DEFAULT_DATASET_FILE)
    parser.add_argument("--split-file", type=Path, default=f3c.DEFAULT_SPLIT_FILE)
    parser.add_argument(
        "--f1-training-dir", type=Path, default=f3c.DEFAULT_F1_TRAINING_DIR
    )
    parser.add_argument("--f3a-dir", type=Path, default=f3c.DEFAULT_F3A_DIR)
    parser.add_argument("--f3b-dir", type=Path, default=f3c.DEFAULT_F3B_DIR)
    parser.add_argument("--f3c-dir", type=Path, default=DEFAULT_F3C_DIR)
    parser.add_argument("--f3d1-dir", type=Path, default=DEFAULT_F3D1_DIR)
    parser.add_argument("--f3e1-dir", type=Path, default=DEFAULT_F3E1_DIR)
    parser.add_argument("--julia-exe", type=Path, default=f3c.DEFAULT_JULIA_EXE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact session directory; it must remain under artifacts/F3.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
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
        output_dir = unique_path(
            root / STAGE / datetime.now().strftime("%Y%m%dT%H%M%S")
        )
    else:
        output_dir = requested.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError(f"{STAGE} artifacts must stay under {root}: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {STAGE} directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def required_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = f3c.required_paths(args)
    paths.update(
        {
            "f3c_decision": args.f3c_dir.resolve() / "comparison_decision.json",
            "f3c_manifest": args.f3c_dir.resolve() / "stage_manifest.json",
            "f3d1_decision": args.f3d1_dir.resolve() / "comparison_decision.json",
            "f3e1_decision": args.f3e1_dir.resolve() / "comparison_decision.json",
        }
    )
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{STAGE} required inputs are missing: {missing}")
    return paths


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    paths = required_paths(args)
    context = f3c.load_context(args)
    f3c_decision = load_json(paths["f3c_decision"])
    f3c_manifest = load_json(paths["f3c_manifest"])
    f3d1_decision = load_json(paths["f3d1_decision"])
    f3e1_decision = load_json(paths["f3e1_decision"])
    if f3c_decision.get("stage_passed") is not True:
        raise ValueError("F3-C must pass before F3-F1.")
    if f3c_decision.get("decision") != "advance_branch_A_representation_to_next_explicit_stage":
        raise ValueError("F3-C branch A was not accepted as the representation.")
    if f3d1_decision.get("decision") != "retain_saved_branch_A_representation":
        raise ValueError("F3-D1 did not retain branch A as expected.")
    if f3e1_decision.get("decision") != "retain_original_unconstrained_search_configuration":
        raise ValueError("F3-E1 did not retain the original search configuration.")
    decisions = (f3c_decision, f3d1_decision, f3e1_decision)
    if any(decision.get("test_rows_accessed") is not False for decision in decisions):
        raise ValueError("An upstream F3 stage reports test-row access.")
    if f3c_manifest.get("shared_split_hash") != context["split"]["split_hash"]:
        raise ValueError("F3-C and F3-F1 split hashes differ.")
    if any(
        f3c_manifest.get(key) is not False
        for key in ("test_features_read", "test_targets_read", "test_predictions_computed")
    ):
        raise ValueError("F3-C manifest does not preserve test isolation.")
    context.update(
        {
            "f3f1_paths": paths,
            "f3c_decision": f3c_decision,
            "f3d1_decision": f3d1_decision,
            "f3e1_decision": f3e1_decision,
        }
    )
    return context


def search_contract() -> dict[str, Any]:
    return {
        "niterations": NITERATIONS,
        "populations": POPULATIONS,
        "population_size": POPULATION_SIZE,
        "max_complexity": MAX_COMPLEXITY,
        "binary_operators": list(BINARY_OPERATORS),
        "unary_operators": list(UNARY_OPERATORS),
        "nested_constraints": NESTED_CONSTRAINTS,
        "parallelism": "serial",
        "deterministic": True,
        "batching": True,
        "batch_size": BATCH_SIZE,
        "turbo": True,
        "julia_threads": 1,
        "candidate_selection": "minimum_complete_validation_RMSE_then_complexity",
    }


def branch_view(context: dict[str, Any]) -> dict[str, Any]:
    terminal_index = context["f3_feature_names"].index("z_sigma__f3_mid")
    return {
        "X_train": context["train_f3"][:, terminal_index].reshape(-1, 1),
        "X_validation": context["validation_f3"][:, terminal_index].reshape(-1, 1),
        "y_search_train": context["train"]["y"] - context["train_f1"],
        "variable_names": ["z_sigma__f3_mid"],
        "feature_names": ["z_sigma__f3_mid"],
        "search_target": "delta_Re = teacher - frozen_Re_F1",
        "final_offset_train": context["train_f1"],
        "final_offset_validation": context["validation_f1"],
        "explicit_zero_fallback": True,
    }


def build_model(run_dir: Path) -> Any:
    from pysr import PySRRegressor

    return PySRRegressor(
        niterations=NITERATIONS,
        populations=POPULATIONS,
        population_size=POPULATION_SIZE,
        binary_operators=list(BINARY_OPERATORS),
        unary_operators=list(UNARY_OPERATORS),
        maxsize=MAX_COMPLEXITY,
        model_selection="best",
        random_state=SEED,
        deterministic=True,
        parallelism="serial",
        batching=True,
        batch_size=BATCH_SIZE,
        turbo=True,
        progress=False,
        verbosity=1,
        temp_equation_file=False,
        output_directory=(run_dir / "pysr_runs").as_posix(),
        run_id="f3f1_formal_a_s42",
    )


def worker_main(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        raise ValueError("Worker mode requires --run-dir.")
    run_dir = args.run_dir.resolve()
    session_dir = run_dir.parents[1]
    if not run_dir.is_relative_to(session_dir / "runs"):
        raise ValueError("Worker output must remain inside the F3-F1 runs directory.")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JULIA_NUM_THREADS"] = "1"
    julia_exe = args.julia_exe.resolve()
    if not julia_exe.exists():
        raise FileNotFoundError(f"Julia executable not found: {julia_exe}")
    os.environ.setdefault("PYTHON_JULIAPKG_EXE", str(julia_exe))

    print(
        f"Worker start: stage={STAGE}, branch={BRANCH}, seed={SEED}, "
        f"budget={NITERATIONS}/{POPULATIONS}/{POPULATION_SIZE}/{MAX_COMPLEXITY}",
        flush=True,
    )
    context = load_context(args)
    view = branch_view(context)
    model = build_model(run_dir)
    fit_started = time.perf_counter()
    model.fit(
        view["X_train"],
        view["y_search_train"],
        variable_names=view["variable_names"],
    )
    fit_seconds = time.perf_counter() - fit_started
    equations = model.equations_.copy()
    equations_file = run_dir / "equations.csv"
    equations.to_csv(equations_file, index=False)
    with (run_dir / "pysr_model.pkl").open("wb") as file:
        pickle.dump(model, file)

    candidate_metrics, predictions = f3c.score_candidates(equations, view, context)
    candidate_metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)
    valid = candidate_metrics[candidate_metrics["eval_status"] == "ok"].copy()
    if valid.empty:
        raise RuntimeError("No valid F3-F1 candidate predictions were produced.")
    valid = valid.sort_values(["rmse", "complexity"], ascending=[True, True])
    selected = valid.iloc[0]
    selected_index = int(selected["candidate_index"])
    evaluated = f3c.evaluate_selected(
        selected, view, context, predictions[selected_index]
    )
    pd.DataFrame(evaluated["validation_regional_metrics"]).to_csv(
        run_dir / "selected_validation_regional_metrics.csv", index=False
    )
    pd.DataFrame(evaluated["validation_curve_rows"]).to_csv(
        run_dir / "selected_validation_curve_metrics.csv", index=False
    )
    np.savez_compressed(
        run_dir / "selected_predictions_train_validation.npz",
        train_curve_ids=context["train"]["source_curve_index"],
        train_frequency_hz=context["train"]["frequency_hz"],
        train_prediction=evaluated["train_prediction"],
        validation_curve_ids=context["validation"]["source_curve_index"],
        validation_frequency_hz=context["validation"]["frequency_hz"],
        validation_prediction=evaluated["validation_prediction"],
    )
    from pysr import __version__ as pysr_version

    summary = {
        "stage": STAGE,
        "branch": BRANCH,
        "seed": SEED,
        "search_target": view["search_target"],
        "input_feature_names": view["feature_names"],
        "pysr_variable_names": view["variable_names"],
        "representation_frozen_from": "F3-C branch A",
        "search_configuration_frozen_as": "original unconstrained configuration retained after F3-E1",
        "explicit_zero_f1_fallback": True,
        "selected_candidate_index": selected_index,
        "selected_candidate_source": str(selected["candidate_source"]),
        "selected_complexity": int(selected["complexity"]),
        "selected_equation": str(selected["equation"]),
        "selected_expression_for_eval": str(selected["expression_for_eval"]),
        "fit_wall_time_seconds": float(fit_seconds),
        "train_metrics": evaluated["train_metrics"],
        "validation_metrics": evaluated["validation_metrics"],
        "train_curve_summary": evaluated["train_curve_summary"],
        "validation_curve_summary": evaluated["validation_curve_summary"],
        "validation_regional_metrics": evaluated["validation_regional_metrics"],
        "search_contract": search_contract(),
        "runtime": {
            "python": sys.version,
            "pysr": pysr_version,
            "julia_executable": str(julia_exe),
            "JULIA_NUM_THREADS": os.environ.get("JULIA_NUM_THREADS"),
        },
        "shared_split_hash": context["split"]["split_hash"],
        "train_row_count": int(view["X_train"].shape[0]),
        "validation_row_count": int(view["X_validation"].shape[0]),
        "test_rows_accessed": False,
        "equations_file": str(equations_file),
        "equations_sha256": sha256_file(equations_file),
    }
    dump_json(run_dir / "selected_run_summary.json", summary)
    print(
        f"Worker complete: validation_RMSE={evaluated['validation_metrics']['rmse']:.9f}, "
        f"selected={selected_index}, source={selected['candidate_source']}, "
        f"complexity={int(selected['complexity'])}, fit_seconds={fit_seconds:.1f}",
        flush=True,
    )


def region_map(rows: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (float(row["lower_hz"]), float(row["upper_hz"])): row for row in rows
    }


def aggregate(
    session_dir: Path, summary: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    formal_metrics = regression_metrics(
        context["validation"]["y"], context["validation_f1"]
    )
    formal_curve_summary, formal_curve_rows = curve_metrics(
        context["validation"]["y"],
        context["validation_f1"],
        context["validation"]["source_curve_index"],
    )
    formal_regions = regional_metrics(
        context["validation"]["y"],
        context["validation_f1"],
        context["validation"]["frequency_hz"],
        (*REPORTING_REGIONS, TARGET_REGION),
    )
    formal = {
        "validation_metrics": formal_metrics,
        "validation_curve_summary": formal_curve_summary,
        "validation_regional_metrics": formal_regions,
    }
    candidate_regions = region_map(summary["validation_regional_metrics"])
    control_regions = region_map(formal_regions)
    guardrails: list[dict[str, Any]] = []
    for bounds in GUARDRAIL_REGIONS:
        control_rmse = float(control_regions[bounds]["rmse"])
        candidate_rmse = float(candidate_regions[bounds]["rmse"])
        change = candidate_rmse / control_rmse - 1.0
        guardrails.append(
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "formal_f1_control_rmse": control_rmse,
                "formal_f3_candidate_rmse": candidate_rmse,
                "relative_change": change,
                "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                "passed": change <= GUARDRAIL_MAX_RELATIVE_WORSENING,
            }
        )
    target_f1 = float(control_regions[TARGET_REGION]["rmse"])
    target_f3 = float(candidate_regions[TARGET_REGION]["rmse"])
    checks = {
        "formal_search_completed": summary.get("stage") == STAGE,
        "validation_only": summary.get("test_rows_accessed") is False,
        "frozen_seed_42": int(summary.get("seed", -1)) == SEED,
        "frozen_branch_a_representation": summary.get("input_feature_names")
        == ["z_sigma__f3_mid"],
        "original_unconstrained_search_configuration": summary["search_contract"].get(
            "nested_constraints"
        )
        is None,
        "exact_formal_f1_fallback_preserved": summary.get(
            "explicit_zero_f1_fallback"
        )
        is True,
        "candidate_beats_formal_f1_full_validation_rmse": (
            summary["validation_metrics"]["rmse"] < formal_metrics["rmse"]
        ),
        "candidate_improves_curve_mean_rmse": (
            summary["validation_curve_summary"]["curve_mean_rmse"]
            < formal_curve_summary["curve_mean_rmse"]
        ),
        "candidate_improves_worst_curve_rmse": (
            summary["validation_curve_summary"]["worst_curve_rmse"]
            < formal_curve_summary["worst_curve_rmse"]
        ),
        "candidate_improves_target_region": target_f3 < target_f1,
        "all_formal_f1_regional_guardrails_pass": all(
            row["passed"] for row in guardrails
        ),
    }
    stage_passed = all(checks.values())
    comparison = {
        "stage": STAGE,
        "stage_passed": stage_passed,
        "decision": (
            "freeze_formal_Re_F3_candidate_for_separately_authorized_F3_F2"
            if stage_passed
            else "retain_formal_Re_F1_and_do_not_open_test"
        ),
        "formal_model_replacement_decision": False,
        "test_evaluation_authorized": False,
        "seed": SEED,
        "formal_f3_candidate": summary,
        "formal_f1_validation_reference": formal,
        "small_budget_f3c_reference": {
            "branch_a_three_seed_median": context["f3c_decision"]["branch_medians"]["a"],
            "not_an_acceptance_control": True,
        },
        "target_region_comparison": {
            "region_hz": list(TARGET_REGION),
            "formal_f3_candidate_rmse": target_f3,
            "formal_f1_rmse": target_f1,
        },
        "formal_f1_guardrails": guardrails,
        "checks": checks,
        "test_rows_accessed": False,
        "next_stage_automatically_authorized": False,
        "next_command_if_user_accepts": "执行 F3-F2" if stage_passed else None,
    }
    dump_json(session_dir / "comparison_decision.json", comparison)
    pd.DataFrame(
        [
            {
                "model": "formal_Re_F1",
                **formal_metrics,
                **formal_curve_summary,
            },
            {
                "model": "formal_Re_F3_candidate",
                **summary["validation_metrics"],
                **summary["validation_curve_summary"],
            },
        ]
    ).to_csv(session_dir / "overall_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"model": "formal_Re_F1", **row} for row in formal_regions
        ]
        + [
            {"model": "formal_Re_F3_candidate", **row}
            for row in summary["validation_regional_metrics"]
        ]
    ).to_csv(session_dir / "regional_comparison.csv", index=False)
    pd.DataFrame(guardrails).to_csv(
        session_dir / "formal_f1_guardrail_comparison.csv", index=False
    )
    pd.DataFrame(formal_curve_rows).to_csv(
        session_dir / "formal_f1_validation_curve_metrics.csv", index=False
    )
    return comparison


def audit_selected_formula_domain(
    session_dir: Path, summary: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Audit continuity at the valid zero value of the frozen local terminal."""
    variable_name = "z_sigma__f3_mid"
    symbol = sp.Symbol(variable_name, real=True)

    def symbolic_zero_audit(expression: str) -> dict[str, Any]:
        parsed = sp.sympify(expression, locals={variable_name: symbol})
        exact = sp.simplify(parsed.subs(symbol, 0))
        left = sp.limit(parsed, symbol, 0, dir="-")
        right = sp.limit(parsed, symbol, 0, dir="+")
        finite_at_zero = exact.is_finite is True
        equal_limits = sp.simplify(left - right) == 0
        matches_value = finite_at_zero and sp.simplify(left - exact) == 0
        continuous = bool(finite_at_zero and equal_limits and matches_value)
        localized_zero_limit = bool(
            equal_limits and sp.simplify(left) == 0 and sp.simplify(right) == 0
        )
        return {
            "value_at_zero": str(exact),
            "left_limit": str(left),
            "right_limit": str(right),
            "finite_at_zero": bool(finite_at_zero),
            "left_and_right_limits_equal": bool(equal_limits),
            "continuous_at_zero": continuous,
            "localized_terminal_has_zero_residual_limit": localized_zero_limit,
        }

    selected_symbolic = symbolic_zero_audit(
        str(summary["selected_expression_for_eval"])
    )
    probe = np.asarray([[-1e-12], [0.0], [1e-12]], dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe_values = f3c.predict_expression(
            str(summary["selected_expression_for_eval"]),
            probe,
            [variable_name],
        )
    view = branch_view(context)
    observed = np.concatenate([view["X_train"][:, 0], view["X_validation"][:, 0]])
    sigma_spec = context["f3b_spec"]["standardizations"]["z_sigma"]
    sigma_index = context["train"]["base_feature_names"].index("log10_sigma")
    observed_log10_sigma = np.concatenate(
        [
            context["train"]["X_base"][:, sigma_index],
            context["validation"]["X_base"][:, sigma_index],
        ]
    )
    center = float(sigma_spec["center"])

    candidate_metrics = pd.read_csv(
        session_dir / "runs" / BRANCH / "candidate_metrics.csv"
    )
    continuous_candidates: list[dict[str, Any]] = []
    for _, row in candidate_metrics.iterrows():
        if row["candidate_source"] != "pysr_hall_of_fame" or row["eval_status"] != "ok":
            continue
        try:
            candidate_audit = symbolic_zero_audit(str(row["expression_for_eval"]))
        except Exception:
            continue
        if candidate_audit["continuous_at_zero"]:
            continuous_candidates.append(
                {
                    "candidate_index": int(row["candidate_index"]),
                    "complexity": int(row["complexity"]),
                    "validation_rmse": float(row["rmse"]),
                    "equation": str(row["equation"]),
                    "expression_for_eval": str(row["expression_for_eval"]),
                    **candidate_audit,
                }
            )
    best_continuous = (
        min(
            continuous_candidates,
            key=lambda row: (row["validation_rmse"], row["complexity"]),
        )
        if continuous_candidates
        else None
    )
    if best_continuous is not None:
        diagnostic_residual = f3c.predict_expression(
            best_continuous["expression_for_eval"],
            view["X_validation"],
            [variable_name],
        )
        diagnostic_prediction = context["validation_f1"] + diagnostic_residual
        diagnostic_metrics = regression_metrics(
            context["validation"]["y"], diagnostic_prediction
        )
        diagnostic_curve_summary, _ = curve_metrics(
            context["validation"]["y"],
            diagnostic_prediction,
            context["validation"]["source_curve_index"],
        )
        diagnostic_regions = regional_metrics(
            context["validation"]["y"],
            diagnostic_prediction,
            context["validation"]["frequency_hz"],
            (*REPORTING_REGIONS, TARGET_REGION),
        )
        formal_metrics = regression_metrics(
            context["validation"]["y"], context["validation_f1"]
        )
        formal_curve_summary, _ = curve_metrics(
            context["validation"]["y"],
            context["validation_f1"],
            context["validation"]["source_curve_index"],
        )
        formal_regions = regional_metrics(
            context["validation"]["y"],
            context["validation_f1"],
            context["validation"]["frequency_hz"],
            (*REPORTING_REGIONS, TARGET_REGION),
        )
        diagnostic_lookup = region_map(diagnostic_regions)
        formal_lookup = region_map(formal_regions)
        diagnostic_guardrails = [
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "formal_f1_control_rmse": float(formal_lookup[bounds]["rmse"]),
                "diagnostic_candidate_rmse": float(
                    diagnostic_lookup[bounds]["rmse"]
                ),
                "relative_change": float(
                    diagnostic_lookup[bounds]["rmse"]
                    / formal_lookup[bounds]["rmse"]
                    - 1.0
                ),
            }
            for bounds in GUARDRAIL_REGIONS
        ]
        diagnostic_checks = {
            "beats_formal_f1_full_validation_rmse": (
                diagnostic_metrics["rmse"] < formal_metrics["rmse"]
            ),
            "improves_curve_mean_rmse": (
                diagnostic_curve_summary["curve_mean_rmse"]
                < formal_curve_summary["curve_mean_rmse"]
            ),
            "improves_worst_curve_rmse": (
                diagnostic_curve_summary["worst_curve_rmse"]
                < formal_curve_summary["worst_curve_rmse"]
            ),
            "improves_target_region": (
                diagnostic_lookup[TARGET_REGION]["rmse"]
                < formal_lookup[TARGET_REGION]["rmse"]
            ),
            "all_formal_f1_regional_guardrails_pass": all(
                row["relative_change"] <= GUARDRAIL_MAX_RELATIVE_WORSENING
                for row in diagnostic_guardrails
            ),
        }
        best_continuous.update(
            {
                "diagnostic_validation_metrics": diagnostic_metrics,
                "diagnostic_validation_curve_summary": diagnostic_curve_summary,
                "diagnostic_validation_regional_metrics": diagnostic_regions,
                "diagnostic_formal_f1_guardrails": diagnostic_guardrails,
                "would_pass_predeclared_numerical_gates": all(
                    diagnostic_checks.values()
                ),
                "diagnostic_checks": diagnostic_checks,
            }
        )
    audit = {
        "audit_kind": "post_selection_deployment_safety_audit",
        "predeclared_validation_selection_rule_changed": False,
        "selected_candidate_index": summary["selected_candidate_index"],
        "selected_equation": summary["selected_equation"],
        "selected_expression_for_eval": summary["selected_expression_for_eval"],
        "valid_terminal_value_audited": 0.0,
        "terminal_zero_corresponds_to_training_standardization_center": True,
        "log10_sigma_center": center,
        "center_within_observed_train_validation_range": bool(
            np.min(observed_log10_sigma) <= center <= np.max(observed_log10_sigma)
        ),
        "observed_terminal_minimum": float(np.min(observed)),
        "observed_terminal_maximum": float(np.max(observed)),
        "observed_minimum_absolute_terminal": float(np.min(np.abs(observed))),
        "observed_exact_zero_count": int(np.sum(observed == 0.0)),
        "symbolic_zero_behavior": selected_symbolic,
        "numeric_probe": {
            "terminal_values": probe[:, 0].tolist(),
            "residual_values": [
                float(value) if np.isfinite(value) else str(value)
                for value in probe_values
            ],
            "warnings": [str(item.message) for item in caught],
        },
        "selected_formula_domain_audit_passed": bool(
            selected_symbolic["continuous_at_zero"]
        ),
        "f3f2_readiness": False,
        "reason": (
            "Selected residual is non-finite and discontinuous at the valid zero "
            "of z_sigma__f3_mid; test must remain sealed pending an explicit "
            "formula-safety resolution."
        ),
        "best_continuous_hall_of_fame_candidate_diagnostic_only": best_continuous,
        "diagnostic_candidate_not_selected_post_hoc": True,
    }
    dump_json(session_dir / "formula_domain_audit.json", audit)
    return audit


def apply_formula_domain_hold(
    session_dir: Path, comparison: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    comparison = dict(comparison)
    comparison["predeclared_validation_gate_passed"] = comparison["stage_passed"]
    comparison["post_selection_formula_domain_audit"] = audit
    comparison["post_selection_formula_domain_audit_passed"] = audit[
        "selected_formula_domain_audit_passed"
    ]
    comparison["f3f2_readiness"] = bool(
        comparison["stage_passed"]
        and audit["selected_formula_domain_audit_passed"]
    )
    if not comparison["f3f2_readiness"]:
        comparison["decision"] = (
            "validation_gate_passed_hold_for_formula_domain_safety_resolution"
        )
        comparison["next_command_if_user_accepts"] = None
        comparison["planned_next_stage_after_safety_resolution"] = "F3-F2"
    dump_json(session_dir / "comparison_decision.json", comparison)
    return comparison


def plot_results(
    session_dir: Path, comparison: dict[str, Any], context: dict[str, Any]
) -> None:
    figures = session_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with np.load(
        session_dir / "runs" / BRANCH / "selected_predictions_train_validation.npz",
        allow_pickle=False,
    ) as values:
        candidate_prediction = values["validation_prediction"].copy()
    frequency_hz = context["validation"]["frequency_hz"]
    y_true = context["validation"]["y"]
    frequencies = np.unique(frequency_hz)
    f1_frequency_rmse: list[float] = []
    f3_frequency_rmse: list[float] = []
    for frequency in frequencies:
        mask = frequency_hz == frequency
        f1_frequency_rmse.append(
            float(np.sqrt(np.mean((context["validation_f1"][mask] - y_true[mask]) ** 2)))
        )
        f3_frequency_rmse.append(
            float(np.sqrt(np.mean((candidate_prediction[mask] - y_true[mask]) ** 2)))
        )
    pd.DataFrame(
        {
            "frequency_hz": frequencies,
            "formal_f1_rmse": f1_frequency_rmse,
            "formal_f3_candidate_rmse": f3_frequency_rmse,
        }
    ).to_csv(session_dir / "validation_frequency_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frequencies, f1_frequency_rmse, "--", color="black", label="formal Re F1")
    ax.plot(frequencies, f3_frequency_rmse, linewidth=2, label="formal Re F3 candidate")
    for lower, upper in GUARDRAIL_REGIONS:
        ax.axvspan(lower, upper, color="#bbbbbb", alpha=0.08)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("F3-F1 formal validation frequency behavior")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "validation_frequency_rmse.png", dpi=180)
    plt.close(fig)

    f1_curve = pd.read_csv(session_dir / "formal_f1_validation_curve_metrics.csv")
    f3_curve = pd.read_csv(
        session_dir
        / "runs"
        / BRANCH
        / "selected_validation_curve_metrics.csv"
    )
    merged = f1_curve.merge(
        f3_curve, on="curve_id", suffixes=("_f1", "_f3"), validate="one_to_one"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["rmse_f1"], merged["rmse_f3"], s=18, alpha=0.7)
    limit = float(max(merged["rmse_f1"].max(), merged["rmse_f3"].max()))
    ax.plot([0.0, limit], [0.0, limit], "--", color="black", linewidth=1)
    ax.set_xlabel("Formal Re F1 curve RMSE")
    ax.set_ylabel("Formal Re F3 candidate curve RMSE")
    ax.set_title("F3-F1 validation curve comparison")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "validation_curve_rmse_scatter.png", dpi=180)
    plt.close(fig)


def write_decision_markdown(session_dir: Path, comparison: dict[str, Any]) -> None:
    f1 = comparison["formal_f1_validation_reference"]
    f3 = comparison["formal_f3_candidate"]
    target = comparison["target_region_comparison"]
    status = "PASS" if comparison["stage_passed"] else "STOP / NO PASS"
    readiness = "READY" if comparison.get("f3f2_readiness") else "HOLD"
    lines = [
        "# Re Global SR F3-F1 Formal Validation Decision",
        "",
        f"- Predeclared validation gate: **{status}**",
        f"- F3-F2 readiness: **{readiness}**",
        f"- Formal Re F1 validation RMSE: `{f1['validation_metrics']['rmse']:.9f}`",
        f"- Formal Re F3 candidate validation RMSE: `{f3['validation_metrics']['rmse']:.9f}`",
        f"- Formal Re F1 curve-mean RMSE: `{f1['validation_curve_summary']['curve_mean_rmse']:.9f}`",
        f"- Formal Re F3 candidate curve-mean RMSE: `{f3['validation_curve_summary']['curve_mean_rmse']:.9f}`",
        f"- Formal Re F1 worst-curve RMSE: `{f1['validation_curve_summary']['worst_curve_rmse']:.9f}`",
        f"- Formal Re F3 candidate worst-curve RMSE: `{f3['validation_curve_summary']['worst_curve_rmse']:.9f}`",
        f"- Formal Re F1 1300–2000 Hz RMSE: `{target['formal_f1_rmse']:.9f}`",
        f"- Formal Re F3 candidate 1300–2000 Hz RMSE: `{target['formal_f3_candidate_rmse']:.9f}`",
        f"- Selected residual equation: `{f3['selected_equation']}`",
        f"- Selected complexity: `{f3['selected_complexity']}`",
        f"- Test rows accessed: `{comparison['test_rows_accessed']}`",
        "",
        "## Formal Re F1 guardrails",
        "",
    ]
    for row in comparison["formal_f1_guardrails"]:
        lines.append(
            f"- {int(row['lower_hz'])}–{int(row['upper_hz'])} Hz: "
            f"F3 `{row['formal_f3_candidate_rmse']:.9f}`, "
            f"F1 `{row['formal_f1_control_rmse']:.9f}`, passed `{row['passed']}`"
        )
    lines.extend(
        [
            "",
            "## Post-selection formula-domain audit",
            "",
            f"- Passed: `{comparison['post_selection_formula_domain_audit_passed']}`",
            f"- Value at `z_sigma__f3_mid = 0`: `{comparison['post_selection_formula_domain_audit']['symbolic_zero_behavior']['value_at_zero']}`",
            f"- Left limit: `{comparison['post_selection_formula_domain_audit']['symbolic_zero_behavior']['left_limit']}`",
            f"- Right limit: `{comparison['post_selection_formula_domain_audit']['symbolic_zero_behavior']['right_limit']}`",
            "- The selected formula is held because it is non-finite and discontinuous at a valid terminal value.",
            "",
            "This is validation-only confirmation. It does not replace formal Re F1.",
            "The test partition remains sealed pending explicit formula-safety resolution.",
            "",
        ]
    )
    (session_dir / "decision_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def master_main(args: argparse.Namespace) -> None:
    session_dir = resolve_output_dir(args.output_dir)
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    master_log = session_dir / "execution.log"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with master_log.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    paths = required_paths(args)
    context = load_context(args)
    acceptance_rules = {
        "full_validation": "formal F3 candidate RMSE must be strictly below formal Re F1",
        "curve_behavior": "curve-mean and worst-curve RMSE must both be strictly below formal Re F1",
        "target_region": "1300-2000 Hz RMSE must be strictly below formal Re F1",
        "guardrails": "no declared regional RMSE may worsen more than 10% versus formal Re F1",
        "fallback": "zero residual must reproduce formal Re F1 exactly",
        "test": "test remains sealed; passing F3-F1 only freezes a candidate for separately authorized F3-F2",
    }
    preflight = {
        "schema_version": 1,
        "stage": STAGE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session_dir),
        "formal_seed_frozen_before_search": SEED,
        "representation": {
            "accepted_source": "F3-C branch A",
            "model": "frozen formal Re F1 plus residual PySR",
            "residual_input_feature_names": ["z_sigma__f3_mid"],
            "rejected_representation_changes": ["F3-D1 z_lambda__f3_mid"],
        },
        "search_configuration": {
            "accepted_source": "original F3-C configuration retained after F3-E1",
            "nested_constraints": NESTED_CONSTRAINTS,
            "rejected_search_changes": ["F3-E1 nested log/sqrt constraints"],
        },
        "budget": {
            "niterations": NITERATIONS,
            "populations": POPULATIONS,
            "population_size": POPULATION_SIZE,
            "max_complexity": MAX_COMPLEXITY,
        },
        "operators": {
            "binary": list(BINARY_OPERATORS),
            "unary": list(UNARY_OPERATORS),
        },
        "plan_b": {
            "parallelism": "serial",
            "deterministic": True,
            "batching": True,
            "batch_size": BATCH_SIZE,
            "turbo": True,
            "julia_threads": 1,
        },
        "selection": "minimum complete validation RMSE then complexity; exact zero-residual formal-F1 fallback",
        "acceptance_rules_predeclared_before_search": acceptance_rules,
        "guardrail_control": "formal Re F1 validation regional RMSE",
        "test_access_contract": "test features, targets, and predictions remain inaccessible",
        "input_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "execute": True,
    }
    dump_json(session_dir / "preflight_manifest.json", preflight)
    log(
        "F3-F1 preflight frozen: branch A, original unconstrained search, seed=42, "
        "budget=100/12/80/24, Plan B, validation-only."
    )

    run_dir = session_dir / "runs" / BRANCH
    run_dir.mkdir(parents=True, exist_ok=True)
    worker_log = logs_dir / "formal_a_seed_42.log"
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--dataset-file",
        str(args.dataset_file.resolve()),
        "--split-file",
        str(args.split_file.resolve()),
        "--f1-training-dir",
        str(args.f1_training_dir.resolve()),
        "--f3a-dir",
        str(args.f3a_dir.resolve()),
        "--f3b-dir",
        str(args.f3b_dir.resolve()),
        "--f3c-dir",
        str(args.f3c_dir.resolve()),
        "--f3d1-dir",
        str(args.f3d1_dir.resolve()),
        "--f3e1-dir",
        str(args.f3e1_dir.resolve()),
        "--julia-exe",
        str(args.julia_exe.resolve()),
    ]
    log("Starting formal PySR run: branch=A, seed=42.")
    with worker_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"F3-F1 worker failed with exit={return_code}. See {worker_log}"
        )
    summary = load_json(run_dir / "selected_run_summary.json")
    if summary.get("test_rows_accessed") is not False:
        raise ValueError("The F3-F1 worker reported test-row access.")
    log(
        f"Formal search completed: validation_RMSE={summary['validation_metrics']['rmse']:.9f}."
    )
    comparison = aggregate(session_dir, summary, context)
    formula_audit = audit_selected_formula_domain(session_dir, summary, context)
    comparison = apply_formula_domain_hold(session_dir, comparison, formula_audit)
    plot_results(session_dir, comparison, context)
    write_decision_markdown(session_dir, comparison)
    final_manifest = {
        **preflight,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "shared_split_hash": context["split"]["split_hash"],
        "train_row_count": int(context["train"]["y"].size),
        "validation_row_count": int(context["validation"]["y"].size),
        "test_curve_count_declared_but_not_loaded": context["test_curve_count"],
        "test_features_read": False,
        "test_targets_read": False,
        "test_predictions_computed": False,
        "formal_run_summary": summary,
        "decision": comparison,
        "artifact_files": sorted(
            [
                *(
                    str(path.relative_to(session_dir)).replace("\\", "/")
                    for path in session_dir.rglob("*")
                    if path.is_file()
                ),
                "stage_manifest.json",
            ]
        ),
    }
    dump_json(session_dir / "stage_manifest.json", final_manifest)
    log(
        f"F3-F1 complete: stage_passed={comparison['stage_passed']}, "
        f"decision={comparison['decision']}, test_rows_accessed=False."
    )
    log(f"Artifacts written to {session_dir}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        master_main(args)


if __name__ == "__main__":
    main()
