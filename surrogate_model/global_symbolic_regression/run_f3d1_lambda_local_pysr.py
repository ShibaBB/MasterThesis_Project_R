"""Run validation-only Re Global SR F3-D1 against the saved F3-C branch A."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
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
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURROGATE_ROOT))

import run_f3c_residual_pysr_pilot as f3c  # noqa: E402
from evaluate_global_symbolic_candidates import predict_expression  # noqa: E402
from f3_local_terminals import _raw_envelope  # noqa: E402
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


STAGE = "F3-D1"
BRANCH = "b"
SEEDS = f3c.SEEDS
NITERATIONS = 20
POPULATIONS = 6
POPULATION_SIZE = 40
MAX_COMPLEXITY = 24
BATCH_SIZE = 4096
BINARY_OPERATORS = ("+", "-", "*", "/")
UNARY_OPERATORS = ("log", "sqrt")
DEFAULT_F3C_DIR = F3_ARTIFACT_ROOT / "F3-C" / "20260817T115632"


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
    parser.add_argument("--julia-exe", type=Path, default=f3c.DEFAULT_JULIA_EXE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact session directory; it must remain under artifacts/F3.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
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
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = unique_path(root / STAGE / timestamp)
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
    f3c_dir = args.f3c_dir.resolve()
    paths.update(
        {
            "f3a_standardization": args.f3a_dir.resolve()
            / "training_only_standardization.json",
            "f3c_decision": f3c_dir / "comparison_decision.json",
            "f3c_manifest": f3c_dir / "stage_manifest.json",
        }
    )
    for seed in SEEDS:
        source = f3c_dir / "runs" / "a" / f"seed_{seed}"
        paths[f"f3c_a_seed_{seed}_summary"] = source / "selected_run_summary.json"
        paths[f"f3c_a_seed_{seed}_predictions"] = (
            source / "selected_predictions_train_validation.npz"
        )
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{STAGE} required inputs are missing: {missing}")
    return paths


def search_contract() -> dict[str, Any]:
    return {
        "niterations": NITERATIONS,
        "populations": POPULATIONS,
        "population_size": POPULATION_SIZE,
        "max_complexity": MAX_COMPLEXITY,
        "binary_operators": list(BINARY_OPERATORS),
        "unary_operators": list(UNARY_OPERATORS),
        "parallelism": "serial",
        "deterministic": True,
        "batching": True,
        "batch_size": BATCH_SIZE,
        "turbo": True,
        "julia_threads": 1,
        "candidate_selection": "minimum_complete_validation_RMSE_then_complexity",
    }


def _validate_saved_a_summary(summary: dict[str, Any], seed: int) -> None:
    expected = search_contract()
    if summary.get("stage") != "F3-C" or summary.get("branch") != "a":
        raise ValueError(f"Saved seed {seed} control is not F3-C branch A.")
    if int(summary.get("seed", -1)) != seed:
        raise ValueError(f"Saved branch A seed mismatch for seed {seed}.")
    if summary.get("input_feature_names") != ["z_sigma__f3_mid"]:
        raise ValueError("Saved branch A representation changed.")
    if summary.get("test_rows_accessed") is not False:
        raise ValueError("Saved branch A reports test-row access.")
    for key, value in expected.items():
        if summary["search_contract"].get(key) != value:
            raise ValueError(f"Saved branch A contract differs at {key}.")


def _normalized_g_mid(X_base: np.ndarray, base_names: list[str], spec: dict[str, Any]) -> np.ndarray:
    frequency_index = base_names.index(spec["frequency_feature"])
    envelope_spec = spec["envelopes"]["g_mid"]
    envelope = _raw_envelope(X_base[:, frequency_index], envelope_spec)
    envelope = envelope / float(envelope_spec["normalization_max"])
    if not np.all(np.isfinite(envelope)):
        raise ValueError("The frozen g_mid envelope produced non-finite values.")
    return envelope


def _training_unique_values(
    values: np.ndarray, curve_ids: np.ndarray
) -> np.ndarray:
    unique_ids, first_indices = np.unique(curve_ids, return_index=True)
    if unique_ids.size != 700:
        raise ValueError("D1 expected exactly 700 training curves.")
    unique_values = values[first_indices]
    replay = dict(zip(unique_ids.tolist(), unique_values.tolist(), strict=True))
    expected_rows = np.asarray([replay[int(curve_id)] for curve_id in curve_ids])
    if not np.array_equal(values, expected_rows):
        raise ValueError("log10_lambda is not constant within a source curve.")
    return unique_values


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    paths = required_paths(args)
    context = f3c.load_context(args)
    decision = load_json(paths["f3c_decision"])
    manifest = load_json(paths["f3c_manifest"])
    if decision.get("stage_passed") is not True:
        raise ValueError("F3-C must pass before F3-D1.")
    if decision.get("decision") != "advance_branch_A_representation_to_next_explicit_stage":
        raise ValueError("F3-C did not authorize branch A as the D1 predecessor.")
    if decision.get("test_rows_accessed") is not False:
        raise ValueError("F3-C decision reports test-row access.")
    if any(
        manifest.get(key) is not False
        for key in ("test_features_read", "test_targets_read", "test_predictions_computed")
    ):
        raise ValueError("F3-C manifest does not preserve test isolation.")
    if manifest.get("shared_split_hash") != context["split"]["split_hash"]:
        raise ValueError("F3-C and F3-D1 split hashes differ.")

    standardizations = load_json(paths["f3a_standardization"])
    stored_lambda = standardizations["log10_lambda"]
    if stored_lambda.get("method") != "training_unique_curve_population_mean_std_ddof0":
        raise ValueError("Unexpected z_lambda standardization method.")
    if int(stored_lambda.get("training_curve_count", -1)) != 700:
        raise ValueError("z_lambda standardization was not fitted on 700 training curves.")
    center = float(stored_lambda["center"])
    scale = float(stored_lambda["scale"])
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Stored z_lambda normalization is invalid.")

    base_names = list(context["train"]["base_feature_names"])
    lambda_index = base_names.index("log10_lambda")
    train_lambda = context["train"]["X_base"][:, lambda_index]
    unique_lambda = _training_unique_values(
        train_lambda, context["train"]["source_curve_index"]
    )
    recomputed_center = float(np.mean(unique_lambda))
    recomputed_scale = float(np.std(unique_lambda, ddof=0))
    if recomputed_center != center or recomputed_scale != scale:
        raise ValueError("Stored z_lambda normalization does not replay exactly.")

    terminal_index = context["f3_feature_names"].index("z_sigma__f3_mid")
    train_g_mid = _normalized_g_mid(
        context["train"]["X_base"], base_names, context["f3b_spec"]
    )
    validation_g_mid = _normalized_g_mid(
        context["validation"]["X_base"], base_names, context["f3b_spec"]
    )
    train_z_lambda = (train_lambda - center) / scale
    validation_lambda = context["validation"]["X_base"][:, lambda_index]
    validation_z_lambda = (validation_lambda - center) / scale
    train_lambda_terminal = train_z_lambda * train_g_mid
    validation_lambda_terminal = validation_z_lambda * validation_g_mid
    train_sigma_terminal = context["train_f3"][:, terminal_index]
    validation_sigma_terminal = context["validation_f3"][:, terminal_index]
    train_b = np.column_stack([train_sigma_terminal, train_lambda_terminal])
    validation_b = np.column_stack(
        [validation_sigma_terminal, validation_lambda_terminal]
    )
    if not np.all(np.isfinite(train_b)) or not np.all(np.isfinite(validation_b)):
        raise ValueError("D1 branch B terminals contain non-finite values.")

    control_summaries: dict[int, dict[str, Any]] = {}
    control_predictions: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        summary = load_json(paths[f"f3c_a_seed_{seed}_summary"])
        _validate_saved_a_summary(summary, seed)
        if summary.get("shared_split_hash") != context["split"]["split_hash"]:
            raise ValueError(f"Saved branch A seed {seed} split hash differs.")
        control_summaries[seed] = summary
        with np.load(paths[f"f3c_a_seed_{seed}_predictions"], allow_pickle=False) as values:
            if not np.array_equal(
                values["train_curve_ids"], context["train"]["source_curve_index"]
            ):
                raise ValueError(f"Saved branch A seed {seed} training rows differ.")
            if not np.array_equal(
                values["validation_curve_ids"],
                context["validation"]["source_curve_index"],
            ):
                raise ValueError(f"Saved branch A seed {seed} validation rows differ.")
            if not np.array_equal(
                values["train_frequency_hz"], context["train"]["frequency_hz"]
            ) or not np.array_equal(
                values["validation_frequency_hz"],
                context["validation"]["frequency_hz"],
            ):
                raise ValueError(f"Saved branch A seed {seed} frequency rows differ.")
            control_predictions[seed] = {
                "train": values["train_prediction"].copy(),
                "validation": values["validation_prediction"].copy(),
            }
        replay_metrics = regression_metrics(
            context["validation"]["y"], control_predictions[seed]["validation"]
        )
        if replay_metrics["rmse"] != summary["validation_metrics"]["rmse"]:
            raise ValueError(f"Saved branch A seed {seed} metrics do not replay exactly.")

    context.update(
        {
            "d1_paths": paths,
            "f3c_decision": decision,
            "lambda_center": center,
            "lambda_scale": scale,
            "train_g_mid": train_g_mid,
            "validation_g_mid": validation_g_mid,
            "train_z_lambda": train_z_lambda,
            "validation_z_lambda": validation_z_lambda,
            "train_lambda_terminal": train_lambda_terminal,
            "validation_lambda_terminal": validation_lambda_terminal,
            "train_b": train_b,
            "validation_b": validation_b,
            "control_summaries": control_summaries,
            "control_predictions": control_predictions,
        }
    )
    return context


def d1_terminal_spec(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    envelope = context["f3b_spec"]["envelopes"]["g_mid"]
    return {
        "schema_version": 1,
        "stage_frozen": STAGE,
        "name": "re_f3_lambda_local_bandpass_v1",
        "shared_split_hash": context["split"]["split_hash"],
        "source_feature": "log10_lambda",
        "standardized_feature": "z_lambda",
        "standardization": {
            "center": context["lambda_center"],
            "scale": context["lambda_scale"],
            "fitted_partition": "train",
            "fitted_unique_curve_count": 700,
            "method": "training_unique_curve_population_mean_std_ddof0",
            "ddof": 0,
            "source_file": str(
                args.f3a_dir.resolve() / "training_only_standardization.json"
            ),
        },
        "envelope": {
            "name": "g_mid",
            "source_stage": "F3-B",
            "source_file": str(args.f3b_dir.resolve() / "frozen_sigma_local_spec.json"),
            **envelope,
        },
        "terminal": {
            "name": "z_lambda__f3_mid",
            "analytic_definition": "((log10_lambda - center) / scale) * g_mid(log10_f)",
        },
        "branch_b_feature_names": ["z_sigma__f3_mid", "z_lambda__f3_mid"],
        "teacher_or_sobol_values_used_as_row_inputs": False,
        "test_rows_accessed": False,
    }


def branch_view(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "X_train": context["train_b"],
        "X_validation": context["validation_b"],
        "y_search_train": context["train"]["y"] - context["train_f1"],
        "variable_names": ["z_sigma__f3_mid", "z_lambda__f3_mid"],
        "feature_names": ["z_sigma__f3_mid", "z_lambda__f3_mid"],
        "search_target": "delta_Re = teacher - frozen_Re_F1",
        "final_offset_train": context["train_f1"],
        "final_offset_validation": context["validation_f1"],
    }


def score_candidates(
    equations: pd.DataFrame,
    view: dict[str, Any],
    context: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    validation_predictions: dict[int, np.ndarray] = {}

    zero_prediction = view["final_offset_validation"].copy()
    rows.append(
        {
            "candidate_index": 0,
            "candidate_source": "explicit_zero_residual_f1_fallback",
            "complexity": 0,
            "pysr_loss": math.nan,
            "pysr_score": math.nan,
            "equation": "0.0",
            "expression_for_eval": "0.0",
            "eval_status": "ok",
            **regression_metrics(context["validation"]["y"], zero_prediction),
        }
    )
    validation_predictions[0] = zero_prediction

    predecessor = context["control_summaries"][seed]
    predecessor_prediction = context["control_predictions"][seed]["validation"].copy()
    rows.append(
        {
            "candidate_index": 1,
            "candidate_source": "explicit_saved_f3c_branch_a_fallback",
            "complexity": int(predecessor["selected_complexity"]),
            "pysr_loss": math.nan,
            "pysr_score": math.nan,
            "equation": str(predecessor["selected_equation"]),
            "expression_for_eval": str(predecessor["selected_expression_for_eval"]),
            "eval_status": "ok",
            **regression_metrics(
                context["validation"]["y"], predecessor_prediction
            ),
        }
    )
    validation_predictions[1] = predecessor_prediction

    expression_column = "sympy_format" if "sympy_format" in equations else "equation"
    for row_index, equation_row in equations.reset_index(drop=True).iterrows():
        candidate_index = row_index + 2
        expression = str(equation_row[expression_column])
        try:
            searched_prediction = predict_expression(
                expression, view["X_validation"], view["variable_names"]
            )
            final_prediction = view["final_offset_validation"] + searched_prediction
            if not np.all(np.isfinite(final_prediction)):
                raise ValueError("candidate prediction is non-finite")
            metrics = regression_metrics(context["validation"]["y"], final_prediction)
            validation_predictions[candidate_index] = final_prediction
            status = "ok"
        except Exception as error:
            metrics = {
                "rmse": math.inf,
                "mae": math.inf,
                "max_abs_error": math.inf,
                "r2": -math.inf,
                "bias": math.nan,
                "prediction_min": math.nan,
                "prediction_max": math.nan,
            }
            status = f"failed: {error}"
        rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_source": "pysr_hall_of_fame",
                "complexity": int(equation_row["complexity"]),
                "pysr_loss": float(equation_row["loss"]),
                "pysr_score": (
                    float(equation_row["score"])
                    if "score" in equation_row and pd.notna(equation_row["score"])
                    else math.nan
                ),
                "equation": str(equation_row["equation"]),
                "expression_for_eval": expression,
                "eval_status": status,
                **metrics,
            }
        )
    return pd.DataFrame(rows), validation_predictions


def evaluate_selected(
    selected: pd.Series,
    view: dict[str, Any],
    context: dict[str, Any],
    seed: int,
    validation_prediction: np.ndarray,
) -> dict[str, Any]:
    source = str(selected["candidate_source"])
    if source == "explicit_zero_residual_f1_fallback":
        train_prediction = view["final_offset_train"].copy()
    elif source == "explicit_saved_f3c_branch_a_fallback":
        train_prediction = context["control_predictions"][seed]["train"].copy()
    else:
        searched_train = predict_expression(
            str(selected["expression_for_eval"]),
            view["X_train"],
            view["variable_names"],
        )
        train_prediction = view["final_offset_train"] + searched_train
    if not np.all(np.isfinite(train_prediction)):
        raise ValueError("Selected candidate has non-finite training predictions.")
    train_curve_summary, _ = curve_metrics(
        context["train"]["y"],
        train_prediction,
        context["train"]["source_curve_index"],
    )
    validation_curve_summary, validation_curve_rows = curve_metrics(
        context["validation"]["y"],
        validation_prediction,
        context["validation"]["source_curve_index"],
    )
    validation_regions = regional_metrics(
        context["validation"]["y"],
        validation_prediction,
        context["validation"]["frequency_hz"],
        (*REPORTING_REGIONS, TARGET_REGION),
    )
    return {
        "train_prediction": train_prediction,
        "validation_prediction": validation_prediction,
        "train_metrics": regression_metrics(context["train"]["y"], train_prediction),
        "validation_metrics": regression_metrics(
            context["validation"]["y"], validation_prediction
        ),
        "train_curve_summary": train_curve_summary,
        "validation_curve_summary": validation_curve_summary,
        "validation_curve_rows": validation_curve_rows,
        "validation_regional_metrics": validation_regions,
    }


def worker_main(args: argparse.Namespace) -> None:
    if args.seed is None or args.run_dir is None:
        raise ValueError("Worker mode requires --seed and --run-dir.")
    if args.seed not in SEEDS:
        raise ValueError(f"Worker seed is outside the frozen D1 seeds: {args.seed}")
    run_dir = args.run_dir.resolve()
    session_dir = run_dir.parents[2]
    if not run_dir.is_relative_to(session_dir / "runs"):
        raise ValueError("Worker output must remain inside the D1 session runs directory.")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JULIA_NUM_THREADS"] = "1"
    julia_exe = args.julia_exe.resolve()
    if not julia_exe.exists():
        raise FileNotFoundError(f"Julia executable not found: {julia_exe}")
    os.environ.setdefault("PYTHON_JULIAPKG_EXE", str(julia_exe))

    print(
        f"Worker start: stage={STAGE}, branch={BRANCH}, seed={args.seed}, "
        f"budget={NITERATIONS}/{POPULATIONS}/{POPULATION_SIZE}/{MAX_COMPLEXITY}",
        flush=True,
    )
    context = load_context(args)
    view = branch_view(context)
    model = f3c.build_model(args.seed, run_dir, f"f3d1_b_s{args.seed}")
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

    candidate_metrics, predictions = score_candidates(
        equations, view, context, args.seed
    )
    candidate_metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)
    valid = candidate_metrics[candidate_metrics["eval_status"] == "ok"].copy()
    if valid.empty:
        raise RuntimeError("No valid D1 candidate predictions were produced.")
    valid = valid.sort_values(["rmse", "complexity"], ascending=[True, True])
    selected = valid.iloc[0]
    selected_index = int(selected["candidate_index"])
    evaluated = evaluate_selected(
        selected, view, context, args.seed, predictions[selected_index]
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

    predecessor = context["control_summaries"][args.seed]
    summary = {
        "stage": STAGE,
        "branch": BRANCH,
        "seed": args.seed,
        "search_target": view["search_target"],
        "input_feature_names": view["feature_names"],
        "pysr_variable_names": view["variable_names"],
        "explicit_zero_f1_fallback": True,
        "explicit_saved_a_predecessor_fallback": True,
        "predecessor_a_validation_rmse": predecessor["validation_metrics"]["rmse"],
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
        "lambda_terminal": {
            "name": "z_lambda__f3_mid",
            "center": context["lambda_center"],
            "scale": context["lambda_scale"],
            "fitted_partition": "train",
        },
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
        f"Worker complete: branch={BRANCH}, seed={args.seed}, "
        f"validation_RMSE={evaluated['validation_metrics']['rmse']:.9f}, "
        f"selected={selected_index}, source={selected['candidate_source']}, "
        f"complexity={int(selected['complexity'])}, fit_seconds={fit_seconds:.1f}",
        flush=True,
    )


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def region_map(rows: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (float(row["lower_hz"]), float(row["upper_hz"])): row for row in rows
    }


def branch_median(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "validation_rmse": median(
            [float(row["validation_metrics"]["rmse"]) for row in summaries]
        ),
        "validation_mae": median(
            [float(row["validation_metrics"]["mae"]) for row in summaries]
        ),
        "validation_r2": median(
            [float(row["validation_metrics"]["r2"]) for row in summaries]
        ),
        "curve_mean_rmse": median(
            [float(row["validation_curve_summary"]["curve_mean_rmse"]) for row in summaries]
        ),
        "worst_curve_rmse": median(
            [float(row["validation_curve_summary"]["worst_curve_rmse"]) for row in summaries]
        ),
        "fit_wall_time_seconds": median(
            [float(row["fit_wall_time_seconds"]) for row in summaries]
        ),
        "regional_metrics": [],
    }
    for bounds in (*REPORTING_REGIONS, TARGET_REGION):
        matching = [
            region_map(row["validation_regional_metrics"])[bounds]
            for row in summaries
        ]
        result["regional_metrics"].append(
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "median_rmse": median([float(row["rmse"]) for row in matching]),
                "median_mae": median([float(row["mae"]) for row in matching]),
                "median_bias": median([float(row["bias"]) for row in matching]),
            }
        )
    return result


def aggregate_runs(
    session_dir: Path,
    b_summaries: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    a_summaries = [context["control_summaries"][seed] for seed in SEEDS]
    all_summaries = [("a", "saved_F3-C", row) for row in a_summaries] + [
        ("b", "new_F3-D1", row) for row in b_summaries
    ]
    seed_rows: list[dict[str, Any]] = []
    regional_rows: list[dict[str, Any]] = []
    for branch, origin, summary in all_summaries:
        seed_rows.append(
            {
                "branch": branch,
                "result_origin": origin,
                "seed": summary["seed"],
                "selected_candidate_index": summary["selected_candidate_index"],
                "selected_candidate_source": summary["selected_candidate_source"],
                "selected_complexity": summary["selected_complexity"],
                "fit_wall_time_seconds": summary["fit_wall_time_seconds"],
                **summary["validation_metrics"],
                **summary["validation_curve_summary"],
                "selected_equation": summary["selected_equation"],
            }
        )
        for region in summary["validation_regional_metrics"]:
            regional_rows.append(
                {
                    "branch": branch,
                    "result_origin": origin,
                    "seed": summary["seed"],
                    **region,
                }
            )
    pd.DataFrame(seed_rows).to_csv(session_dir / "seed_metrics.csv", index=False)
    pd.DataFrame(regional_rows).to_csv(
        session_dir / "seed_regional_metrics.csv", index=False
    )

    medians = {"a": branch_median(a_summaries), "b": branch_median(b_summaries)}
    a_regions = region_map(medians["a"]["regional_metrics"])
    b_regions = region_map(medians["b"]["regional_metrics"])
    median_guardrails: list[dict[str, Any]] = []
    for bounds in GUARDRAIL_REGIONS:
        a_rmse = float(a_regions[bounds]["median_rmse"])
        b_rmse = float(b_regions[bounds]["median_rmse"])
        change = b_rmse / a_rmse - 1.0
        median_guardrails.append(
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "saved_a_control_median_rmse": a_rmse,
                "branch_b_median_rmse": b_rmse,
                "relative_change": change,
                "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                "passed": change <= GUARDRAIL_MAX_RELATIVE_WORSENING,
            }
        )
    per_seed_guardrails: list[dict[str, Any]] = []
    for seed in SEEDS:
        a_summary = context["control_summaries"][seed]
        b_summary = next(row for row in b_summaries if int(row["seed"]) == seed)
        a_lookup = region_map(a_summary["validation_regional_metrics"])
        b_lookup = region_map(b_summary["validation_regional_metrics"])
        for bounds in GUARDRAIL_REGIONS:
            a_rmse = float(a_lookup[bounds]["rmse"])
            b_rmse = float(b_lookup[bounds]["rmse"])
            change = b_rmse / a_rmse - 1.0
            per_seed_guardrails.append(
                {
                    "seed": seed,
                    "lower_hz": bounds[0],
                    "upper_hz": bounds[1],
                    "saved_a_control_rmse": a_rmse,
                    "branch_b_rmse": b_rmse,
                    "relative_change": change,
                    "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                    "passed": change <= GUARDRAIL_MAX_RELATIVE_WORSENING,
                }
            )

    target_a = float(a_regions[TARGET_REGION]["median_rmse"])
    target_b = float(b_regions[TARGET_REGION]["median_rmse"])
    formal = context["f3c_decision"]["formal_f1_validation_reference"]
    formal_target = float(
        region_map(formal["validation_regional_metrics"])[TARGET_REGION]["rmse"]
    )
    checks = {
        "all_three_b_runs_completed": len(b_summaries) == 3,
        "all_runs_validation_only": all(
            row.get("test_rows_accessed") is False for row in b_summaries
        ),
        "matched_predeclared_seed_sets": (
            {int(row["seed"]) for row in b_summaries}
            == set(context["control_summaries"])
            == set(SEEDS)
        ),
        "all_b_runs_preserve_exact_f1_and_saved_a_fallbacks": all(
            row.get("explicit_zero_f1_fallback") is True
            and row.get("explicit_saved_a_predecessor_fallback") is True
            and row["validation_metrics"]["rmse"]
            <= context["control_summaries"][int(row["seed"])]["validation_metrics"]["rmse"]
            for row in b_summaries
        ),
        "b_beats_saved_a_median_full_validation_rmse": (
            medians["b"]["validation_rmse"] < medians["a"]["validation_rmse"]
        ),
        "b_improves_median_curve_mean_rmse": (
            medians["b"]["curve_mean_rmse"] < medians["a"]["curve_mean_rmse"]
        ),
        "b_improves_median_worst_curve_rmse": (
            medians["b"]["worst_curve_rmse"] < medians["a"]["worst_curve_rmse"]
        ),
        "b_improves_target_region_vs_saved_a": target_b < target_a,
        "all_median_predecessor_regional_guardrails_pass": all(
            row["passed"] for row in median_guardrails
        ),
        "all_per_seed_predecessor_regional_guardrails_pass": all(
            row["passed"] for row in per_seed_guardrails
        ),
    }
    stage_passed = all(checks.values())
    comparison = {
        "stage": STAGE,
        "stage_passed": stage_passed,
        "decision": (
            "accept_branch_B_as_predecessor_for_next_explicit_stage"
            if stage_passed
            else "retain_saved_branch_A_representation"
        ),
        "formal_model_replacement_decision": False,
        "small_budget_models_are_not_formal_replacements": True,
        "seeds": list(SEEDS),
        "branch_medians": medians,
        "formal_f1_validation_reference": formal,
        "target_region_comparison": {
            "region_hz": list(TARGET_REGION),
            "saved_branch_a_median_rmse": target_a,
            "branch_b_median_rmse": target_b,
            "formal_f1_rmse": formal_target,
        },
        "predecessor_a_guardrails": median_guardrails,
        "per_seed_predecessor_a_guardrails": per_seed_guardrails,
        "checks": checks,
        "test_rows_accessed": False,
        "next_stage_automatically_authorized": False,
        "next_command_if_user_accepts": "执行 F3-D2" if stage_passed else None,
    }
    dump_json(session_dir / "comparison_decision.json", comparison)
    pd.DataFrame(
        [
            {
                "branch": branch,
                **{key: value for key, value in metrics.items() if key != "regional_metrics"},
            }
            for branch, metrics in medians.items()
        ]
    ).to_csv(session_dir / "median_metrics.csv", index=False)
    pd.DataFrame(median_guardrails).to_csv(
        session_dir / "predecessor_guardrail_comparison.csv", index=False
    )
    pd.DataFrame(per_seed_guardrails).to_csv(
        session_dir / "per_seed_predecessor_guardrail_comparison.csv", index=False
    )
    return comparison


def load_b_prediction(session_dir: Path, seed: int) -> np.ndarray:
    with np.load(
        session_dir
        / "runs"
        / BRANCH
        / f"seed_{seed}"
        / "selected_predictions_train_validation.npz",
        allow_pickle=False,
    ) as values:
        return values["validation_prediction"].copy()


def plot_results(
    session_dir: Path,
    b_summaries: list[dict[str, Any]],
    context: dict[str, Any],
) -> None:
    figures = session_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    a_summaries = context["control_summaries"]
    formal_rmse = float(
        context["f3c_decision"]["formal_f1_validation_reference"]["validation_metrics"]["rmse"]
    )
    x = np.arange(len(SEEDS))
    width = 0.36
    a_rmse = [a_summaries[seed]["validation_metrics"]["rmse"] for seed in SEEDS]
    b_rmse = [
        next(row["validation_metrics"]["rmse"] for row in b_summaries if row["seed"] == seed)
        for seed in SEEDS
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, a_rmse, width, label="A saved F3-C predecessor")
    ax.bar(x + width / 2, b_rmse, width, label="B + z_lambda x g_mid")
    ax.axhline(formal_rmse, color="black", linestyle="--", label="formal Re F1")
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Complete validation RMSE")
    ax.set_title("F3-D1 matched three-seed validation comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "three_seed_validation_rmse.png", dpi=180)
    plt.close(fig)

    frequency_hz = context["validation"]["frequency_hz"]
    y_true = context["validation"]["y"]
    frequencies = np.unique(frequency_hz)
    b_predictions = {seed: load_b_prediction(session_dir, seed) for seed in SEEDS}
    curves = {"f1": [], "a": [], "b": []}
    for frequency in frequencies:
        mask = frequency_hz == frequency
        curves["f1"].append(
            float(np.sqrt(np.mean((context["validation_f1"][mask] - y_true[mask]) ** 2)))
        )
        for branch, predictions in (
            (
                "a",
                [context["control_predictions"][seed]["validation"] for seed in SEEDS],
            ),
            ("b", [b_predictions[seed] for seed in SEEDS]),
        ):
            curves[branch].append(
                median(
                    [
                        float(np.sqrt(np.mean((prediction[mask] - y_true[mask]) ** 2)))
                        for prediction in predictions
                    ]
                )
            )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frequencies, curves["f1"], "--", color="black", label="formal Re F1")
    ax.plot(frequencies, curves["a"], linewidth=2, label="A saved median")
    ax.plot(frequencies, curves["b"], linewidth=2, label="B median")
    for lower, upper in GUARDRAIL_REGIONS:
        ax.axvspan(lower, upper, color="#bbbbbb", alpha=0.08)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Median-across-seeds validation RMSE")
    ax.set_title("F3-D1 frequency-wise validation behavior")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "median_frequency_rmse.png", dpi=180)
    plt.close(fig)


def write_decision_markdown(session_dir: Path, comparison: dict[str, Any]) -> None:
    a = comparison["branch_medians"]["a"]
    b = comparison["branch_medians"]["b"]
    target = comparison["target_region_comparison"]
    status = "PASS" if comparison["stage_passed"] else "STOP / NO PASS"
    lines = [
        "# Re Global SR F3-D1 Decision Report",
        "",
        f"- Stage decision: **{status}**",
        f"- Saved A median validation RMSE: `{a['validation_rmse']:.9f}`",
        f"- B median validation RMSE: `{b['validation_rmse']:.9f}`",
        f"- Saved A median curve-mean RMSE: `{a['curve_mean_rmse']:.9f}`",
        f"- B median curve-mean RMSE: `{b['curve_mean_rmse']:.9f}`",
        f"- Saved A median worst-curve RMSE: `{a['worst_curve_rmse']:.9f}`",
        f"- B median worst-curve RMSE: `{b['worst_curve_rmse']:.9f}`",
        f"- Saved A median 1300–2000 Hz RMSE: `{target['saved_branch_a_median_rmse']:.9f}`",
        f"- B median 1300–2000 Hz RMSE: `{target['branch_b_median_rmse']:.9f}`",
        f"- Test rows accessed: `{comparison['test_rows_accessed']}`",
        "",
        "## Predecessor guardrails",
        "",
    ]
    for row in comparison["predecessor_a_guardrails"]:
        lines.append(
            f"- {int(row['lower_hz'])}–{int(row['upper_hz'])} Hz: "
            f"B median `{row['branch_b_median_rmse']:.9f}`, "
            f"saved A `{row['saved_a_control_median_rmse']:.9f}`, "
            f"passed `{row['passed']}`"
        )
    lines.extend(
        [
            "",
            "This is a small-budget representation decision, not a formal Re model replacement.",
            "A later stage requires a new explicit user command.",
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
    terminal_spec = d1_terminal_spec(context, args)
    acceptance_rules = {
        "full_validation": "B three-seed median RMSE must be strictly below saved A",
        "curve_behavior": "B median curve-mean and worst-curve RMSE must both be strictly below saved A",
        "target_region": "B median 1300-2000 Hz RMSE must be strictly below saved A",
        "guardrails": "B may not worsen more than 10% versus saved A in any declared region, both by median and matched seed",
        "fallbacks": "every B candidate set includes exact saved-A and zero-residual formal-F1 fallbacks",
    }
    preflight = {
        "schema_version": 1,
        "stage": STAGE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session_dir),
        "seeds_predeclared_before_search": list(SEEDS),
        "comparison": {
            "control_a": "saved F3-C branch A; no rerun",
            "candidate_b": "frozen Re F1 plus residual PySR on z_sigma__f3_mid and z_lambda__f3_mid",
            "single_changed_factor": "add z_lambda * g_mid terminal",
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
        "selection": "complete validation RMSE per run with exact saved-A and formal-F1 fallbacks",
        "acceptance_rules_predeclared_before_search": acceptance_rules,
        "guardrail_control": "matched saved F3-C branch A result",
        "test_access_contract": "test features, targets, and predictions remain inaccessible",
        "d1_terminal_spec": terminal_spec,
        "input_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "execute": True,
    }
    dump_json(session_dir / "preflight_manifest.json", preflight)
    dump_json(session_dir / "frozen_d1_terminal_spec.json", terminal_spec)
    np.savez_compressed(
        session_dir / "terminal_values_train_validation.npz",
        train_curve_ids=context["train"]["source_curve_index"],
        train_frequency_hz=context["train"]["frequency_hz"],
        train_g_mid=context["train_g_mid"],
        train_z_lambda=context["train_z_lambda"],
        train_terminal=context["train_lambda_terminal"],
        validation_curve_ids=context["validation"]["source_curve_index"],
        validation_frequency_hz=context["validation"]["frequency_hz"],
        validation_g_mid=context["validation_g_mid"],
        validation_z_lambda=context["validation_z_lambda"],
        validation_terminal=context["validation_lambda_terminal"],
    )
    pd.DataFrame(
        [
            {
                "partition": partition,
                "row_count": int(terminal.size),
                "finite": bool(np.all(np.isfinite(terminal))),
                "minimum": float(np.min(terminal)),
                "maximum": float(np.max(terminal)),
                "mean": float(np.mean(terminal)),
                "std": float(np.std(terminal, ddof=0)),
            }
            for partition, terminal in (
                ("train", context["train_lambda_terminal"]),
                ("validation", context["validation_lambda_terminal"]),
            )
        ]
    ).to_csv(session_dir / "terminal_summary.csv", index=False)
    log(
        "F3-D1 preflight frozen: seeds=42/123/456, saved A versus new B, "
        "budget=20/6/40/24, Plan B, validation-only."
    )

    b_summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = session_dir / "runs" / BRANCH / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        worker_log = logs_dir / f"b_seed_{seed}.log"
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            "--seed",
            str(seed),
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
            "--julia-exe",
            str(args.julia_exe.resolve()),
        ]
        log(f"Starting PySR run: branch=B, seed={seed}.")
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
                f"F3-D1 worker failed: branch=B, seed={seed}, exit={return_code}. "
                f"See {worker_log}"
            )
        summary = load_json(run_dir / "selected_run_summary.json")
        if summary.get("test_rows_accessed") is not False:
            raise ValueError("An F3-D1 worker reported test-row access.")
        b_summaries.append(summary)
        log(
            f"Completed PySR run: branch=B, seed={seed}, "
            f"validation_RMSE={summary['validation_metrics']['rmse']:.9f}."
        )

    log("All three B searches completed; comparing against saved branch A.")
    comparison = aggregate_runs(session_dir, b_summaries, context)
    plot_results(session_dir, b_summaries, context)
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
        "saved_control_a_summaries": [
            context["control_summaries"][seed] for seed in SEEDS
        ],
        "new_branch_b_run_summaries": b_summaries,
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
        f"F3-D1 complete: stage_passed={comparison['stage_passed']}, "
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
