"""Run the validation-only Re Global SR F3-C three-seed PySR pilot."""

from __future__ import annotations

import argparse
import hashlib
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
if str(SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURROGATE_ROOT))

from evaluate_global_symbolic_candidates import (  # noqa: E402
    make_pysr_variable_names,
    predict_expression,
)
from f3_local_terminals import (  # noqa: E402
    apply_f3_local_terminals,
    validate_f3_local_terminal_spec,
)
from run_f3a_residual_diagnosis import (  # noqa: E402
    DEFAULT_DATASET_FILE,
    DEFAULT_F1_TRAINING_DIR,
    DEFAULT_SPLIT_FILE,
    F3_ARTIFACT_ROOT,
    GUARDRAIL_MAX_RELATIVE_WORSENING,
    GUARDRAIL_REGIONS,
    REPORTING_REGIONS,
    TARGET_REGION,
    curve_metrics,
    load_partition_rows,
    prepare_f1_features,
    regional_metrics,
    regression_metrics,
    sha256_file,
)
from shared_split_utils import load_shared_split  # noqa: E402


SEEDS = (42, 123, 456)
BRANCHES = ("c0", "a")
NITERATIONS = 20
POPULATIONS = 6
POPULATION_SIZE = 40
MAX_COMPLEXITY = 24
BATCH_SIZE = 4096
BINARY_OPERATORS = ("+", "-", "*", "/")
UNARY_OPERATORS = ("log", "sqrt")
DEFAULT_F3A_DIR = F3_ARTIFACT_ROOT / "F3-A" / "20260816T153654"
DEFAULT_F3B_DIR = F3_ARTIFACT_ROOT / "F3-B" / "20260816T155822"
DEFAULT_JULIA_EXE = (
    SURROGATE_ROOT
    / "segmented_symbolic_regression"
    / ".venv_py311"
    / "julia_env"
    / "pyjuliapkg"
    / "install"
    / "bin"
    / "julia.exe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--f1-training-dir", type=Path, default=DEFAULT_F1_TRAINING_DIR)
    parser.add_argument("--f3a-dir", type=Path, default=DEFAULT_F3A_DIR)
    parser.add_argument("--f3b-dir", type=Path, default=DEFAULT_F3B_DIR)
    parser.add_argument("--julia-exe", type=Path, default=DEFAULT_JULIA_EXE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact session directory; it must remain under artifacts/F3.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--branch", choices=BRANCHES, default=None, help=argparse.SUPPRESS)
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
        output_dir = unique_path(root / "F3-C" / timestamp)
    else:
        output_dir = requested.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError(f"F3-C artifacts must stay under {root}: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty F3-C directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_paths(args: argparse.Namespace) -> dict[str, Path]:
    f3a_dir = args.f3a_dir.resolve()
    f3b_dir = args.f3b_dir.resolve()
    f1_training_dir = args.f1_training_dir.resolve()
    paths = {
        "dataset": args.dataset_file.resolve(),
        "split": args.split_file.resolve(),
        "f1_metadata": f1_training_dir / "training_metadata.json",
        "f3a_decision": f3a_dir / "decision_report.json",
        "f3a_predictions": f3a_dir / "f1_and_selected_sigma_predictions.npz",
        "f3b_decision": f3b_dir / "decision_report.json",
        "f3b_spec": f3b_dir / "frozen_sigma_local_spec.json",
        "f3b_terminals": f3b_dir / "terminal_values_train_validation.npz",
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"F3-C required inputs are missing: {missing}")
    return paths


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    paths = required_paths(args)
    split = load_shared_split(
        paths["split"], expected_dataset_run="run1", expected_curve_count=1000
    )
    f1_metadata = load_json(paths["f1_metadata"])
    f3a_decision = load_json(paths["f3a_decision"])
    f3b_decision = load_json(paths["f3b_decision"])
    f3b_spec = load_json(paths["f3b_spec"])
    if not f3a_decision.get("stage_passed") or not f3b_decision.get("stage_passed"):
        raise ValueError("Both F3-A and F3-B must pass before F3-C.")
    if f3a_decision.get("selected_sigma_candidate") != "sigma_bandpass_1250_2100":
        raise ValueError("F3-C received an unexpected F3-A representation.")
    if f3b_decision.get("frozen_terminal_name") != "z_sigma__f3_mid":
        raise ValueError("F3-C received an unexpected F3-B terminal.")
    if f1_metadata.get("shared_split_hash") != split["split_hash"]:
        raise ValueError("Re F1 metadata and F3-C split hashes differ.")
    if f3b_spec.get("shared_split_hash") != split["split_hash"]:
        raise ValueError("F3-B specification and F3-C split hashes differ.")

    train_ids = np.asarray(split["train_curve_indices"], dtype=int)
    validation_ids = np.asarray(split["validation_curve_indices"], dtype=int)
    test_ids = np.asarray(split["test_curve_indices"], dtype=int)
    if np.intersect1d(np.concatenate([train_ids, validation_ids]), test_ids).size:
        raise ValueError("Allowed F3-C curves overlap the forbidden test partition.")
    train = prepare_f1_features(
        load_partition_rows(paths["dataset"], train_ids, 1000), f1_metadata
    )
    validation = prepare_f1_features(
        load_partition_rows(paths["dataset"], validation_ids, 1000), f1_metadata
    )
    base_names = list(train["base_feature_names"])
    f3b_spec = validate_f3_local_terminal_spec(f3b_spec, base_names)
    train_f3, f3_feature_names = apply_f3_local_terminals(
        train["X_base"], base_names, f3b_spec
    )
    validation_f3, validation_f3_names = apply_f3_local_terminals(
        validation["X_base"], base_names, f3b_spec
    )
    if f3_feature_names != validation_f3_names:
        raise ValueError("F3-B feature order differs between train and validation.")

    with np.load(paths["f3a_predictions"]) as values:
        if not np.array_equal(values["train_curve_ids"], train["source_curve_index"]):
            raise ValueError("F3-A training prediction rows do not align with F3-C.")
        if not np.array_equal(
            values["validation_curve_ids"], validation["source_curve_index"]
        ):
            raise ValueError("F3-A validation prediction rows do not align with F3-C.")
        if not np.array_equal(values["train_teacher"], train["y"]):
            raise ValueError("F3-A/F3-C training targets differ.")
        if not np.array_equal(values["validation_teacher"], validation["y"]):
            raise ValueError("F3-A/F3-C validation targets differ.")
        train_f1 = values["train_f1_prediction"].copy()
        validation_f1 = values["validation_f1_prediction"].copy()
    with np.load(paths["f3b_terminals"]) as values:
        if not np.array_equal(values["train_terminal"], train_f3[:, -1]):
            raise ValueError("F3-B/F3-C training terminal replay differs.")
        if not np.array_equal(values["validation_terminal"], validation_f3[:, -1]):
            raise ValueError("F3-B/F3-C validation terminal replay differs.")

    return {
        "paths": paths,
        "split": split,
        "f1_metadata": f1_metadata,
        "f3b_spec": f3b_spec,
        "train": train,
        "validation": validation,
        "train_f1": train_f1,
        "validation_f1": validation_f1,
        "train_f3": train_f3,
        "validation_f3": validation_f3,
        "f3_feature_names": f3_feature_names,
        "test_curve_count": int(test_ids.size),
    }


def build_model(seed: int, run_dir: Path, run_id: str) -> Any:
    from pysr import PySRRegressor

    return PySRRegressor(
        niterations=NITERATIONS,
        populations=POPULATIONS,
        population_size=POPULATION_SIZE,
        binary_operators=list(BINARY_OPERATORS),
        unary_operators=list(UNARY_OPERATORS),
        maxsize=MAX_COMPLEXITY,
        model_selection="best",
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        batching=True,
        batch_size=BATCH_SIZE,
        turbo=True,
        progress=False,
        verbosity=1,
        temp_equation_file=False,
        output_directory=(run_dir / "pysr_runs").as_posix(),
        run_id=run_id,
    )


def branch_data(context: dict[str, Any], branch: str) -> dict[str, Any]:
    if branch == "c0":
        return {
            "X_train": context["train"]["X_f1"],
            "X_validation": context["validation"]["X_f1"],
            "y_search_train": context["train"]["y"],
            "variable_names": make_pysr_variable_names(
                context["train"]["f1_feature_names"]
            ),
            "feature_names": list(context["train"]["f1_feature_names"]),
            "search_target": "R_real",
            "final_offset_train": np.zeros_like(context["train"]["y"]),
            "final_offset_validation": np.zeros_like(context["validation"]["y"]),
            "explicit_zero_fallback": False,
        }
    if branch == "a":
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
    raise ValueError(f"Unknown F3-C branch: {branch}")


def score_candidates(
    equations: pd.DataFrame,
    branch_view: dict[str, Any],
    context: dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    validation_predictions: dict[int, np.ndarray] = {}
    if branch_view["explicit_zero_fallback"]:
        zero_prediction = branch_view["final_offset_validation"].copy()
        zero_metrics = regression_metrics(context["validation"]["y"], zero_prediction)
        rows.append(
            {
                "candidate_index": 0,
                "candidate_source": "explicit_zero_residual_fallback",
                "complexity": 0,
                "pysr_loss": math.nan,
                "pysr_score": math.nan,
                "equation": "0.0",
                "expression_for_eval": "0.0",
                "eval_status": "ok",
                **zero_metrics,
            }
        )
        validation_predictions[0] = zero_prediction

    expression_column = "sympy_format" if "sympy_format" in equations else "equation"
    for row_index, equation_row in equations.reset_index(drop=True).iterrows():
        candidate_index = row_index + 1
        expression = str(equation_row[expression_column])
        try:
            searched_prediction = predict_expression(
                expression,
                branch_view["X_validation"],
                branch_view["variable_names"],
            )
            final_prediction = branch_view["final_offset_validation"] + searched_prediction
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
    branch_view: dict[str, Any],
    context: dict[str, Any],
    validation_prediction: np.ndarray,
) -> dict[str, Any]:
    if int(selected["candidate_index"]) == 0:
        train_prediction = branch_view["final_offset_train"].copy()
    else:
        searched_train = predict_expression(
            str(selected["expression_for_eval"]),
            branch_view["X_train"],
            branch_view["variable_names"],
        )
        train_prediction = branch_view["final_offset_train"] + searched_train
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
    if args.branch is None or args.seed is None or args.run_dir is None:
        raise ValueError("Worker mode requires --branch, --seed, and --run-dir.")
    if args.seed not in SEEDS:
        raise ValueError(f"Worker seed is outside the frozen F3-C seeds: {args.seed}")
    run_dir = args.run_dir.resolve()
    session_dir = run_dir.parents[2]
    if not run_dir.is_relative_to(session_dir / "runs"):
        raise ValueError("Worker output must remain inside the F3-C session runs directory.")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JULIA_NUM_THREADS"] = "1"
    julia_exe = args.julia_exe.resolve()
    if not julia_exe.exists():
        raise FileNotFoundError(f"Julia executable not found: {julia_exe}")
    os.environ.setdefault("PYTHON_JULIAPKG_EXE", str(julia_exe))

    print(
        f"Worker start: branch={args.branch}, seed={args.seed}, "
        f"budget={NITERATIONS}/{POPULATIONS}/{POPULATION_SIZE}/{MAX_COMPLEXITY}",
        flush=True,
    )
    context = load_context(args)
    view = branch_data(context, args.branch)
    run_id = f"f3c_{args.branch}_s{args.seed}"
    model = build_model(args.seed, run_dir, run_id)
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
    candidate_metrics, predictions = score_candidates(equations, view, context)
    candidate_metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)
    valid = candidate_metrics[candidate_metrics["eval_status"] == "ok"].copy()
    if valid.empty:
        raise RuntimeError("No valid F3-C candidate predictions were produced.")
    valid = valid.sort_values(["rmse", "complexity"], ascending=[True, True])
    selected = valid.iloc[0]
    selected_index = int(selected["candidate_index"])
    evaluated = evaluate_selected(
        selected, view, context, predictions[selected_index]
    )
    regional_frame = pd.DataFrame(evaluated["validation_regional_metrics"])
    regional_frame.to_csv(run_dir / "selected_validation_regional_metrics.csv", index=False)
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
        "stage": "F3-C",
        "branch": args.branch,
        "seed": args.seed,
        "search_target": view["search_target"],
        "input_feature_names": view["feature_names"],
        "pysr_variable_names": view["variable_names"],
        "explicit_zero_fallback": view["explicit_zero_fallback"],
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
        "search_contract": {
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
        "equations_sha256": file_sha256(equations_file),
    }
    dump_json(run_dir / "selected_run_summary.json", summary)
    print(
        f"Worker complete: branch={args.branch}, seed={args.seed}, "
        f"validation_RMSE={evaluated['validation_metrics']['rmse']:.9f}, "
        f"selected={selected_index}, complexity={int(selected['complexity'])}, "
        f"fit_seconds={fit_seconds:.1f}",
        flush=True,
    )


def region_map(rows: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (float(row["lower_hz"]), float(row["upper_hz"])): row for row in rows
    }


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def aggregate_runs(
    session_dir: Path, run_summaries: list[dict[str, Any]], context: dict[str, Any]
) -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    regional_rows: list[dict[str, Any]] = []
    for summary in run_summaries:
        row = {
            "branch": summary["branch"],
            "seed": summary["seed"],
            "selected_candidate_index": summary["selected_candidate_index"],
            "selected_candidate_source": summary["selected_candidate_source"],
            "selected_complexity": summary["selected_complexity"],
            "fit_wall_time_seconds": summary["fit_wall_time_seconds"],
            **summary["validation_metrics"],
            **summary["validation_curve_summary"],
            "selected_equation": summary["selected_equation"],
        }
        seed_rows.append(row)
        for region in summary["validation_regional_metrics"]:
            regional_rows.append(
                {
                    "branch": summary["branch"],
                    "seed": summary["seed"],
                    **region,
                }
            )
    pd.DataFrame(seed_rows).to_csv(session_dir / "seed_metrics.csv", index=False)
    pd.DataFrame(regional_rows).to_csv(
        session_dir / "seed_regional_metrics.csv", index=False
    )

    branch_medians: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        branch_rows = [row for row in seed_rows if row["branch"] == branch]
        branch_medians[branch] = {
            "validation_rmse": median([float(row["rmse"]) for row in branch_rows]),
            "validation_mae": median([float(row["mae"]) for row in branch_rows]),
            "validation_r2": median([float(row["r2"]) for row in branch_rows]),
            "curve_mean_rmse": median(
                [float(row["curve_mean_rmse"]) for row in branch_rows]
            ),
            "worst_curve_rmse": median(
                [float(row["worst_curve_rmse"]) for row in branch_rows]
            ),
            "fit_wall_time_seconds": median(
                [float(row["fit_wall_time_seconds"]) for row in branch_rows]
            ),
        }
        for bounds in (*REPORTING_REGIONS, TARGET_REGION):
            matching = [
                row
                for row in regional_rows
                if row["branch"] == branch
                and float(row["lower_hz"]) == bounds[0]
                and float(row["upper_hz"]) == bounds[1]
            ]
            branch_medians[branch].setdefault("regional_metrics", []).append(
                {
                    "lower_hz": bounds[0],
                    "upper_hz": bounds[1],
                    "median_rmse": median([float(row["rmse"]) for row in matching]),
                    "median_mae": median([float(row["mae"]) for row in matching]),
                    "median_bias": median([float(row["bias"]) for row in matching]),
                }
            )

    f1_metrics = regression_metrics(
        context["validation"]["y"], context["validation_f1"]
    )
    f1_curve_summary, _ = curve_metrics(
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
    f1_reference = {
        "validation_metrics": f1_metrics,
        "validation_curve_summary": f1_curve_summary,
        "validation_regional_metrics": f1_regions,
    }

    a_regions = {
        (row["lower_hz"], row["upper_hz"]): row
        for row in branch_medians["a"]["regional_metrics"]
    }
    c0_regions = {
        (row["lower_hz"], row["upper_hz"]): row
        for row in branch_medians["c0"]["regional_metrics"]
    }
    f1_region_lookup = region_map(f1_regions)
    guardrails: list[dict[str, Any]] = []
    for bounds in GUARDRAIL_REGIONS:
        a_rmse = float(a_regions[bounds]["median_rmse"])
        control_rmse = float(f1_region_lookup[bounds]["rmse"])
        relative_change = a_rmse / control_rmse - 1.0
        guardrails.append(
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "formal_f1_control_rmse": control_rmse,
                "branch_a_median_rmse": a_rmse,
                "relative_change": relative_change,
                "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                "passed": relative_change <= GUARDRAIL_MAX_RELATIVE_WORSENING,
            }
        )
    per_seed_guardrails: list[dict[str, Any]] = []
    for seed in SEEDS:
        for bounds in GUARDRAIL_REGIONS:
            matching = next(
                row
                for row in regional_rows
                if row["branch"] == "a"
                and int(row["seed"]) == seed
                and float(row["lower_hz"]) == bounds[0]
                and float(row["upper_hz"]) == bounds[1]
            )
            candidate_rmse = float(matching["rmse"])
            control_rmse = float(f1_region_lookup[bounds]["rmse"])
            relative_change = candidate_rmse / control_rmse - 1.0
            per_seed_guardrails.append(
                {
                    "seed": seed,
                    "lower_hz": bounds[0],
                    "upper_hz": bounds[1],
                    "formal_f1_control_rmse": control_rmse,
                    "branch_a_rmse": candidate_rmse,
                    "relative_change": relative_change,
                    "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                    "passed": relative_change <= GUARDRAIL_MAX_RELATIVE_WORSENING,
                }
            )
    target_a = float(a_regions[TARGET_REGION]["median_rmse"])
    target_c0 = float(c0_regions[TARGET_REGION]["median_rmse"])
    target_f1 = float(f1_region_lookup[TARGET_REGION]["rmse"])
    checks = {
        "all_six_runs_completed": len(run_summaries) == 6,
        "all_runs_validation_only": all(
            summary["test_rows_accessed"] is False for summary in run_summaries
        ),
        "matched_seed_sets": (
            {summary["seed"] for summary in run_summaries if summary["branch"] == "c0"}
            == {summary["seed"] for summary in run_summaries if summary["branch"] == "a"}
            == set(SEEDS)
        ),
        "a_beats_c0_median_full_validation_rmse": (
            branch_medians["a"]["validation_rmse"]
            < branch_medians["c0"]["validation_rmse"]
        ),
        "a_preserves_exact_f1_fallback_in_every_seed": all(
            summary["validation_metrics"]["rmse"] <= f1_metrics["rmse"]
            for summary in run_summaries
            if summary["branch"] == "a"
        ),
        "a_improves_target_region_vs_c0_median": target_a < target_c0,
        "a_improves_target_region_vs_formal_f1": target_a < target_f1,
        "all_formal_f1_regional_guardrails_pass": all(
            row["passed"] for row in guardrails
        ),
        "all_a_seed_formal_f1_regional_guardrails_pass": all(
            row["passed"] for row in per_seed_guardrails
        ),
    }
    stage_passed = all(checks.values())
    comparison = {
        "stage": "F3-C",
        "stage_passed": stage_passed,
        "decision": (
            "advance_branch_A_representation_to_next_explicit_stage"
            if stage_passed
            else "stop_and_retain_formal_Re_F1"
        ),
        "formal_model_replacement_decision": False,
        "small_budget_models_are_not_formal_replacements": True,
        "seeds": list(SEEDS),
        "branch_medians": branch_medians,
        "formal_f1_validation_reference": f1_reference,
        "target_region_comparison": {
            "region_hz": list(TARGET_REGION),
            "branch_a_median_rmse": target_a,
            "branch_c0_median_rmse": target_c0,
            "formal_f1_rmse": target_f1,
        },
        "formal_f1_guardrails": guardrails,
        "per_seed_formal_f1_guardrails": per_seed_guardrails,
        "checks": checks,
        "test_rows_accessed": False,
        "next_stage_automatically_authorized": False,
        "next_command_if_user_accepts": "执行 F3-D1",
    }
    dump_json(session_dir / "comparison_decision.json", comparison)
    median_rows: list[dict[str, Any]] = []
    for branch, values in branch_medians.items():
        median_rows.append(
            {
                "branch": branch,
                **{key: value for key, value in values.items() if key != "regional_metrics"},
            }
        )
    pd.DataFrame(median_rows).to_csv(session_dir / "median_metrics.csv", index=False)
    pd.DataFrame(guardrails).to_csv(
        session_dir / "formal_f1_guardrail_comparison.csv", index=False
    )
    pd.DataFrame(per_seed_guardrails).to_csv(
        session_dir / "per_seed_formal_f1_guardrail_comparison.csv", index=False
    )
    return comparison


def load_selected_prediction(session_dir: Path, branch: str, seed: int) -> np.ndarray:
    path = (
        session_dir
        / "runs"
        / branch
        / f"seed_{seed}"
        / "selected_predictions_train_validation.npz"
    )
    with np.load(path) as values:
        return values["validation_prediction"].copy()


def plot_seed_rmse(
    session_dir: Path,
    run_summaries: list[dict[str, Any]],
    f1_rmse: float,
) -> None:
    figures_dir = session_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(SEEDS))
    width = 0.36
    c0 = [
        next(
            summary["validation_metrics"]["rmse"]
            for summary in run_summaries
            if summary["branch"] == "c0" and summary["seed"] == seed
        )
        for seed in SEEDS
    ]
    a = [
        next(
            summary["validation_metrics"]["rmse"]
            for summary in run_summaries
            if summary["branch"] == "a" and summary["seed"] == seed
        )
        for seed in SEEDS
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, c0, width, label="C0 direct small-budget search")
    ax.bar(x + width / 2, a, width, label="A fixed F1 + residual PySR")
    ax.axhline(f1_rmse, color="black", linestyle="--", label="formal Re F1")
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Complete validation RMSE")
    ax.set_title("F3-C matched three-seed validation comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "three_seed_validation_rmse.png", dpi=180)
    plt.close(fig)


def plot_frequency_median(
    session_dir: Path,
    context: dict[str, Any],
) -> None:
    frequency_hz = context["validation"]["frequency_hz"]
    y_true = context["validation"]["y"]
    frequencies = np.unique(frequency_hz)
    branch_curves: dict[str, list[float]] = {"c0": [], "a": [], "f1": []}
    predictions = {
        branch: [load_selected_prediction(session_dir, branch, seed) for seed in SEEDS]
        for branch in BRANCHES
    }
    for frequency in frequencies:
        mask = frequency_hz == frequency
        branch_curves["f1"].append(
            float(math.sqrt(np.mean((context["validation_f1"][mask] - y_true[mask]) ** 2)))
        )
        for branch in BRANCHES:
            per_seed = [
                float(math.sqrt(np.mean((prediction[mask] - y_true[mask]) ** 2)))
                for prediction in predictions[branch]
            ]
            branch_curves[branch].append(median(per_seed))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frequencies, branch_curves["f1"], linestyle="--", color="black", label="formal Re F1")
    ax.plot(frequencies, branch_curves["c0"], linewidth=2, label="C0 median")
    ax.plot(frequencies, branch_curves["a"], linewidth=2, label="A median")
    for lower, upper in GUARDRAIL_REGIONS:
        ax.axvspan(lower, upper, color="#bbbbbb", alpha=0.08)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Median-across-seeds validation RMSE")
    ax.set_title("F3-C frequency-wise validation behavior")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(session_dir / "figures" / "median_frequency_rmse.png", dpi=180)
    plt.close(fig)


def write_decision_markdown(session_dir: Path, comparison: dict[str, Any]) -> None:
    c0 = comparison["branch_medians"]["c0"]
    a = comparison["branch_medians"]["a"]
    target = comparison["target_region_comparison"]
    status = "PASS" if comparison["stage_passed"] else "STOP / NO PASS"
    lines = [
        "# Re Global SR F3-C Decision Report",
        "",
        f"- Stage decision: **{status}**",
        f"- C0 median validation RMSE: `{c0['validation_rmse']:.9f}`",
        f"- A median validation RMSE: `{a['validation_rmse']:.9f}`",
        f"- Formal Re F1 validation RMSE: `{comparison['formal_f1_validation_reference']['validation_metrics']['rmse']:.9f}`",
        f"- A median 1300–2000 Hz RMSE: `{target['branch_a_median_rmse']:.9f}`",
        f"- C0 median 1300–2000 Hz RMSE: `{target['branch_c0_median_rmse']:.9f}`",
        f"- Formal Re F1 1300–2000 Hz RMSE: `{target['formal_f1_rmse']:.9f}`",
        f"- Test rows accessed: `{comparison['test_rows_accessed']}`",
        "",
        "## Guardrails",
        "",
    ]
    for row in comparison["formal_f1_guardrails"]:
        lines.append(
            f"- {int(row['lower_hz'])}–{int(row['upper_hz'])} Hz: "
            f"A median `{row['branch_a_median_rmse']:.9f}`, "
            f"F1 `{row['formal_f1_control_rmse']:.9f}`, passed `{row['passed']}`"
        )
    lines.extend(
        [
            "",
            "This is a small-budget representation decision, not a formal Re model replacement.",
            "A later stage requires a new explicit user command.",
            "",
        ]
    )
    with (session_dir / "decision_report.md").open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


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
    preflight = {
        "schema_version": 1,
        "stage": "F3-C",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session_dir),
        "seeds_predeclared_before_search": list(SEEDS),
        "branches": {
            "c0": "original Re F1 11-feature representation trained directly at the small budget",
            "a": "frozen Re F1 plus residual PySR using only z_sigma__f3_mid",
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
        "selection": "complete validation RMSE independently per run; zero residual explicitly available to A",
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
        "F3-C preflight frozen: seeds=42/123/456, branches=C0/A, "
        "budget=20/6/40/24, Plan B, validation-only."
    )

    run_summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        for branch in BRANCHES:
            run_dir = session_dir / "runs" / branch / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            worker_log = logs_dir / f"{branch}_seed_{seed}.log"
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                "--branch",
                branch,
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
                "--julia-exe",
                str(args.julia_exe.resolve()),
            ]
            log(f"Starting PySR run: branch={branch}, seed={seed}.")
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
                    f"F3-C worker failed: branch={branch}, seed={seed}, exit={return_code}. "
                    f"See {worker_log}"
                )
            summary = load_json(run_dir / "selected_run_summary.json")
            if summary.get("test_rows_accessed") is not False:
                raise ValueError("An F3-C worker reported test-row access.")
            run_summaries.append(summary)
            log(
                f"Completed PySR run: branch={branch}, seed={seed}, "
                f"validation_RMSE={summary['validation_metrics']['rmse']:.9f}."
            )

    log("All six PySR searches completed; computing the frozen three-seed decision.")
    context = load_context(args)
    comparison = aggregate_runs(session_dir, run_summaries, context)
    plot_seed_rmse(
        session_dir,
        run_summaries,
        comparison["formal_f1_validation_reference"]["validation_metrics"]["rmse"],
    )
    plot_frequency_median(session_dir, context)
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
        "run_summaries": run_summaries,
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
        f"F3-C complete: stage_passed={comparison['stage_passed']}, "
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
