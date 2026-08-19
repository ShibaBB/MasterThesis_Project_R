"""Run validation-only Re Global SR F3-E1 nested log/sqrt constraints."""

from __future__ import annotations

import argparse
import ast
import json
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
for import_path in (SCRIPT_DIR, SURROGATE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_f3c_residual_pysr_pilot as f3c  # noqa: E402
import run_f3d1_lambda_local_pysr as d1  # noqa: E402
from run_f3a_residual_diagnosis import (  # noqa: E402
    F3_ARTIFACT_ROOT,
    GUARDRAIL_MAX_RELATIVE_WORSENING,
    GUARDRAIL_REGIONS,
    REPORTING_REGIONS,
    TARGET_REGION,
    sha256_file,
)


STAGE = "F3-E1"
BRANCH = "e1"
SEEDS = f3c.SEEDS
NITERATIONS = 20
POPULATIONS = 6
POPULATION_SIZE = 40
MAX_COMPLEXITY = 24
BATCH_SIZE = 4096
BINARY_OPERATORS = ("+", "-", "*", "/")
UNARY_OPERATORS = ("log", "sqrt")
NESTED_CONSTRAINTS = {
    "log": {"log": 0, "sqrt": 0},
    "sqrt": {"log": 0, "sqrt": 0},
}
CONSTRAINED_FUNCTIONS = frozenset(NESTED_CONSTRAINTS)
DEFAULT_F3C_DIR = d1.DEFAULT_F3C_DIR


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


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def _call_name(node: ast.Call) -> str | None:
    return node.func.id if isinstance(node.func, ast.Name) else None


def expression_obeys_nested_constraints(expression: str) -> bool:
    """Return whether log/sqrt never occurs inside another log/sqrt call."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in CONSTRAINED_FUNCTIONS:
            continue
        for argument in node.args:
            for descendant in ast.walk(argument):
                if (
                    isinstance(descendant, ast.Call)
                    and _call_name(descendant) in CONSTRAINED_FUNCTIONS
                ):
                    return False
    return True


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    context = d1.load_context(args)
    for seed, summary in context["control_summaries"].items():
        if not expression_obeys_nested_constraints(
            str(summary["selected_equation"])
        ):
            raise ValueError(
                f"Saved branch A seed {seed} is not a valid constrained fallback."
            )
    return context


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
    }


def build_model(seed: int, run_dir: Path, run_id: str) -> Any:
    from pysr import PySRRegressor

    return PySRRegressor(
        niterations=NITERATIONS,
        populations=POPULATIONS,
        population_size=POPULATION_SIZE,
        binary_operators=list(BINARY_OPERATORS),
        unary_operators=list(UNARY_OPERATORS),
        nested_constraints=NESTED_CONSTRAINTS,
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


def worker_main(args: argparse.Namespace) -> None:
    if args.seed is None or args.run_dir is None:
        raise ValueError("Worker mode requires --seed and --run-dir.")
    if args.seed not in SEEDS:
        raise ValueError(f"Worker seed is outside the frozen E1 seeds: {args.seed}")
    run_dir = args.run_dir.resolve()
    session_dir = run_dir.parents[2]
    if not run_dir.is_relative_to(session_dir / "runs"):
        raise ValueError("Worker output must remain inside the E1 session runs directory.")
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
    model = build_model(args.seed, run_dir, f"f3e1_s{args.seed}")
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

    equation_compliance = equations["equation"].map(
        lambda value: expression_obeys_nested_constraints(str(value))
    )
    if not bool(equation_compliance.all()):
        invalid = equations.loc[~equation_compliance, "equation"].tolist()
        raise ValueError(f"PySR emitted equations outside the E1 constraint: {invalid}")
    candidate_metrics, predictions = d1.score_candidates(
        equations, view, context, args.seed
    )
    candidate_metrics["nested_constraint_compliant"] = candidate_metrics[
        "equation"
    ].map(lambda value: expression_obeys_nested_constraints(str(value)))
    candidate_metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)
    valid = candidate_metrics[
        (candidate_metrics["eval_status"] == "ok")
        & candidate_metrics["nested_constraint_compliant"]
    ].copy()
    if valid.empty:
        raise RuntimeError("No valid E1 candidate predictions were produced.")
    valid = valid.sort_values(["rmse", "complexity"], ascending=[True, True])
    selected = valid.iloc[0]
    selected_index = int(selected["candidate_index"])
    evaluated = d1.evaluate_selected(
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
        "single_changed_search_factor": "nested_constraints",
        "nested_constraints": NESTED_CONSTRAINTS,
        "all_hall_of_fame_equations_constraint_compliant": True,
        "selected_equation_constraint_compliant": bool(
            selected["nested_constraint_compliant"]
        ),
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


def aggregate_runs(
    session_dir: Path,
    e1_summaries: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    a_summaries = [context["control_summaries"][seed] for seed in SEEDS]
    all_summaries = [("a", "saved_F3-C", row) for row in a_summaries] + [
        ("e1", "new_F3-E1", row) for row in e1_summaries
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

    medians = {
        "a": d1.branch_median(a_summaries),
        "e1": d1.branch_median(e1_summaries),
    }
    a_regions = d1.region_map(medians["a"]["regional_metrics"])
    e1_regions = d1.region_map(medians["e1"]["regional_metrics"])
    median_guardrails: list[dict[str, Any]] = []
    for bounds in GUARDRAIL_REGIONS:
        a_rmse = float(a_regions[bounds]["median_rmse"])
        e1_rmse = float(e1_regions[bounds]["median_rmse"])
        change = e1_rmse / a_rmse - 1.0
        median_guardrails.append(
            {
                "lower_hz": bounds[0],
                "upper_hz": bounds[1],
                "saved_a_control_median_rmse": a_rmse,
                "branch_e1_median_rmse": e1_rmse,
                "relative_change": change,
                "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                "passed": change <= GUARDRAIL_MAX_RELATIVE_WORSENING,
            }
        )
    per_seed_guardrails: list[dict[str, Any]] = []
    for seed in SEEDS:
        a_summary = context["control_summaries"][seed]
        e1_summary = next(row for row in e1_summaries if int(row["seed"]) == seed)
        a_lookup = d1.region_map(a_summary["validation_regional_metrics"])
        e1_lookup = d1.region_map(e1_summary["validation_regional_metrics"])
        for bounds in GUARDRAIL_REGIONS:
            a_rmse = float(a_lookup[bounds]["rmse"])
            e1_rmse = float(e1_lookup[bounds]["rmse"])
            change = e1_rmse / a_rmse - 1.0
            per_seed_guardrails.append(
                {
                    "seed": seed,
                    "lower_hz": bounds[0],
                    "upper_hz": bounds[1],
                    "saved_a_control_rmse": a_rmse,
                    "branch_e1_rmse": e1_rmse,
                    "relative_change": change,
                    "maximum_allowed_relative_worsening": GUARDRAIL_MAX_RELATIVE_WORSENING,
                    "passed": change <= GUARDRAIL_MAX_RELATIVE_WORSENING,
                }
            )

    target_a = float(a_regions[TARGET_REGION]["median_rmse"])
    target_e1 = float(e1_regions[TARGET_REGION]["median_rmse"])
    formal = context["f3c_decision"]["formal_f1_validation_reference"]
    formal_target = float(
        d1.region_map(formal["validation_regional_metrics"])[TARGET_REGION]["rmse"]
    )
    checks = {
        "all_three_e1_runs_completed": len(e1_summaries) == 3,
        "all_runs_validation_only": all(
            row.get("test_rows_accessed") is False for row in e1_summaries
        ),
        "matched_predeclared_seed_sets": (
            {int(row["seed"]) for row in e1_summaries}
            == set(context["control_summaries"])
            == set(SEEDS)
        ),
        "only_nested_constraints_changed": all(
            row.get("single_changed_search_factor") == "nested_constraints"
            for row in e1_summaries
        ),
        "all_equations_obey_nested_constraints": all(
            row.get("all_hall_of_fame_equations_constraint_compliant") is True
            and row.get("selected_equation_constraint_compliant") is True
            for row in e1_summaries
        ),
        "all_e1_runs_preserve_exact_f1_and_saved_a_fallbacks": all(
            row.get("explicit_zero_f1_fallback") is True
            and row.get("explicit_saved_a_predecessor_fallback") is True
            and row["validation_metrics"]["rmse"]
            <= context["control_summaries"][int(row["seed"])]["validation_metrics"]["rmse"]
            for row in e1_summaries
        ),
        "e1_beats_saved_a_median_full_validation_rmse": (
            medians["e1"]["validation_rmse"] < medians["a"]["validation_rmse"]
        ),
        "e1_improves_median_curve_mean_rmse": (
            medians["e1"]["curve_mean_rmse"] < medians["a"]["curve_mean_rmse"]
        ),
        "e1_improves_median_worst_curve_rmse": (
            medians["e1"]["worst_curve_rmse"] < medians["a"]["worst_curve_rmse"]
        ),
        "e1_improves_target_region_vs_saved_a": target_e1 < target_a,
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
            "accept_nested_constraints_as_frozen_search_configuration"
            if stage_passed
            else "retain_original_unconstrained_search_configuration"
        ),
        "formal_model_replacement_decision": False,
        "small_budget_models_are_not_formal_replacements": True,
        "seeds": list(SEEDS),
        "single_changed_search_factor": "nested_constraints",
        "nested_constraints": NESTED_CONSTRAINTS,
        "branch_medians": medians,
        "formal_f1_validation_reference": formal,
        "target_region_comparison": {
            "region_hz": list(TARGET_REGION),
            "saved_branch_a_median_rmse": target_a,
            "branch_e1_median_rmse": target_e1,
            "formal_f1_rmse": formal_target,
        },
        "predecessor_a_guardrails": median_guardrails,
        "per_seed_predecessor_a_guardrails": per_seed_guardrails,
        "checks": checks,
        "test_rows_accessed": False,
        "next_stage_automatically_authorized": False,
        "next_command_if_user_wants_next_independent_adjustment": "执行 F3-E2",
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


def load_e1_prediction(session_dir: Path, seed: int) -> np.ndarray:
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
    e1_summaries: list[dict[str, Any]],
    context: dict[str, Any],
) -> None:
    figures = session_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    formal_rmse = float(
        context["f3c_decision"]["formal_f1_validation_reference"]["validation_metrics"]["rmse"]
    )
    x = np.arange(len(SEEDS))
    width = 0.36
    a_rmse = [
        context["control_summaries"][seed]["validation_metrics"]["rmse"]
        for seed in SEEDS
    ]
    e1_rmse = [
        next(
            row["validation_metrics"]["rmse"]
            for row in e1_summaries
            if row["seed"] == seed
        )
        for seed in SEEDS
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, a_rmse, width, label="A saved unconstrained control")
    ax.bar(x + width / 2, e1_rmse, width, label="E1 constrained log/sqrt nesting")
    ax.axhline(formal_rmse, color="black", linestyle="--", label="formal Re F1")
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Complete validation RMSE")
    ax.set_title("F3-E1 matched three-seed validation comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "three_seed_validation_rmse.png", dpi=180)
    plt.close(fig)

    frequency_hz = context["validation"]["frequency_hz"]
    y_true = context["validation"]["y"]
    frequencies = np.unique(frequency_hz)
    e1_predictions = {seed: load_e1_prediction(session_dir, seed) for seed in SEEDS}
    curves = {"f1": [], "a": [], "e1": []}
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
            ("e1", [e1_predictions[seed] for seed in SEEDS]),
        ):
            curves[branch].append(
                d1.median(
                    [
                        float(np.sqrt(np.mean((prediction[mask] - y_true[mask]) ** 2)))
                        for prediction in predictions
                    ]
                )
            )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frequencies, curves["f1"], "--", color="black", label="formal Re F1")
    ax.plot(frequencies, curves["a"], linewidth=2, label="A saved median")
    ax.plot(frequencies, curves["e1"], linewidth=2, label="E1 median")
    for lower, upper in GUARDRAIL_REGIONS:
        ax.axvspan(lower, upper, color="#bbbbbb", alpha=0.08)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Median-across-seeds validation RMSE")
    ax.set_title("F3-E1 frequency-wise validation behavior")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "median_frequency_rmse.png", dpi=180)
    plt.close(fig)


def write_decision_markdown(session_dir: Path, comparison: dict[str, Any]) -> None:
    a = comparison["branch_medians"]["a"]
    e1 = comparison["branch_medians"]["e1"]
    target = comparison["target_region_comparison"]
    status = "PASS" if comparison["stage_passed"] else "STOP / NO PASS"
    lines = [
        "# Re Global SR F3-E1 Decision Report",
        "",
        f"- Stage decision: **{status}**",
        f"- Saved A median validation RMSE: `{a['validation_rmse']:.9f}`",
        f"- E1 median validation RMSE: `{e1['validation_rmse']:.9f}`",
        f"- Saved A median curve-mean RMSE: `{a['curve_mean_rmse']:.9f}`",
        f"- E1 median curve-mean RMSE: `{e1['curve_mean_rmse']:.9f}`",
        f"- Saved A median worst-curve RMSE: `{a['worst_curve_rmse']:.9f}`",
        f"- E1 median worst-curve RMSE: `{e1['worst_curve_rmse']:.9f}`",
        f"- Saved A median 1300–2000 Hz RMSE: `{target['saved_branch_a_median_rmse']:.9f}`",
        f"- E1 median 1300–2000 Hz RMSE: `{target['branch_e1_median_rmse']:.9f}`",
        f"- Test rows accessed: `{comparison['test_rows_accessed']}`",
        "",
        "## Predecessor guardrails",
        "",
    ]
    for row in comparison["predecessor_a_guardrails"]:
        lines.append(
            f"- {int(row['lower_hz'])}–{int(row['upper_hz'])} Hz: "
            f"E1 median `{row['branch_e1_median_rmse']:.9f}`, "
            f"saved A `{row['saved_a_control_median_rmse']:.9f}`, "
            f"passed `{row['passed']}`"
        )
    lines.extend(
        [
            "",
            "Only nested log/sqrt constraints changed in this small-budget screen.",
            "This is not a formal Re model replacement decision.",
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

    paths = d1.required_paths(args)
    context = load_context(args)
    acceptance_rules = {
        "full_validation": "E1 three-seed median RMSE must be strictly below saved A",
        "curve_behavior": "E1 median curve-mean and worst-curve RMSE must both be strictly below saved A",
        "target_region": "E1 median 1300-2000 Hz RMSE must be strictly below saved A",
        "guardrails": "E1 may not worsen more than 10% versus saved A in any declared region, both by median and matched seed",
        "constraint_compliance": "every Hall-of-Fame and selected equation must obey the frozen nesting constraint",
        "fallbacks": "every E1 candidate set includes exact saved-A and zero-residual formal-F1 fallbacks",
    }
    preflight = {
        "schema_version": 1,
        "stage": STAGE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session_dir),
        "seeds_predeclared_before_search": list(SEEDS),
        "comparison": {
            "control_a": "saved F3-C branch A with original unconstrained search; no rerun",
            "candidate_e1": "same branch A representation with nested log/sqrt constraints",
            "single_changed_factor": "nested_constraints",
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
        "nested_constraints": NESTED_CONSTRAINTS,
        "nested_constraint_semantics": "zero forbids the named inner operator anywhere under the named outer operator",
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
        "input_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "execute": True,
    }
    dump_json(session_dir / "preflight_manifest.json", preflight)
    log(
        "F3-E1 preflight frozen: seeds=42/123/456, saved A versus nested-constrained E1, "
        "budget=20/6/40/24, Plan B, validation-only."
    )

    e1_summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = session_dir / "runs" / BRANCH / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        worker_log = logs_dir / f"e1_seed_{seed}.log"
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
        log(f"Starting PySR run: branch=E1, seed={seed}.")
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
                f"F3-E1 worker failed: seed={seed}, exit={return_code}. See {worker_log}"
            )
        summary = load_json(run_dir / "selected_run_summary.json")
        if summary.get("test_rows_accessed") is not False:
            raise ValueError("An F3-E1 worker reported test-row access.")
        e1_summaries.append(summary)
        log(
            f"Completed PySR run: branch=E1, seed={seed}, "
            f"validation_RMSE={summary['validation_metrics']['rmse']:.9f}."
        )

    log("All three E1 searches completed; comparing against saved branch A.")
    comparison = aggregate_runs(session_dir, e1_summaries, context)
    plot_results(session_dir, e1_summaries, context)
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
        "new_e1_run_summaries": e1_summaries,
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
        f"F3-E1 complete: stage_passed={comparison['stage_passed']}, "
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
