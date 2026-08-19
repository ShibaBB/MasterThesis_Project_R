"""Run target-specific staged F2 modulation screens and formal Global SR fits."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from global_frequency_modulation import build_custom_frequency_modulation


SCRIPT_DIR = Path(__file__).resolve().parent
RE_GUARDRAIL_BANDS = ((700.0, 1000.0), (1300.0, 1650.0), (1650.0, 2000.0))
IM_GUARDRAIL_BANDS = (
    (700.0, 1000.0),
    (1000.0, 1300.0),
    (1300.0, 1650.0),
    (3000.0, 4000.0),
    (4000.0, 4950.0),
)
F1_TRAIN_DIRS = {
    "re": SCRIPT_DIR / "artifacts" / "re" / "train" / "20260816_run1_sobol_modulated",
    "im": SCRIPT_DIR / "artifacts" / "im" / "train" / "20260816_run1_sobol_modulated",
}
F1_EVAL_DIRS = {
    "re": SCRIPT_DIR / "artifacts" / "re" / "eval" / "20260816_run1_sobol_modulated_best_loss",
    "im": SCRIPT_DIR / "artifacts" / "im" / "eval" / "20260816_run1_sobol_modulated_best_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", default="run1")
    parser.add_argument("--targets", nargs="+", choices=("re", "im"), default=["re", "im"])
    parser.add_argument("--artifact-root", type=Path, default=SCRIPT_DIR / "artifacts")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--screen-niterations", type=int, default=20)
    parser.add_argument("--screen-populations", type=int, default=6)
    parser.add_argument("--screen-population-size", type=int, default=40)
    parser.add_argument("--formal-niterations", type=int, default=100)
    parser.add_argument("--formal-populations", type=int, default=12)
    parser.add_argument("--formal-population-size", type=int, default=80)
    parser.add_argument("--maxsize", type=int, default=24)
    parser.add_argument("--guardrail-relative-limit", type=float, default=0.10)
    parser.add_argument("--session-label", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-formal", action="store_true")
    return parser.parse_args()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def run(command: list[str], execute: bool) -> None:
    print("COMMAND:", subprocess.list2cmdline(command), flush=True)
    if execute:
        subprocess.run(command, check=True, cwd=SCRIPT_DIR.parents[1])


def evaluation_summary(eval_dir: Path) -> dict[str, Any]:
    with (eval_dir / "selected_global_summary.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_validation_reference(
    target: str, dataset_run: str, reference_dir: Path, execute: bool
) -> None:
    summary_path = reference_dir / "selected_global_summary.json"
    if summary_path.exists():
        summary = evaluation_summary(reference_dir)
        if summary.get("test_rows_accessed") is not False:
            raise ValueError(f"F2 validation reference is not validation-only: {reference_dir}")
        return
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_global_symbolic_candidates.py"),
        "--target", target,
        "--dataset-run", dataset_run,
        "--training-dir", str(F1_TRAIN_DIRS[target]),
        "--output-dir", str(reference_dir),
        "--selection-rule", "best_loss",
        "--validation-only",
    ]
    run(command, execute)


def spec_feature(name: str, base: str, components: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "base_feature": base, "components": components}


def re_spec(onset_center: float, onset_transition: float, high_width: float) -> dict[str, Any]:
    features = [
        spec_feature(
            "alpha_infinity__f2_onset",
            "alpha_infinity",
            [{"kind": "highpass", "center_hz": onset_center, "transition_log10": onset_transition, "weight": 1.0}],
        ),
        spec_feature(
            "alpha_infinity__f2_high",
            "alpha_infinity",
            [{"kind": "gaussian", "center_hz": 3422.0, "width_log10": high_width, "weight": 1.0}],
        ),
    ]
    spec = build_custom_frequency_modulation(
        "re", "global_sr_f2_re_split_alpha_infinity", features,
        "F2 frozen residual diagnosis and teacher Sobol envelope locations",
    )
    spec["f2_design"] = {
        "onset_center_hz": onset_center,
        "onset_transition_log10": onset_transition,
        "high_center_hz": 3422.0,
        "high_width_log10": high_width,
    }
    return spec


def im_spec(
    low_center: float,
    low_transition: float,
    lambda_variant: str = "L0",
) -> dict[str, Any]:
    features = [
        spec_feature(
            "log10_k0_prime__f2_low",
            "log10_k0_prime",
            [{"kind": "lowpass", "center_hz": low_center, "transition_log10": low_transition, "weight": 1.0}],
        ),
        spec_feature(
            "log10_k0_prime__f2_high",
            "log10_k0_prime",
            [{"kind": "highpass", "center_hz": 3900.0, "transition_log10": 0.10, "weight": 1.0}],
        ),
    ]
    if lambda_variant == "L0":
        features.append(spec_feature(
            "log10_lambda_prime__f2_mod", "log10_lambda_prime",
            [
                {"kind": "lowpass", "center_hz": 700.0, "transition_log10": 0.13, "weight": 0.35},
                {"kind": "highpass", "center_hz": 4000.0, "transition_log10": 0.10, "weight": 1.0},
            ],
        ))
    elif lambda_variant == "L1":
        features.append(spec_feature(
            "log10_lambda_prime__f2_high", "log10_lambda_prime",
            [{"kind": "highpass", "center_hz": 4000.0, "transition_log10": 0.10, "weight": 1.0}],
        ))
    elif lambda_variant != "L2":
        raise ValueError(f"Unsupported lambda_prime ablation: {lambda_variant}")
    spec = build_custom_frequency_modulation(
        "im", "global_sr_f2_im_split_k0_prime", features,
        "F2 frozen residual diagnosis and teacher Sobol envelope locations",
    )
    spec["f2_design"] = {
        "k0_low_center_hz": low_center,
        "k0_low_transition_log10": low_transition,
        "k0_high_center_hz": 3900.0,
        "k0_high_transition_log10": 0.10,
        "lambda_prime_variant": lambda_variant,
    }
    return spec


def branch_paths(root: Path, target: str, session: str, label: str) -> tuple[Path, Path]:
    slug = f"{session}_{label}"
    return (
        root / target / "train" / slug,
        root / target / "eval" / f"{slug}_v",
    )


def run_branch(
    args: argparse.Namespace,
    target: str,
    session: str,
    label: str,
    spec: dict[str, Any],
    spec_dir: Path,
    formal: bool = False,
) -> dict[str, Any]:
    train_dir, eval_dir = branch_paths(args.artifact_root, target, session, label)
    spec_path = spec_dir / f"{target}_{label}.json"
    dump_json(spec_path, spec)
    summary_path = eval_dir / "selected_global_summary.json"
    if summary_path.exists():
        return evaluation_summary(eval_dir)
    niterations = args.formal_niterations if formal else args.screen_niterations
    populations = args.formal_populations if formal else args.screen_populations
    population_size = args.formal_population_size if formal else args.screen_population_size
    train_command = [
        sys.executable, str(SCRIPT_DIR / "train_global_symbolic_model.py"),
        "--target", target,
        "--dataset-run", args.dataset_run,
        "--output-dir", str(train_dir),
        "--feature-branch", "sobol_modulated",
        "--frequency-modulation-spec", str(spec_path),
        "--random-seed", str(args.random_seed),
        "--niterations", str(niterations),
        "--populations", str(populations),
        "--population-size", str(population_size),
        "--maxsize", str(args.maxsize),
        "--model-selection", "best",
    ]
    eval_command = [
        sys.executable, str(SCRIPT_DIR / "evaluate_global_symbolic_candidates.py"),
        "--target", target,
        "--dataset-run", args.dataset_run,
        "--training-dir", str(train_dir),
        "--output-dir", str(eval_dir),
        "--selection-rule", "best_loss",
    ]
    # Training never needs test rows. The frozen formal Hall-of-Fame is opened
    # on test exactly once by the evaluator after validation-only selection.
    train_command.append("--validation-only")
    if not formal:
        eval_command.append("--validation-only")
    print(f"\n[{target.upper()} {label}] {'formal' if formal else 'screen'} training", flush=True)
    run(train_command, args.execute)
    run(eval_command, args.execute)
    if not args.execute:
        return {"planned": True, "eval_dir": str(eval_dir)}
    return evaluation_summary(eval_dir)


def region_map(summary: dict[str, Any]) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (float(row["lower_hz"]), float(row["upper_hz"])): row
        for row in summary["validation_regional_metrics"]
    }


def compare_screen(
    stage: str,
    target: str,
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]],
    reference: dict[str, Any],
    guardrail_bands: tuple[tuple[float, float], ...],
    relative_limit: float,
    output_dir: Path,
) -> tuple[str, dict[str, Any]]:
    reference_regions = region_map(reference)
    rows: list[dict[str, Any]] = []
    by_label: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for label, spec, summary in candidates:
        by_label[label] = (spec, summary)
        row: dict[str, Any] = {
            "stage": stage,
            "target": target,
            "branch": label,
            "validation_rmse": summary["validation_selection_metrics"]["rmse"],
            "validation_mae": summary["validation_selection_metrics"]["mae"],
            "validation_r2": summary["validation_selection_metrics"]["r2"],
            "test_rows_accessed": summary.get("test_rows_accessed"),
        }
        candidate_regions = region_map(summary)
        passes = True
        for band in guardrail_bands:
            key = f"{int(band[0])}_{int(band[1])}"
            base = float(reference_regions[band]["rmse"])
            current = float(candidate_regions[band]["rmse"])
            relative = current / base - 1.0
            row[f"rmse_{key}"] = current
            row[f"rmse_relative_vs_f1_{key}"] = relative
            row[f"bias_{key}"] = candidate_regions[band]["bias"]
            passes = passes and relative <= relative_limit
        row["passes_regional_guardrails"] = passes
        rows.append(row)
    feasible = [row for row in rows if row["passes_regional_guardrails"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["selected_by_validation_rmse"] = False
    comparison_path = output_dir / f"{stage}_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not feasible:
        dump_json(output_dir / f"{stage}_blocked.json", {
            "stage": stage,
            "target": target,
            "reason": "No screening branch passed every declared F1 regional RMSE guardrail.",
            "guardrail_relative_limit": relative_limit,
            "test_rows_accessed": False,
        })
        raise RuntimeError(f"{target} {stage}: no branch passes all regional guardrails")
    winner_row = min(feasible, key=lambda row: (row["validation_rmse"], row["branch"]))
    winner = str(winner_row["branch"])
    for row in rows:
        row["selected_by_validation_rmse"] = row["branch"] == winner
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    winner_spec, winner_summary = by_label[winner]
    dump_json(output_dir / f"{stage}_decision.json", {
        "schema_version": 1,
        "stage": stage,
        "target": target,
        "decision_data": "complete full-range validation rows with declared regional F1 guardrails",
        "test_metrics_used_for_decision": False,
        "test_rows_accessed": False,
        "guardrail_relative_limit": relative_limit,
        "guardrail_bands_hz": [list(band) for band in guardrail_bands],
        "selected_branch": winner,
        "selected_validation_metrics": winner_summary["validation_selection_metrics"],
        "selected_specification": winner_spec,
    })
    return winner, winner_spec


def formal_decision(
    args: argparse.Namespace,
    target: str,
    formal: dict[str, Any],
    reference_validation: dict[str, Any],
    output_dir: Path,
) -> None:
    bands = RE_GUARDRAIL_BANDS if target == "re" else IM_GUARDRAIL_BANDS
    base_regions = region_map(reference_validation)
    candidate_regions = region_map(formal)
    regional = []
    passes = True
    for band in bands:
        relative = float(candidate_regions[band]["rmse"]) / float(base_regions[band]["rmse"]) - 1.0
        regional.append({
            "lower_hz": band[0], "upper_hz": band[1],
            "f1_validation_rmse": base_regions[band]["rmse"],
            "f2_validation_rmse": candidate_regions[band]["rmse"],
            "relative_delta": relative,
            "f1_validation_bias": base_regions[band]["bias"],
            "f2_validation_bias": candidate_regions[band]["bias"],
        })
        passes = passes and relative <= args.guardrail_relative_limit
    f1_validation_rmse = float(reference_validation["validation_selection_metrics"]["rmse"])
    f2_validation_rmse = float(formal["validation_selection_metrics"]["rmse"])
    accepted = passes and f2_validation_rmse <= f1_validation_rmse
    with (F1_EVAL_DIRS[target] / "selected_global_summary.json").open("r", encoding="utf-8") as file:
        f1_formal = json.load(file)
    with (Path(formal["training_dir"]) / "training_metadata.json").open("r", encoding="utf-8") as file:
        formal_training = json.load(file)
    training_accessed_test = not bool(
        formal_training.get("training_args", {}).get("validation_only", False)
    )
    dump_json(output_dir / "formal_acceptance_decision.json", {
        "schema_version": 1,
        "target": target,
        "selection_data": "validation only",
        "test_metrics_used_for_decision": False,
        "selected_checkpoint": "F2" if accepted else "F1",
        "f1_validation_metrics": reference_validation["validation_selection_metrics"],
        "f2_validation_metrics": formal["validation_selection_metrics"],
        "regional_guardrails_passed": passes,
        "regional_guardrails": regional,
        "f1_test_metrics_final_reporting": f1_formal["overall_metrics"],
        "f2_test_metrics_final_reporting": formal["overall_metrics"],
        "formal_training_accessed_test_rows": training_accessed_test,
        "formal_evaluator_accessed_test_rows": True,
        "f2_test_access_count_after_freeze": 2 if training_accessed_test else 1,
    })


def main() -> None:
    args = parse_args()
    session = args.session_label or datetime.now().strftime("%Y%m%d_%H%M%S_f2")
    session_dir = args.artifact_root / "f2_manifests" / session
    spec_dir = session_dir / "specs"
    comparison_dir = session_dir / "comparisons"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "session": session,
        "dataset_run": args.dataset_run,
        "targets": args.targets,
        "execute": args.execute,
        "screen_budget": {"niterations": args.screen_niterations, "populations": args.screen_populations, "population_size": args.screen_population_size, "maxsize": args.maxsize, "random_seed": args.random_seed},
        "formal_budget": {"niterations": args.formal_niterations, "populations": args.formal_populations, "population_size": args.formal_population_size, "maxsize": args.maxsize, "random_seed": args.random_seed},
        "test_access_contract": "No screening train/evaluation subsets or predicts test rows; frozen formal winners open test once.",
    }
    dump_json(session_dir / "experiment_manifest.json", manifest)

    for target in args.targets:
        print(f"\n===== TARGET {target.upper()} F2 =====", flush=True)
        reference_dir = session_dir / "f1_validation_reference" / target
        ensure_validation_reference(target, args.dataset_run, reference_dir, args.execute)
        if not args.execute:
            continue
        reference = evaluation_summary(reference_dir)
        bands = RE_GUARDRAIL_BANDS if target == "re" else IM_GUARDRAIL_BANDS
        if target == "re":
            stage1 = []
            for center in (1500.0, 1800.0, 2100.0):
                for transition in (0.12, 0.18, 0.24):
                    label = f"r1_c{int(center)}_t{int(transition * 100):02d}"
                    spec = re_spec(center, transition, 0.22)
                    stage1.append((label, spec, run_branch(args, target, session, label, spec, spec_dir)))
            _, winner_spec = compare_screen("re_stage1_onset", target, stage1, reference, bands, args.guardrail_relative_limit, comparison_dir / target)
            design = winner_spec["f2_design"]
            stage2 = []
            for width in (0.18, 0.22, 0.28):
                label = f"r2_w{int(width * 100):02d}"
                spec = re_spec(design["onset_center_hz"], design["onset_transition_log10"], width)
                stage2.append((label, spec, run_branch(args, target, session, label, spec, spec_dir)))
            _, frozen_spec = compare_screen("re_stage2_high_width", target, stage2, reference, bands, args.guardrail_relative_limit, comparison_dir / target)
        else:
            stage1 = []
            for center in (600.0, 750.0, 900.0):
                for transition in (0.10, 0.16, 0.22):
                    label = f"i1_c{int(center)}_t{int(transition * 100):02d}"
                    spec = im_spec(center, transition, "L0")
                    stage1.append((label, spec, run_branch(args, target, session, label, spec, spec_dir)))
            _, winner_spec = compare_screen("im_stage1_k0_low", target, stage1, reference, bands, args.guardrail_relative_limit, comparison_dir / target)
            design = winner_spec["f2_design"]
            stage2 = []
            for variant in ("L0", "L1", "L2"):
                label = f"i2_{variant.lower()}"
                spec = im_spec(design["k0_low_center_hz"], design["k0_low_transition_log10"], variant)
                stage2.append((label, spec, run_branch(args, target, session, label, spec, spec_dir)))
            _, frozen_spec = compare_screen("im_stage2_lambda_ablation", target, stage2, reference, bands, args.guardrail_relative_limit, comparison_dir / target)
        frozen_path = spec_dir / f"{target}_frozen_winner.json"
        dump_json(frozen_path, frozen_spec)
        print(f"Frozen {target} F2 specification: {frozen_path}", flush=True)
        if not args.skip_formal:
            formal = run_branch(args, target, session, "f2_formal", frozen_spec, spec_dir, formal=True)
            formal_decision(args, target, formal, reference, comparison_dir / target)

    print(f"\nF2 experiment session complete: {session_dir}", flush=True)


if __name__ == "__main__":
    main()
