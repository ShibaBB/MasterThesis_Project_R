"""Benchmark threaded PySR C and fall back to deterministic serial B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS = SCRIPT_DIR / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", default="run1")
    parser.add_argument("--target", choices=("re", "im"), default="re")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--niterations", type=int, default=10)
    parser.add_argument("--populations", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--maxsize", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--julia-threads", type=int, default=8)
    parser.add_argument("--run-prefix", default="20260816_perf")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(args: argparse.Namespace, variant: str, replicate: int) -> dict[str, Any]:
    slug = f"{args.run_prefix}_{variant.lower()}{replicate}"
    train_dir = ARTIFACTS / args.target / "train" / slug
    eval_dir = ARTIFACTS / args.target / "eval" / f"{slug}_v"
    common = [
        "--target", args.target,
        "--dataset-run", args.dataset_run,
        "--feature-branch", "sobol_modulated",
        "--random-seed", str(args.random_seed),
        "--niterations", str(args.niterations),
        "--populations", str(args.populations),
        "--population-size", str(args.population_size),
        "--maxsize", str(args.maxsize),
        "--model-selection", "best",
        "--validation-only",
        "--batching",
        "--batch-size", str(args.batch_size),
        "--turbo",
    ]
    if variant == "C":
        performance = [
            "--parallelism", "multithreading",
            "--julia-threads", str(args.julia_threads),
            "--no-deterministic",
        ]
    elif variant == "B":
        performance = [
            "--parallelism", "serial",
            "--julia-threads", "1",
            "--deterministic",
        ]
    else:
        raise ValueError(f"Unknown benchmark variant: {variant}")
    train_command = [
        sys.executable,
        str(SCRIPT_DIR / "train_global_symbolic_model.py"),
        "--output-dir", str(train_dir),
        *common,
        *performance,
    ]
    eval_command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_global_symbolic_candidates.py"),
        "--target", args.target,
        "--dataset-run", args.dataset_run,
        "--training-dir", str(train_dir),
        "--output-dir", str(eval_dir),
        "--selection-rule", "best_loss",
        "--validation-only",
    ]
    print(f"\n[{variant}{replicate}] {subprocess.list2cmdline(train_command)}", flush=True)
    if args.execute:
        subprocess.run(train_command, check=True, cwd=SCRIPT_DIR.parents[1])
        subprocess.run(eval_command, check=True, cwd=SCRIPT_DIR.parents[1])
    if not args.execute:
        return {"variant": variant, "replicate": replicate, "planned": True}

    with (train_dir / "global_model_summary.json").open("r", encoding="utf-8") as file:
        training = json.load(file)
    with (eval_dir / "selected_global_summary.json").open("r", encoding="utf-8") as file:
        evaluation = json.load(file)
    with (eval_dir / "selected_candidate.csv").open("r", encoding="utf-8-sig", newline="") as file:
        selected = next(csv.DictReader(file))
    equations_path = train_dir / "equations" / "global_100_4950_equations.csv"
    return {
        "variant": variant,
        "replicate": replicate,
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "fit_wall_time_seconds": training["fit_wall_time_seconds"],
        "validation_rmse": evaluation["validation_selection_metrics"]["rmse"],
        "validation_mae": evaluation["validation_selection_metrics"]["mae"],
        "validation_r2": evaluation["validation_selection_metrics"]["r2"],
        "selected_candidate": evaluation["selected_candidate"],
        "selected_complexity": int(selected["complexity"]),
        "selected_equation": selected["equation"],
        "equations_sha256": sha256(equations_path),
        "test_rows_accessed": evaluation["test_rows_accessed"],
        "shared_split_hash": evaluation["shared_split_hash"],
    }


def compare_replicates(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    equation_match = left["selected_equation"] == right["selected_equation"]
    metrics_match = all(
        left[key] == right[key]
        for key in ("validation_rmse", "validation_mae", "validation_r2")
    )
    hall_of_fame_match = left["equations_sha256"] == right["equations_sha256"]
    return {
        "selected_equation_exact_match": equation_match,
        "validation_metrics_exact_match": metrics_match,
        "hall_of_fame_csv_exact_match": hall_of_fame_match,
        "exact_replay": equation_match and metrics_match and hall_of_fame_match,
        "validation_rmse_absolute_difference": abs(
            left["validation_rmse"] - right["validation_rmse"]
        ),
    }


def write_results(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "benchmark_decision.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    rows = result["runs"]
    if rows:
        fields = [
            "variant", "replicate", "fit_wall_time_seconds", "validation_rmse",
            "validation_mae", "validation_r2", "selected_candidate",
            "selected_complexity", "test_rows_accessed", "shared_split_hash",
            "train_dir", "eval_dir", "selected_equation", "equations_sha256",
        ]
        with (output_dir / "benchmark_runs.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    args = parse_args()
    output_dir = ARTIFACTS / "performance_benchmarks" / args.run_prefix
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "Seed replay and performance check for PySR acceleration plans C and B",
        "dataset_run": args.dataset_run,
        "target": args.target,
        "random_seed": args.random_seed,
        "budget": {
            "niterations": args.niterations,
            "populations": args.populations,
            "population_size": args.population_size,
            "maxsize": args.maxsize,
        },
        "batch_size": args.batch_size,
        "plan_c": "8-thread multithreading + batching + turbo; nondeterministic by PySR contract",
        "plan_b": "serial deterministic + batching + turbo",
        "test_access": "validation-only benchmark; test rows inaccessible",
        "execute": args.execute,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    runs = [run_one(args, "C", 1), run_one(args, "C", 2)]
    if not args.execute:
        write_results(output_dir, {"runs": runs, "decision": "planned"})
        return
    comparison_c = compare_replicates(runs[0], runs[1])
    if comparison_c["exact_replay"]:
        decision = "accept_C_empirically_replayed"
        comparison_b = None
    else:
        print("\nC did not replay exactly; executing requested serial plan B fallback.", flush=True)
        b_runs = [run_one(args, "B", 1), run_one(args, "B", 2)]
        runs.extend(b_runs)
        comparison_b = compare_replicates(b_runs[0], b_runs[1])
        decision = (
            "fallback_to_B_exact_replay"
            if comparison_b["exact_replay"]
            else "B_also_failed_exact_replay"
        )
    result = {
        "schema_version": 1,
        "decision": decision,
        "selection_rule": (
            "Use C only if two clean same-seed runs have identical Hall-of-Fame CSV, "
            "selected equation, and validation metrics; otherwise use B."
        ),
        "test_metrics_used": False,
        "runs": runs,
        "plan_c_replay": comparison_c,
        "plan_b_replay": comparison_b,
    }
    write_results(output_dir, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
