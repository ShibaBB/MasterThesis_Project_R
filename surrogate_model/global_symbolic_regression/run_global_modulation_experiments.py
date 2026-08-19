"""Train and evaluate smooth teacher-Sobol-modulated Global SR models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Re/Im Global SR with all base parameters plus smooth modulation."
    )
    parser.add_argument("--dataset-run", default="run1")
    parser.add_argument("--targets", nargs="+", choices=("re", "im"), default=["re", "im"])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--niterations", type=int, default=100)
    parser.add_argument("--populations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--maxsize", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--run-label", default="sobol_modulated")
    parser.add_argument(
        "--artifact-root", type=Path, default=SCRIPT_DIR / "artifacts"
    )
    parser.add_argument(
        "--baseline-re-eval-dir",
        type=Path,
        default=SCRIPT_DIR / "artifacts" / "re" / "eval" / "20260815_run1_sobol_phase1_all_best_loss",
    )
    parser.add_argument(
        "--baseline-im-eval-dir",
        type=Path,
        default=SCRIPT_DIR / "artifacts" / "im" / "eval" / "20260815_run1_sobol_phase1_all_best_loss",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip matched-control comparison, intended only for pipeline smoke runs.",
    )
    return parser.parse_args()


def command_record(args: argparse.Namespace, target: str) -> dict[str, Any]:
    date = datetime.now().strftime("%Y%m%d")
    slug = f"{date}_{args.dataset_run}_{args.run_label}"
    train_dir = args.artifact_root / target / "train" / slug
    eval_dir = args.artifact_root / target / "eval" / f"{slug}_best_loss"
    baseline_dir = (
        args.baseline_re_eval_dir if target == "re" else args.baseline_im_eval_dir
    )
    comparison_dir = (
        args.artifact_root / target / "comparison"
        / f"{date}_{args.dataset_run}_{args.run_label}"
    )
    train_command = [
        sys.executable,
        str(SCRIPT_DIR / "train_global_symbolic_model.py"),
        "--target", target,
        "--dataset-run", args.dataset_run,
        "--output-dir", str(train_dir),
        "--feature-branch", "sobol_modulated",
        "--random-seed", str(args.random_seed),
        "--niterations", str(args.niterations),
        "--populations", str(args.populations),
        "--population-size", str(args.population_size),
        "--maxsize", str(args.maxsize),
        "--model-selection", "best",
    ]
    if args.max_samples is not None:
        train_command.extend(["--max-samples", str(args.max_samples)])
    eval_command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_global_symbolic_candidates.py"),
        "--target", target,
        "--dataset-run", args.dataset_run,
        "--training-dir", str(train_dir),
        "--output-dir", str(eval_dir),
        "--selection-rule", "best_loss",
    ]
    compare_command = [
        sys.executable,
        str(SCRIPT_DIR / "compare_global_modulation_results.py"),
        "--all-eval-dir", str(baseline_dir),
        "--modulated-eval-dir", str(eval_dir),
        "--output-dir", str(comparison_dir),
    ]
    return {
        "target": target,
        "branch": "sobol_modulated",
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "baseline_eval_dir": str(baseline_dir),
        "comparison_dir": str(comparison_dir),
        "train_command": train_command,
        "eval_command": eval_command,
        "compare_command": compare_command,
    }


def main() -> None:
    args = parse_args()
    records = [command_record(args, target) for target in args.targets]
    manifest = {
        "schema_version": 1,
        "purpose": "smooth teacher-Sobol parameter-frequency modulation for full-range Global SR",
        "created_at": datetime.now().astimezone().isoformat(),
        "execute": args.execute,
        "frequency_domains": [{"lower_hz": 100.0, "upper_hz": 4950.0}],
        "base_parameters_retained": True,
        "modulated_features_per_target": 5,
        "selection_contract": (
            "Compare the single full-range modulated model with the matched all-base "
            "control using validation RMSE only; test metrics are final reporting."
        ),
        "matched_settings": {
            "dataset_run": args.dataset_run,
            "random_seed": args.random_seed,
            "niterations": args.niterations,
            "populations": args.populations,
            "population_size": args.population_size,
            "maxsize": args.maxsize,
            "max_samples": args.max_samples,
        },
        "runs": records,
    }
    manifest_root = args.artifact_root / "modulation_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = manifest_root / f"{stamp}_{args.dataset_run}_{args.run_label}.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Experiment manifest: {manifest_path}")
    for record in records:
        stages = ["train_command", "eval_command"]
        if not args.skip_comparison:
            stages.append("compare_command")
        for stage in stages:
            print(f"\n[{record['target']}] {stage}")
            print(subprocess.list2cmdline(record[stage]))
            if args.execute:
                subprocess.run(record[stage], check=True, cwd=SCRIPT_DIR.parents[1])


if __name__ == "__main__":
    main()
