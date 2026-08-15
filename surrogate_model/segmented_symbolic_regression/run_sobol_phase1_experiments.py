"""Plan or execute matched phase-one Sobol feature experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sobol_feature_policy import SUPPORTED_BRANCHES


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched all-feature and Sobol-subset segmented SR experiments."
    )
    parser.add_argument("--dataset-run", default="run1")
    parser.add_argument("--targets", nargs="+", choices=("re", "im"), default=["re", "im"])
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=SUPPORTED_BRANCHES,
        default=["all", "sobol_subset"],
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--niterations", type=int, default=100)
    parser.add_argument("--populations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--maxsize", type=int, default=24)
    parser.add_argument("--max-samples-per-segment", type=int, default=None)
    parser.add_argument(
        "--run-label",
        default="sobol_phase1",
        help="Short artifact label; matched settings remain recorded in metadata.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=SCRIPT_DIR / "artifacts",
        help="Artifact root; override for disposable smoke runs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute commands. Without this flag, only write and print the manifest.",
    )
    return parser.parse_args()


def command_record(args: argparse.Namespace, target: str, branch: str) -> dict[str, Any]:
    date = datetime.now().strftime("%Y%m%d")
    slug = f"{date}_{args.dataset_run}_{args.run_label}_{branch}"
    train_dir = args.artifact_root / target / "train" / slug
    eval_dir = args.artifact_root / target / "eval" / f"{slug}_best_loss"

    train_command = [
        sys.executable,
        str(SCRIPT_DIR / "train_segmented_symbolic_models.py"),
        "--target", target,
        "--dataset-run", args.dataset_run,
        "--output-dir", str(train_dir),
        "--feature-branch", branch,
        "--random-seed", str(args.random_seed),
        "--niterations", str(args.niterations),
        "--populations", str(args.populations),
        "--population-size", str(args.population_size),
        "--maxsize", str(args.maxsize),
        "--model-selection", "best",
    ]
    if args.max_samples_per_segment is not None:
        train_command.extend(
            ["--max-samples-per-segment", str(args.max_samples_per_segment)]
        )

    eval_command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_segmented_symbolic_candidates.py"),
        "--target", target,
        "--dataset-run", args.dataset_run,
        "--training-dir", str(train_dir),
        "--output-dir", str(eval_dir),
        "--selection-rule", "best_loss",
    ]
    return {
        "target": target,
        "branch": branch,
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "train_command": train_command,
        "eval_command": eval_command,
    }


def main() -> None:
    args = parse_args()
    records = [
        command_record(args, target, branch)
        for target in args.targets
        for branch in args.branches
    ]
    manifest = {
        "schema_version": 1,
        "purpose": "matched Sobol-guided segmented SR phase-one feature experiment",
        "created_at": datetime.now().astimezone().isoformat(),
        "execute": args.execute,
        "selection_contract": (
            "Feature-branch and candidate decisions use validation results only; "
            "test metrics are final reporting only."
        ),
        "matched_settings": {
            "dataset_run": args.dataset_run,
            "random_seed": args.random_seed,
            "niterations": args.niterations,
            "populations": args.populations,
            "population_size": args.population_size,
            "maxsize": args.maxsize,
            "max_samples_per_segment": args.max_samples_per_segment,
        },
        "runs": records,
    }
    manifest_root = args.artifact_root / "sobol_phase1_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = manifest_root / f"{stamp}_{args.dataset_run}_{args.run_label}.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Experiment manifest: {manifest_path}")
    for record in records:
        print(f"\n[{record['target']} / {record['branch']}] training")
        print(subprocess.list2cmdline(record["train_command"]))
        print(f"[{record['target']} / {record['branch']}] validation selection + final reporting")
        print(subprocess.list2cmdline(record["eval_command"]))
        if args.execute:
            subprocess.run(record["train_command"], check=True, cwd=SCRIPT_DIR.parents[1])
            subprocess.run(record["eval_command"], check=True, cwd=SCRIPT_DIR.parents[1])

    if args.execute and {"all", "sobol_subset"}.issubset(args.branches):
        date = datetime.now().strftime("%Y%m%d")
        for target in args.targets:
            by_branch = {
                record["branch"]: record
                for record in records
                if record["target"] == target
            }
            comparison_dir = (
                args.artifact_root
                / target
                / "comparison"
                / f"{date}_{args.dataset_run}_{args.run_label}"
            )
            compare_command = [
                sys.executable,
                str(SCRIPT_DIR / "compare_sobol_phase1_results.py"),
                "--all-eval-dir", by_branch["all"]["eval_dir"],
                "--sobol-eval-dir", by_branch["sobol_subset"]["eval_dir"],
                "--output-dir", str(comparison_dir),
            ]
            print(f"\n[{target}] validation-only A/B comparison")
            print(subprocess.list2cmdline(compare_command))
            subprocess.run(compare_command, check=True, cwd=SCRIPT_DIR.parents[1])


if __name__ == "__main__":
    main()
