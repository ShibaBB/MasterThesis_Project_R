"""Plan or execute the segmented SR local-frequency F1 experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train F1 with frozen phase-one material policies plus local t/t^2/t^3."
    )
    parser.add_argument("--dataset-run", default="run1")
    parser.add_argument("--targets", nargs="+", choices=("re", "im"), default=["re", "im"])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--niterations", type=int, default=100)
    parser.add_argument("--populations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--maxsize", type=int, default=24)
    parser.add_argument("--max-samples-per-segment", type=int, default=None)
    parser.add_argument("--run-label", default="frequency_f1")
    parser.add_argument(
        "--artifact-root", type=Path, default=SCRIPT_DIR / "artifacts"
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date = datetime.now().strftime("%Y%m%d")
    records = []
    for target in args.targets:
        slug = f"{date}_{args.dataset_run}_{args.run_label}"
        train_dir = args.artifact_root / target / "train" / slug
        eval_dir = args.artifact_root / target / "eval" / f"{slug}_best_loss"
        train_command = [
            sys.executable,
            str(SCRIPT_DIR / "train_segmented_symbolic_models.py"),
            "--target", target,
            "--dataset-run", args.dataset_run,
            "--output-dir", str(train_dir),
            "--feature-branch", "frequency_enhanced_f1",
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
        records.append(
            {
                "target": target,
                "train_dir": str(train_dir),
                "eval_dir": str(eval_dir),
                "train_command": train_command,
                "eval_command": eval_command,
            }
        )

    manifest = {
        "schema_version": 1,
        "purpose": "segmented SR F1 local-frequency representation experiment",
        "created_at": datetime.now().astimezone().isoformat(),
        "execute": args.execute,
        "f0_retrained": False,
        "feature_change": "retain log10_f and append local_t, local_t2, local_t3",
        "frequency_dependency_forced": False,
        "phase1_material_policy_frozen": True,
        "candidate_selection": "validation RMSE only",
        "shape_metrics": "diagnostic only",
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
    manifest_root = args.artifact_root / "frequency_f1_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.dataset_run}_{args.run_label}.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Experiment manifest: {manifest_path}", flush=True)

    for record in records:
        print(f"\n[{record['target']}] F1 training", flush=True)
        print(subprocess.list2cmdline(record["train_command"]), flush=True)
        print(f"[{record['target']}] F1 evaluation", flush=True)
        print(subprocess.list2cmdline(record["eval_command"]), flush=True)
        if args.execute:
            subprocess.run(record["train_command"], check=True, cwd=SCRIPT_DIR.parents[1])
            subprocess.run(record["eval_command"], check=True, cwd=SCRIPT_DIR.parents[1])


if __name__ == "__main__":
    main()

