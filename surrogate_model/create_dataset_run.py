"""Create a future dataset-run configuration without editing model code.

Example:
    python surrogate_model/create_dataset_run.py run4 --from-run run3 \
        --random-seed 45 --set-default
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dataset_run_config import (
    ACTIVE_CONFIG_FILE,
    SURROGATE_ROOT,
    default_dataset_run,
    load_dataset_run_config,
    validate_dataset_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_run", type=validate_dataset_run)
    parser.add_argument("--from-run", default=None, type=validate_dataset_run)
    parser.add_argument("--random-seed", required=True, type=int)
    parser.add_argument("--freq-min", type=float, default=None)
    parser.add_argument("--freq-max", type=float, default=None)
    parser.add_argument("--frequency-points", type=int, default=None)
    parser.add_argument("--curve-samples", type=int, default=None)
    parser.add_argument("--set-default", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show the target without writing files.")
    return parser.parse_args()


def write_json_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    source_run = args.from_run or default_dataset_run()
    source = load_dataset_run_config(source_run)
    config = copy.deepcopy({key: value for key, value in source.items() if key not in {"paths", "config_file"}})
    config["run_id"] = args.dataset_run
    config["description"] = f"Dataset run {args.dataset_run}, cloned from {source_run}."
    generation = config["generation"]
    generation["random_seed"] = args.random_seed
    frequency = generation["frequency_grid_hz"]
    if args.freq_min is not None:
        frequency["min"] = args.freq_min
    if args.freq_max is not None:
        frequency["max"] = args.freq_max
    if args.frequency_points is not None:
        frequency["points"] = args.frequency_points
    if args.curve_samples is not None:
        generation["curve_samples"] = args.curve_samples

    # Frequency-domain changes require explicit segment editing; refusing an
    # inconsistent clone is safer than silently creating uncovered frequencies.
    segment_bounds = [item["bounds_hz"] for item in config["segmented_sr"]["segments"]]
    global_bounds = config["global_sr"]["bounds_hz"]
    expected = [frequency["min"], frequency["max"]]
    if segment_bounds[0][0] != expected[0] or segment_bounds[-1][1] != expected[1] or global_bounds != expected:
        raise ValueError(
            "The requested frequency bounds differ from the source run. Create the run config "
            "without frequency overrides, or edit the new segment/global definitions explicitly."
        )

    run_dir = SURROGATE_ROOT / "datasets" / args.dataset_run
    config_file = run_dir / "dataset_config.json"
    if args.dry_run:
        print(f"Would create dataset run config: {config_file}")
        print(f"Source run: {source_run} | random seed: {args.random_seed}")
        print(f"Would change default: {bool(args.set_default)}")
        return
    write_json_new(config_file, config)

    material = generation["material"]
    manifest = {
        "schema_version": 1,
        "run_id": args.dataset_run,
        "status": "configured_not_generated",
        "description": config["description"],
        "config_path": config_file.relative_to(SURROGATE_ROOT.parent).as_posix(),
        "generation": {
            "material": material,
            "porosity_cases": generation["porosity_cases"],
            "sampling_method": generation["sampling_method"],
            "random_seed": generation["random_seed"],
            "curve_samples": generation["curve_samples"],
            "frequency_points": frequency["points"],
            "frequency_range_hz": [frequency["min"], frequency["max"]],
        },
        "teacher_dataset": {
            "path": f"surrogate_model/datasets/{args.dataset_run}/MLP/{material}_surrogate_dataset.mat",
            "status": "not_generated",
        },
        "shared_curve_split": {
            "path": f"surrogate_model/datasets/{args.dataset_run}/shared_curve_split.json",
            "status": "not_generated",
            "frequency_grid_independent": True,
        },
        "derived_datasets": [
            {
                "model_branch": "segmented_SR",
                "path": f"surrogate_model/datasets/{args.dataset_run}/segmented_SR/{material}_symbolic_segmented.mat",
                "status": "not_generated",
            },
            {
                "model_branch": "global_SR",
                "path": f"surrogate_model/datasets/{args.dataset_run}/global_SR/{material}_symbolic_global.mat",
                "status": "not_generated",
            },
        ],
    }
    write_json_new(run_dir / "dataset_manifest.json", manifest)

    if args.set_default:
        with ACTIVE_CONFIG_FILE.open(encoding="utf-8") as file:
            active = json.load(file)
        active["default_dataset_run"] = args.dataset_run
        with ACTIVE_CONFIG_FILE.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(active, file, indent=2)
            file.write("\n")

    print(f"Created dataset run config: {config_file}")
    print(f"Default dataset run changed: {bool(args.set_default)}")


if __name__ == "__main__":
    main()
