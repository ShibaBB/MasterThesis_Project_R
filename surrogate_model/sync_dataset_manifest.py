"""Synchronize a dataset manifest with its run configuration and file state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_run_config import SURROGATE_ROOT, default_dataset_run, load_dataset_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", default=default_dataset_run())
    parser.add_argument("--check", action="store_true", help="Validate and print status without writing.")
    return parser.parse_args()


def relative(path):
    return path.resolve().relative_to(SURROGATE_ROOT.parent).as_posix()


def build_manifest(config: dict) -> dict:
    run_id = config["run_id"]
    generation = config["generation"]
    frequency = generation["frequency_grid_hz"]
    paths = config["paths"]
    file_status = {
        key: paths[key].exists()
        for key in ("teacher_dataset", "shared_split", "segmented_dataset", "global_dataset")
    }
    generated_count = sum(file_status.values())
    overall_status = (
        "generated"
        if generated_count == len(file_status)
        else "configured_not_generated"
        if generated_count == 0
        else "partially_generated"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": overall_status,
        "description": config.get("description", ""),
        "config_path": relative(Path(config["config_file"])),
        "generation": {
            "material": generation["material"],
            "porosity_cases": generation["porosity_cases"],
            "sampling_method": generation["sampling_method"],
            "random_seed": generation["random_seed"],
            "curve_samples": generation["curve_samples"],
            "frequency_points": frequency["points"],
            "frequency_range_hz": [frequency["min"], frequency["max"]],
        },
        "teacher_dataset": {
            "path": relative(paths["teacher_dataset"]),
            "status": "generated" if file_status["teacher_dataset"] else "not_generated",
        },
        "shared_curve_split": {
            "path": relative(paths["shared_split"]),
            "status": "generated" if file_status["shared_split"] else "not_generated",
            "frequency_grid_independent": True,
        },
        "derived_datasets": [
            {
                "model_branch": "segmented_SR",
                "path": relative(paths["segmented_dataset"]),
                "status": "generated" if file_status["segmented_dataset"] else "not_generated",
                "segments_hz": config["segmented_sr"]["segments"],
            },
            {
                "model_branch": "global_SR",
                "path": relative(paths["global_dataset"]),
                "status": "generated" if file_status["global_dataset"] else "not_generated",
                "domain": config["global_sr"],
            },
        ],
    }
    if file_status["shared_split"]:
        with paths["shared_split"].open(encoding="utf-8") as file:
            split = json.load(file)
        if split.get("dataset_run") != run_id:
            raise ValueError(
                f"Shared split belongs to {split.get('dataset_run')!r}, not {run_id!r}."
            )
        if int(split.get("source_curve_count", -1)) != int(generation["curve_samples"]):
            raise ValueError("Shared split curve count does not match the dataset-run configuration.")
        manifest["shared_curve_split"].update(
            {
                "split_hash_algorithm": split.get("split_hash_algorithm"),
                "split_hash": split.get("split_hash"),
                "train_curves": len(split.get("train_curve_indices", [])),
                "validation_curves": len(split.get("validation_curve_indices", [])),
                "test_curves": len(split.get("test_curve_indices", [])),
            }
        )
    return manifest


def main() -> None:
    args = parse_args()
    config = load_dataset_run_config(args.dataset_run)
    manifest = build_manifest(config)
    if not args.check:
        path = config["paths"]["manifest"]
        with path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")
        print(f"Manifest synchronized: {path}")
    print(f"Dataset run {args.dataset_run}: {manifest['status']}")
    for item in (manifest["teacher_dataset"], manifest["shared_curve_split"], *manifest["derived_datasets"]):
        print(f"  {item['status']}: {item['path']}")


if __name__ == "__main__":
    main()
