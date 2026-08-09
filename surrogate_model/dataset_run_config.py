"""Central dataset-run configuration and path resolution.

Core model code reads this module instead of hard-coding run1/run2/run3.
Future runs need one datasets/<run>/dataset_config.json file plus an optional
change to dataset_run_config.json when the new run should become the default.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SURROGATE_ROOT = Path(__file__).resolve().parent
ACTIVE_CONFIG_FILE = SURROGATE_ROOT / "dataset_run_config.json"
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_dataset_run(dataset_run: str) -> str:
    dataset_run = str(dataset_run)
    if not RUN_NAME_PATTERN.fullmatch(dataset_run):
        raise ValueError(f"Invalid dataset run name: {dataset_run!r}")
    return dataset_run


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset-run configuration not found: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def active_config() -> dict[str, Any]:
    config = _read_json(ACTIVE_CONFIG_FILE)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported active dataset config schema: {ACTIVE_CONFIG_FILE}")
    return config


def default_dataset_run() -> str:
    return validate_dataset_run(active_config().get("default_dataset_run", ""))


def run_config_file(dataset_run: str) -> Path:
    dataset_run = validate_dataset_run(dataset_run)
    filename = str(active_config().get("run_config_filename", "dataset_config.json"))
    if Path(filename).name != filename:
        raise ValueError("run_config_filename must be a plain filename.")
    return SURROGATE_ROOT / "datasets" / dataset_run / filename


def load_dataset_run_config(dataset_run: str | None = None) -> dict[str, Any]:
    dataset_run = default_dataset_run() if dataset_run is None else validate_dataset_run(dataset_run)
    config_file = run_config_file(dataset_run)
    config = _read_json(config_file)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported dataset config schema: {config_file}")
    if config.get("run_id") != dataset_run:
        raise ValueError(
            f"Dataset config run_id {config.get('run_id')!r} does not match directory {dataset_run!r}."
        )

    target_schema = config.get("target_schema", {})
    expected_targets = ["R_real", "R_imag"]
    if target_schema.get("target_names") != expected_targets:
        raise ValueError(f"Dataset config target_names must be {expected_targets}.")
    if target_schema.get("complex_source") != "Reflect":
        raise ValueError("Dataset targets must be derived from the JCAL Reflect output.")
    if target_schema.get("target_keys") != {"re": "Y_re", "im": "Y_im"}:
        raise ValueError("Dataset target_keys must map re/im to Y_re/Y_im.")
    if target_schema.get("symbolic_target_keys") != {
        "re": "y_re_symbolic", "im": "y_im_symbolic"
    }:
        raise ValueError("Symbolic target keys must map re/im to paired symbolic arrays.")

    generation = config.get("generation", {})
    frequency = generation.get("frequency_grid_hz", {})
    required_generation = ("material", "porosity_cases", "sampling_method", "random_seed", "curve_samples")
    missing = [key for key in required_generation if key not in generation]
    missing += [f"frequency_grid_hz.{key}" for key in ("min", "max", "points") if key not in frequency]
    if missing:
        raise ValueError(f"Dataset config is missing required fields: {missing}")
    if float(frequency["min"]) >= float(frequency["max"]):
        raise ValueError("frequency_grid_hz.min must be smaller than frequency_grid_hz.max.")
    if int(frequency["points"]) < 2 or int(generation["curve_samples"]) < 1:
        raise ValueError("Frequency points must be >= 2 and curve_samples must be >= 1.")

    segments = config.get("segmented_sr", {}).get("segments", [])
    if not segments:
        raise ValueError("Dataset config must define at least one segmented_sr segment.")
    bounds = [item.get("bounds_hz", []) for item in segments]
    if any(len(item) != 2 or item[0] >= item[1] for item in bounds):
        raise ValueError("Every segmented_sr bound must contain an increasing [lower, upper] pair.")
    if bounds[0][0] != frequency["min"] or bounds[-1][1] != frequency["max"]:
        raise ValueError("Segment bounds must cover the complete configured frequency range.")
    if any(right[0] != left[1] for left, right in zip(bounds, bounds[1:])):
        raise ValueError("Segment bounds must be contiguous.")
    point_count = int(frequency["points"])
    step = (float(frequency["max"]) - float(frequency["min"])) / (point_count - 1)
    grid = [float(frequency["min"]) + index * step for index in range(point_count)]
    segment_counts = []
    for index, (lower, upper) in enumerate(bounds):
        if index < len(bounds) - 1:
            segment_counts.append(sum(lower <= value < upper for value in grid))
        else:
            segment_counts.append(sum(lower <= value <= upper for value in grid))
    if any(count == 0 for count in segment_counts):
        raise ValueError(f"Every segment must contain a configured frequency point; counts={segment_counts}")
    global_sr = config.get("global_sr", {})
    if global_sr.get("bounds_hz") != [frequency["min"], frequency["max"]] or not global_sr.get("name"):
        raise ValueError("global_sr must be named and cover the complete configured frequency range.")

    run_dir = SURROGATE_ROOT / "datasets" / dataset_run
    material = str(generation["material"])
    config["config_file"] = str(config_file.resolve())
    config["paths"] = {
        "run_dir": run_dir,
        "manifest": run_dir / "dataset_manifest.json",
        "teacher_dataset": run_dir / "MLP" / f"{material}_R.mat",
        "shared_split": run_dir / "shared_curve_split.json",
        "segmented_dataset": run_dir / "segmented_SR" / f"{material}_R_segmented.mat",
        "global_dataset": run_dir / "global_SR" / f"{material}_R_global.mat",
    }
    config["segment_frequency_point_counts"] = segment_counts
    return config


def run_id_from_dataset_path(dataset_file: Path) -> str | None:
    parts = dataset_file.resolve().parts
    for index, part in enumerate(parts[:-2]):
        if part.lower() == "datasets":
            return parts[index + 1]
    return None
