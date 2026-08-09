"""Shared dataset-run resolution for segmented symbolic-regression entry points."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_ROOT = SCRIPT_DIR.parent
if str(SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURROGATE_ROOT))

from dataset_run_config import (  # noqa: E402
    default_dataset_run,
    load_dataset_run_config,
    run_id_from_dataset_path,
    validate_dataset_run,
)


def segmented_dataset_path(dataset_run: str) -> Path:
    return load_dataset_run_config(dataset_run)["paths"]["segmented_dataset"]


def resolve_dataset_file(dataset_run: str, dataset_file: Path | None) -> tuple[Path, str]:
    dataset_run = validate_dataset_run(dataset_run)
    resolved_file = segmented_dataset_path(dataset_run) if dataset_file is None else dataset_file
    path_run = run_id_from_dataset_path(resolved_file)
    if path_run is not None and path_run != dataset_run:
        raise ValueError(
            f"Dataset run mismatch: --dataset-run={dataset_run!r}, but the dataset path belongs "
            f"to {path_run!r}: {resolved_file}"
        )
    return resolved_file, dataset_run if path_run is not None else "custom"


def training_dataset_run(training_dir: Path) -> str | None:
    metadata_file = training_dir / "training_metadata.json"
    if not metadata_file.exists():
        return None
    with metadata_file.open(encoding="utf-8") as file:
        metadata = json.load(file)
    recorded_run = metadata.get("dataset_run")
    if recorded_run:
        return str(recorded_run)
    dataset_file = metadata.get("dataset_file")
    if not dataset_file:
        return None
    dataset_path = Path(dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = SURROGATE_ROOT.parent / dataset_path
    return run_id_from_dataset_path(dataset_path)
