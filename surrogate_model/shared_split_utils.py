"""Shared source-curve split loading and validation utilities.

The split is intentionally defined only over source curve indices. Frequency
range, frequency count, and scalar-row count are not part of the split hash,
so a compatible teacher dataset can change its frequency grid without
changing the train/validation/test curve identities.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SURROGATE_ROOT = Path(__file__).resolve().parent
INDEX_KEYS = (
    "train_curve_indices",
    "validation_curve_indices",
    "test_curve_indices",
)


def default_split_file(dataset_run: str) -> Path:
    if dataset_run == "custom":
        raise ValueError("--split-file is required when --dataset-run=custom.")
    return SURROGATE_ROOT / "datasets" / dataset_run / "shared_curve_split.json"


def split_hash_payload(split: dict[str, Any]) -> dict[str, Any]:
    """Return the frequency-independent fields that uniquely define a split."""
    return {
        "schema_version": int(split["schema_version"]),
        "dataset_run": str(split["dataset_run"]),
        "source_curve_count": int(split["source_curve_count"]),
        "index_base": int(split["index_base"]),
        **{key: [int(value) for value in split[key]] for key in INDEX_KEYS},
    }


def compute_split_hash(split: dict[str, Any]) -> str:
    canonical = json.dumps(
        split_hash_payload(split), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_shared_split(
    split_file: Path,
    *,
    expected_dataset_run: str | None = None,
    expected_curve_count: int | None = None,
) -> dict[str, Any]:
    split_file = split_file.resolve()
    if not split_file.exists():
        raise FileNotFoundError(f"Shared curve split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as file:
        split = json.load(file)

    missing = [
        key
        for key in (
            "schema_version",
            "dataset_run",
            "source_curve_count",
            "index_base",
            "split_hash",
            *INDEX_KEYS,
        )
        if key not in split
    ]
    if missing:
        raise ValueError(f"Shared split is missing required fields: {missing}")
    if int(split["schema_version"]) != 1:
        raise ValueError(f"Unsupported shared split schema: {split['schema_version']}")
    if int(split["index_base"]) != 1:
        raise ValueError("Shared curve indices must be 1-based.")
    if expected_dataset_run not in (None, "custom") and split["dataset_run"] != expected_dataset_run:
        raise ValueError(
            f"Shared split run {split['dataset_run']!r} does not match "
            f"dataset run {expected_dataset_run!r}."
        )

    curve_count = int(split["source_curve_count"])
    if expected_curve_count is not None and curve_count != expected_curve_count:
        raise ValueError(
            f"Shared split expects {curve_count} source curves, but the dataset "
            f"contains {expected_curve_count}. Regenerate the split for this curve set."
        )

    arrays = {
        key: np.asarray(split[key], dtype=int).reshape(-1)
        for key in INDEX_KEYS
    }
    combined = np.concatenate(list(arrays.values()))
    expected = np.arange(1, curve_count + 1, dtype=int)
    if combined.size != curve_count or not np.array_equal(np.sort(combined), expected):
        raise ValueError(
            "Shared split indices must be disjoint and cover every 1-based source curve exactly once."
        )

    actual_hash = compute_split_hash(split)
    if split["split_hash"] != actual_hash:
        raise ValueError(
            f"Shared split hash mismatch: stored={split['split_hash']}, computed={actual_hash}."
        )

    split["split_file"] = str(split_file)
    split["split_hash"] = actual_hash
    split.update(arrays)
    return split


def resolve_and_load_shared_split(
    dataset_run: str,
    split_file: Path | None,
    expected_curve_count: int,
) -> dict[str, Any]:
    resolved_file = default_split_file(dataset_run) if split_file is None else split_file
    return load_shared_split(
        resolved_file,
        expected_dataset_run=dataset_run,
        expected_curve_count=expected_curve_count,
    )


def source_curve_mask(
    source_curve_index: np.ndarray,
    split: dict[str, Any],
    split_name: str,
) -> np.ndarray:
    key = f"{split_name}_curve_indices"
    if key not in INDEX_KEYS:
        raise ValueError(f"Unknown split name: {split_name}")
    return np.isin(source_curve_index, split[key])


def validate_training_split(training_dir: Path, split: dict[str, Any]) -> None:
    metadata_file = training_dir / "training_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Training metadata not found: {metadata_file}")
    with metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    training_hash = metadata.get("shared_split_hash")
    if training_hash is None:
        raise ValueError(
            f"Training run has no shared_split_hash and predates the shared-split protocol: "
            f"{training_dir}"
        )
    if training_hash != split["split_hash"]:
        raise ValueError(
            "Training/evaluation split mismatch: "
            f"training={training_hash}, evaluation={split['split_hash']}."
        )


def subset_symbolic_data(
    data: dict[str, Any], source_mask: np.ndarray
) -> dict[str, Any]:
    result = data.copy()
    for key in ("X", "y", "segment_index", "source_curve_index"):
        if key in result:
            result[key] = result[key][source_mask]
    return result
