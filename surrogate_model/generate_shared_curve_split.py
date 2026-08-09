"""Generate a deterministic, frequency-grid-independent source-curve split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from shared_split_utils import compute_split_hash, default_split_file
from dataset_run_config import (
    default_dataset_run,
    load_dataset_run_config,
    run_id_from_dataset_path,
)


SURROGATE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", default=default_dataset_run())
    parser.add_argument("--teacher-dataset", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--validation-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_curve_inputs(dataset_file: Path) -> np.ndarray:
    if not dataset_file.exists():
        raise FileNotFoundError(f"Teacher dataset not found: {dataset_file}")
    try:
        with h5py.File(dataset_file, "r") as file:
            missing = [key for key in ("X", "Y_re", "Y_im", "freq_grid", "dataset_info") if key not in file]
            if missing:
                raise ValueError(f"Teacher dataset is missing paired-target variables {missing}: {dataset_file}")
            X = np.asarray(file["X"]).T
            y_re = np.asarray(file["Y_re"]).T
            y_im = np.asarray(file["Y_im"]).T
    except OSError:
        matlab_data = loadmat(dataset_file, variable_names=["X", "Y_re", "Y_im", "freq_grid", "dataset_info"])
        missing = [key for key in ("X", "Y_re", "Y_im", "freq_grid", "dataset_info") if key not in matlab_data]
        if missing:
            raise ValueError(f"Teacher dataset is missing paired-target variables {missing}: {dataset_file}")
        X = np.asarray(matlab_data["X"])
        y_re = np.asarray(matlab_data["Y_re"])
        y_im = np.asarray(matlab_data["Y_im"])
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError(f"Unexpected teacher X shape: {X.shape}")
    if y_re.ndim != 2 or y_re.shape != y_im.shape or y_re.shape[0] != X.shape[0]:
        raise ValueError(
            f"Teacher paired targets are not aligned: X={X.shape}, Y_re={y_re.shape}, Y_im={y_im.shape}"
        )
    if not np.isfinite(y_re).all() or not np.isfinite(y_im).all():
        raise ValueError("Teacher paired targets contain non-finite values.")
    return X


def stratified_curve_split(
    phi: np.ndarray,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.")

    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []

    for value in np.unique(phi):
        curve_ids = np.flatnonzero(phi == value) + 1
        curve_ids = rng.permutation(curve_ids)
        n_curves = curve_ids.size
        n_train = int(np.floor(ratios[0] * n_curves))
        n_validation = int(np.floor(ratios[1] * n_curves))
        n_test = n_curves - n_train - n_validation
        if min(n_train, n_validation, n_test) < 1:
            raise ValueError(
                f"Porosity group {value} is too small for the requested split ratios."
            )
        train.extend(curve_ids[:n_train].tolist())
        validation.extend(curve_ids[n_train : n_train + n_validation].tolist())
        test.extend(curve_ids[n_train + n_validation :].tolist())

    return (
        rng.permutation(np.asarray(train, dtype=int)),
        rng.permutation(np.asarray(validation, dtype=int)),
        rng.permutation(np.asarray(test, dtype=int)),
    )


def main() -> None:
    args = parse_args()
    run_config = load_dataset_run_config(args.dataset_run)
    split_config = run_config["shared_split"]
    teacher_dataset = args.teacher_dataset or run_config["paths"]["teacher_dataset"]
    output_file = args.output_file or run_config["paths"]["shared_split"]
    for label, path in (("teacher dataset", teacher_dataset), ("split output", output_file)):
        path_run = run_id_from_dataset_path(path)
        if path_run is not None and path_run != args.dataset_run:
            raise ValueError(
                f"Configured run is {args.dataset_run!r}, but the {label} belongs to "
                f"{path_run!r}: {path}"
            )
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing shared split: {output_file}. "
            "Create a new dataset run, or pass --overwrite explicitly."
        )
    split_seed = int(split_config["seed"] if args.split_seed is None else args.split_seed)
    train_ratio = float(split_config["train_ratio"] if args.train_ratio is None else args.train_ratio)
    validation_ratio = float(split_config["validation_ratio"] if args.validation_ratio is None else args.validation_ratio)
    test_ratio = float(split_config["test_ratio"] if args.test_ratio is None else args.test_ratio)
    X = read_curve_inputs(teacher_dataset)
    ratios = (train_ratio, validation_ratio, test_ratio)
    train, validation, test = stratified_curve_split(X[:, 0], ratios, split_seed)

    try:
        source_path = teacher_dataset.resolve().relative_to(SURROGATE_ROOT.parent).as_posix()
    except ValueError:
        source_path = str(teacher_dataset.resolve())

    split = {
        "schema_version": 1,
        "dataset_run": args.dataset_run,
        "source_teacher_dataset": source_path,
        "source_curve_count": int(X.shape[0]),
        "index_base": 1,
        "split_seed": split_seed,
        "split_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "split_strategy": "stratified_by_phi_then_shuffled",
        "stratification_feature": "phi",
        "frequency_grid_independent": True,
        "target_names": ["R_real", "R_imag"],
        "complex_source": "Reflect",
        "frequency_grid_note": (
            "The split is defined over source curves only. Frequency range and "
            "frequency count may change without changing curve membership."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_curve_indices": train.tolist(),
        "validation_curve_indices": validation.tolist(),
        "test_curve_indices": test.tolist(),
    }
    split["split_hash_algorithm"] = "sha256"
    split["split_hash"] = compute_split_hash(split)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(split, file, indent=2)
        file.write("\n")

    default_output = default_split_file(args.dataset_run).resolve()
    manifest_file = SURROGATE_ROOT / "datasets" / args.dataset_run / "dataset_manifest.json"
    if output_file.resolve() == default_output and manifest_file.exists():
        with manifest_file.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["shared_curve_split"] = {
            "path": output_file.resolve().relative_to(SURROGATE_ROOT.parent).as_posix(),
            "unit": "source_curve_index",
            "index_base": 1,
            "train_curves": int(train.size),
            "validation_curves": int(validation.size),
            "test_curves": int(test.size),
            "frequency_grid_independent": True,
            "split_hash_algorithm": "sha256",
            "split_hash": split["split_hash"],
        }
        with manifest_file.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")

    print(f"Shared curve split written to: {output_file.resolve()}")
    print(
        f"Curves: train={train.size}, validation={validation.size}, test={test.size}"
    )
    print(f"Split hash: {split['split_hash']}")


if __name__ == "__main__":
    main()
