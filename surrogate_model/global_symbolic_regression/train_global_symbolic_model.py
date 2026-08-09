"""Train one PySR model over the full-frequency global symbolic dataset.

The script is intentionally dataset-size agnostic: the same entry point can
run a quick 50-curve pipeline test or a later larger teacher dataset.
"""

from __future__ import annotations

import argparse
import json
import keyword
import math
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from global_run_paths import default_dataset_run, resolve_dataset_file


SURROGATE_ROOT = Path(__file__).resolve().parents[1]
if str(SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURROGATE_ROOT))

from shared_split_utils import (  # noqa: E402
    resolve_and_load_shared_split,
    source_curve_mask,
)


SCRIPT_DIR = Path(__file__).resolve().parent
VENV_JULIA_EXE = (
    Path(sys.executable).resolve().parent.parent
    / "julia_env"
    / "pyjuliapkg"
    / "install"
    / "bin"
    / "julia.exe"
)
USER_JULIA_EXE = (
    Path.home()
    / ".julia"
    / "juliaup"
    / "julia-1.11.9+0.x64.w64.mingw32"
    / "bin"
    / "julia.exe"
)
DEFAULT_JULIA_EXE = VENV_JULIA_EXE if VENV_JULIA_EXE.exists() else USER_JULIA_EXE
RESERVED_VARIABLE_NAMES = {
    "lambda",
    "Lambda",
    "I",
    "E",
    "pi",
    "oo",
    "nan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one PySR symbolic-regression model over the full frequency range."
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=("re", "im"),
        help="Reflection-coefficient component to train.",
    )
    parser.add_argument(
        "--dataset-run",
        default=default_dataset_run(),
        help="Dataset run under surrogate_model/datasets (default comes from the central dataset_run_config.json).",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Optional explicit dataset path. Standard datasets/run*/ paths must match --dataset-run.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Shared source-curve split JSON. Defaults to datasets/<run>/shared_curve_split.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Parent directory for automatically named training runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory. If omitted, a unique run directory is created under --output-root.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional readable suffix for the automatically created run directory.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick pipeline tests. Use all rows when omitted.",
    )
    parser.add_argument("--niterations", type=int, default=40)
    parser.add_argument("--populations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--maxsize", type=int, default=24)
    parser.add_argument("--model-selection", choices=["best", "accuracy", "score"], default="best")
    parser.add_argument(
        "--procs",
        type=int,
        default=0,
        help="Deprecated compatibility option. Global training is serial, so this value is ignored.",
    )
    parser.add_argument("--julia-exe", type=Path, default=DEFAULT_JULIA_EXE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite PySR temporary equation files from previous runs.",
    )
    return parser.parse_args()


def unique_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    for suffix in range(2, 1000):
        candidate = base_path.with_name(f"{base_path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create a unique output path for base path: {base_path}")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir

    output_root = args.output_root or SCRIPT_DIR / "artifacts" / args.target / "train"
    timestamp = datetime.now().strftime("%Y%m%d")
    suffix = "_limited" if args.max_samples is not None else ""
    return unique_path(output_root / f"{timestamp}_{sanitize_name(args.dataset_run)}{suffix}")


def validate_output_path_length(output_dir: Path, global_name: str) -> None:
    if os.name != "nt":
        return
    longest_path = (
        output_dir
        / "pysr_runs"
        / sanitize_name(global_name)
        / sanitize_name(global_name)
        / "hall_of_fame.csv"
    )
    if len(str(longest_path.resolve())) > 240:
        raise ValueError(
            "The PySR output path is too long for reliable Windows writes "
            f"({len(str(longest_path.resolve()))} characters): {longest_path}. "
            "Use --output-dir with a shorter path."
        )


def decode_matlab_string(file: h5py.File, value: Any) -> str:
    arr = np.array(value)
    if arr.dtype == object:
        if arr.size == 0:
            return ""
        return decode_matlab_string(file, file[arr.flat[0]])
    if h5py.check_dtype(ref=arr.dtype) is not None:
        if arr.size == 0:
            return ""
        return decode_matlab_string(file, file[arr.flat[0]])
    if np.issubdtype(arr.dtype, np.integer):
        return "".join(chr(int(code)) for code in arr.flatten() if int(code) != 0)
    if arr.dtype.kind in {"S", "U"}:
        return str(arr.flatten()[0])
    return str(arr)


def decode_matlab_string_array(file: h5py.File, dataset: h5py.Dataset) -> list[str]:
    values: list[str] = []
    refs = np.array(dataset)
    for ref in refs.flatten(order="F"):
        values.append(decode_matlab_string(file, file[ref]))
    return values


def read_numeric_dataset(dataset: h5py.Dataset) -> np.ndarray:
    return np.array(dataset).T


def read_symbolic_dataset(dataset_file: Path, target: str, dataset_run: str) -> dict[str, Any]:
    if not dataset_file.exists():
        raise FileNotFoundError(f"Symbolic dataset file not found: {dataset_file}")

    with h5py.File(dataset_file, "r") as file:
        missing_targets = [key for key in ("y_re_symbolic", "y_im_symbolic") if key not in file]
        if missing_targets:
            raise ValueError(f"Symbolic dataset is missing paired target arrays {missing_targets}.")
        x_symbolic = read_numeric_dataset(file["X_symbolic"])
        target_key = {"re": "y_re_symbolic", "im": "y_im_symbolic"}[target]
        if target_key not in file:
            raise ValueError(f"Symbolic dataset is missing target array {target_key}.")
        y_symbolic = np.array(file[target_key]).reshape(-1)
        paired_shape = np.array(file["y_re_symbolic"]).size, np.array(file["y_im_symbolic"]).size
        segment_index = np.array(file["segment_index"]).reshape(-1).astype(int)
        source_curve_index = np.array(file["source_curve_index"]).reshape(-1).astype(int)

        info_group = file["symbolic_dataset_info"]
        feature_names = decode_matlab_string_array(file, info_group["feature_names"])
        segment_bounds = np.array(info_group["segment_bounds_hz"]).T
        segment_names = decode_matlab_string_array(file, info_group["segment_names"])
        target_names = decode_matlab_string_array(file, info_group["target_names"])
        complex_source = decode_matlab_string(file, info_group["complex_source"])
        recorded_run = decode_matlab_string(file, info_group["dataset_run"])

        fiberfolder = decode_matlab_string(file, info_group["fiberfolder"])
        num_curve_samples = int(np.array(info_group["num_curve_samples"]).reshape(-1)[0])
        num_symbolic_samples = int(np.array(info_group["num_symbolic_samples"]).reshape(-1)[0])

    if x_symbolic.shape[0] != y_symbolic.shape[0]:
        raise ValueError("X_symbolic row count does not match y_symbolic length.")
    if paired_shape != (x_symbolic.shape[0], x_symbolic.shape[0]):
        raise ValueError("Paired symbolic target arrays are not aligned with X_symbolic.")
    if x_symbolic.shape[0] != segment_index.shape[0]:
        raise ValueError("X_symbolic row count does not match segment_index length.")
    if x_symbolic.shape[0] != source_curve_index.shape[0]:
        raise ValueError("X_symbolic row count does not match source_curve_index length.")
    if target_names != ["R_real", "R_imag"] or complex_source != "Reflect":
        raise ValueError("Symbolic dataset metadata does not match paired Reflect targets.")
    if dataset_run != "custom" and recorded_run != dataset_run:
        raise ValueError(f"Dataset metadata run {recorded_run!r} does not match {dataset_run!r}.")

    unique_indices = np.unique(segment_index)
    if unique_indices.tolist() != [1] or len(segment_names) != 1 or segment_bounds.shape != (1, 2):
        raise ValueError(
            "Global symbolic dataset must contain exactly one full-range domain "
            "encoded as segment_index=1."
        )

    return {
        "X": x_symbolic,
        "y": y_symbolic,
        "target": target,
        "target_name": {"re": "R_real", "im": "R_imag"}[target],
        "segment_index": segment_index,
        "source_curve_index": source_curve_index,
        "feature_names": feature_names,
        "segment_bounds": segment_bounds,
        "segment_names": segment_names,
        "fiberfolder": fiberfolder,
        "num_curve_samples": num_curve_samples,
        "num_symbolic_samples": num_symbolic_samples,
    }


def sanitize_name(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in safe.split("_") if part)


def make_pysr_variable_names(feature_names: list[str]) -> list[str]:
    variable_names: list[str] = []
    used: set[str] = set()

    for feature_name in feature_names:
        candidate = "".join(ch if ch.isalnum() else "_" for ch in feature_name).strip("_")
        if not candidate:
            candidate = "x"
        if candidate[0].isdigit():
            candidate = f"x_{candidate}"
        if keyword.iskeyword(candidate) or candidate in RESERVED_VARIABLE_NAMES:
            candidate = f"{candidate}_var"

        base = candidate
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1

        used.add(candidate)
        variable_names.append(candidate)

    return variable_names


def build_model(args: argparse.Namespace, pysr_output_dir: Path, run_id: str) -> Any:
    from pysr import PySRRegressor

    model_kwargs: dict[str, Any] = {
        "niterations": args.niterations,
        "populations": args.populations,
        "population_size": args.population_size,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["log", "sqrt"],
        "maxsize": args.maxsize,
        "model_selection": args.model_selection,
        "random_state": args.random_seed,
        "deterministic": True,
        "parallelism": "serial",
        "progress": False,
        "verbosity": 1,
        "temp_equation_file": False,
        "output_directory": pysr_output_dir.as_posix(),
        "run_id": run_id,
    }
    return PySRRegressor(**model_kwargs)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    max_abs_error = float(np.max(np.abs(y_true - y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "max_abs_error": max_abs_error,
        "r2": float(r2),
        "true_min": float(np.min(y_true)),
        "true_max": float(np.max(y_true)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
    }


def train_global_model(
    args: argparse.Namespace,
    data: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    global_name = data["segment_names"][0]
    bounds = data["segment_bounds"][0]
    train_indices = np.flatnonzero(
        source_curve_mask(data["source_curve_index"], data["shared_split"], "train")
    )
    validation_indices = np.flatnonzero(
        source_curve_mask(data["source_curve_index"], data["shared_split"], "validation")
    )
    test_indices = np.flatnonzero(
        source_curve_mask(data["source_curve_index"], data["shared_split"], "test")
    )

    rng = np.random.default_rng(args.random_seed)
    if args.max_samples is not None and train_indices.size > args.max_samples:
        train_indices = rng.choice(train_indices, size=args.max_samples, replace=False)

    if min(train_indices.size, validation_indices.size, test_indices.size) == 0:
        raise ValueError("Global symbolic dataset has an empty shared split partition.")

    X_train = data["X"][train_indices, :]
    y_train = data["y"][train_indices]
    X_validation = data["X"][validation_indices, :]
    y_validation = data["y"][validation_indices]
    X_test = data["X"][test_indices, :]
    y_test = data["y"][test_indices]

    global_slug = sanitize_name(global_name)
    equations_dir = output_dir / "equations"
    models_dir = output_dir / "models"
    pysr_runs_dir = output_dir / "pysr_runs" / global_slug
    equations_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    pysr_runs_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nTraining global model: {global_name} "
        f"({bounds[0]:.2f}-{bounds[1]:.2f} Hz), "
        f"train/validation/test rows="
        f"{len(train_indices)}/{len(validation_indices)}/{len(test_indices)}"
    )
    model = build_model(args, pysr_runs_dir, global_slug)
    model.fit(X_train, y_train, variable_names=data["pysr_variable_names"])

    train_pred = model.predict(X_train)
    validation_pred = model.predict(X_validation)
    test_pred = model.predict(X_test)
    selected = model.get_best()

    equations = model.equations_.copy()
    equations_csv = equations_dir / f"{global_slug}_equations.csv"
    equations.to_csv(equations_csv, index=False)

    model_path = models_dir / f"{global_slug}_pysr_model.pkl"
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    selected_equation = {
        "global_name": global_name,
        "target": data["target"],
        "target_name": data["target_name"],
        "lower_hz": float(bounds[0]),
        "upper_hz": float(bounds[1]),
        "shared_split_file": data["shared_split"]["split_file"],
        "shared_split_hash": data["shared_split"]["split_hash"],
        "equation": str(selected["equation"]),
        "complexity": int(selected["complexity"]),
        "loss": float(selected["loss"]),
        "score": float(selected["score"]) if "score" in selected and pd.notna(selected["score"]) else None,
        "n_rows_used": int(
            len(train_indices) + len(validation_indices) + len(test_indices)
        ),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "n_test": int(len(y_test)),
        "train_metrics": regression_metrics(y_train, train_pred),
        "validation_metrics": regression_metrics(y_validation, validation_pred),
        "test_metrics": regression_metrics(y_test, test_pred),
        "equations_csv": str(equations_csv),
        "model_file": str(model_path),
    }

    with (output_dir / f"{global_slug}_best_equation.json").open("w", encoding="utf-8") as file:
        json.dump(selected_equation, file, indent=2)

    return selected_equation


def main() -> None:
    args = parse_args()
    args.dataset_file, resolved_dataset_run = resolve_dataset_file(
        args.dataset_run, args.dataset_file
    )
    args.dataset_run = resolved_dataset_run

    if args.julia_exe.exists():
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", str(args.julia_exe))

    args.output_dir = resolve_output_dir(args)
    data = read_symbolic_dataset(args.dataset_file, args.target, resolved_dataset_run)
    data["shared_split"] = resolve_and_load_shared_split(
        resolved_dataset_run, args.split_file, data["num_curve_samples"]
    )
    validate_output_path_length(args.output_dir, data["segment_names"][0])
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty training directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data["pysr_variable_names"] = make_pysr_variable_names(data["feature_names"])

    metadata = {
        "dataset_file": str(args.dataset_file),
        "dataset_run": resolved_dataset_run,
        "target": args.target,
        "target_name": data["target_name"],
        "shared_split_file": data["shared_split"]["split_file"],
        "shared_split_hash": data["shared_split"]["split_hash"],
        "fiberfolder": data["fiberfolder"],
        "num_curve_samples": data["num_curve_samples"],
        "num_symbolic_samples": data["num_symbolic_samples"],
        "feature_names": data["feature_names"],
        "pysr_variable_names": data["pysr_variable_names"],
        "feature_name_mapping": dict(zip(data["feature_names"], data["pysr_variable_names"])),
        "global_name": data["segment_names"][0],
        "global_bounds_hz": data["segment_bounds"][0].tolist(),
        "training_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with (args.output_dir / "training_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    summary = train_global_model(args, data, args.output_dir)

    summary_df = pd.DataFrame(
        [
            {
                "global_name": item["global_name"],
                "lower_hz": item["lower_hz"],
                "upper_hz": item["upper_hz"],
                "equation": item["equation"],
                "complexity": item["complexity"],
                "loss": item["loss"],
                "train_rmse": item["train_metrics"]["rmse"],
                "validation_rmse": item["validation_metrics"]["rmse"],
                "test_rmse": item["test_metrics"]["rmse"],
                "train_mae": item["train_metrics"]["mae"],
                "validation_mae": item["validation_metrics"]["mae"],
                "test_mae": item["test_metrics"]["mae"],
                "test_r2": item["test_metrics"]["r2"],
                "n_rows_used": item["n_rows_used"],
            }
            for item in [summary]
        ]
    )
    summary_df.to_csv(args.output_dir / "global_model_summary.csv", index=False)
    with (args.output_dir / "global_model_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nGlobal PySR training complete.")
    print(summary_df[["global_name", "complexity", "test_rmse", "test_mae", "test_r2"]].to_string(index=False))
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
