"""Production inference for the accepted run1 Global SR Re F3 / Im F2 pair.

The active model is a JSON bundle containing the frozen feature transforms,
frequency envelopes, equations, and provenance.  Loading or predicting never
fits statistics and never selects an equation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_ROOT = SCRIPT_DIR.parent
for import_path in (SCRIPT_DIR, SURROGATE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from f3_local_terminals import (  # noqa: E402
    apply_f3_local_terminals,
    validate_f3_local_terminal_spec,
)
from global_frequency_modulation import (  # noqa: E402
    apply_frequency_modulation,
    validate_frequency_modulation,
)
from global_sobol_feature_policy import validate_stored_feature_policy  # noqa: E402
from symbolic_feature_transform import apply_feature_transform  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_POINTER = SCRIPT_DIR / "formal_models" / "active_model.json"
EXPECTED_TARGET_MODELS = {"re": "Re_F3", "im": "Im_F2"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def resolve_model_manifest(path: Path | None = None) -> Path:
    """Resolve either an active-model pointer or a model manifest."""
    requested = (path or DEFAULT_POINTER).resolve()
    if not requested.exists():
        raise FileNotFoundError(f"Formal Global SR model file not found: {requested}")
    record = _load_json(requested)
    if "model_manifest" not in record:
        return requested
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported active-model pointer schema version.")
    if record.get("model_family") != "global_symbolic_regression":
        raise ValueError("Active-model pointer has the wrong model family.")
    if record.get("active_dataset_run") != "run1":
        raise ValueError("Active-model pointer must remain on dataset run1.")
    if record.get("active_targets") != EXPECTED_TARGET_MODELS:
        raise ValueError("Active-model pointer must remain Re_F3 / Im_F2.")
    manifest = (requested.parent / str(record["model_manifest"])).resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Active Global SR manifest not found: {manifest}")
    return manifest


def _build_predictor(expression: str, variable_names: list[str]) -> Any:
    symbols = sp.symbols(variable_names)
    local_dict = {name: symbol for name, symbol in zip(variable_names, symbols)}
    parsed = sp.sympify(expression, locals=local_dict)
    unexpected = {str(symbol) for symbol in parsed.free_symbols} - set(variable_names)
    if unexpected:
        raise ValueError(f"Equation references undeclared variables: {sorted(unexpected)}")
    return sp.lambdify(symbols, parsed, modules=["numpy"])


def _predict_expression(
    predictor: Any,
    X: np.ndarray,
    variable_names: list[str],
    label: str,
) -> np.ndarray:
    with np.errstate(all="ignore"):
        values = predictor(*[X[:, index] for index in range(X.shape[1])])
    if np.isscalar(values):
        prediction = np.full(X.shape[0], float(values))
    else:
        prediction = np.asarray(values, dtype=float).reshape(-1)
    if prediction.shape != (X.shape[0],):
        raise ValueError(f"{label} equation returned an unexpected prediction shape.")
    if not np.all(np.isfinite(prediction)):
        raise ValueError(f"{label} equation produced non-finite predictions.")
    return prediction


class FormalGlobalSymbolicModel:
    """Immutable inference wrapper for the accepted paired Global SR model."""

    def __init__(self, manifest: dict[str, Any], manifest_path: Path | None = None):
        self.manifest = deepcopy(manifest)
        self.manifest_path = manifest_path.resolve() if manifest_path else None
        self._validate_manifest()

        re_equation = self.manifest["targets"]["re"]["f1_equation"]
        residual = self.manifest["targets"]["re"]["f3_residual"]
        im_equation = self.manifest["targets"]["im"]["equation"]
        self._re_f1_predictor = _build_predictor(
            re_equation["expression_for_eval"], re_equation["variable_names"]
        )
        self._re_residual_predictor = _build_predictor(
            residual["expression_for_eval"], residual["variable_names"]
        )
        self._im_predictor = _build_predictor(
            im_equation["expression_for_eval"], im_equation["variable_names"]
        )

    @classmethod
    def load(cls, path: Path | str | None = None) -> "FormalGlobalSymbolicModel":
        manifest_path = resolve_model_manifest(None if path is None else Path(path))
        return cls(_load_json(manifest_path), manifest_path)

    @property
    def input_feature_names(self) -> list[str]:
        return list(self.manifest["input_contract"]["feature_names"])

    @property
    def split_hash(self) -> str:
        return str(self.manifest["dataset"]["shared_split_hash"])

    def serialize(self, destination: Path | str) -> Path:
        """Write a self-contained copy of the immutable model manifest."""
        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        _dump_json(destination, self.manifest)
        return destination

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported formal Global SR manifest schema version.")
        if manifest.get("model_family") != "global_symbolic_regression":
            raise ValueError("Formal model manifest has the wrong model family.")
        if manifest.get("dataset", {}).get("run_id") != "run1":
            raise ValueError("The accepted formal Global SR pair is defined for run1 only.")
        if manifest.get("output_postprocessing") != {"method": "none"}:
            raise ValueError("Formal Global SR predictions must remain raw and unclipped.")

        targets = manifest.get("targets", {})
        if set(targets) != set(EXPECTED_TARGET_MODELS):
            raise ValueError("Formal model manifest must contain independent Re and Im targets.")
        for target, expected_model in EXPECTED_TARGET_MODELS.items():
            if targets[target].get("active_model") != expected_model:
                raise ValueError(
                    f"Active {target} pointer must remain {expected_model}, found "
                    f"{targets[target].get('active_model')!r}."
                )
            if targets[target].get("output_postprocessing") != {"method": "none"}:
                raise ValueError(f"Formal {target} target may not clip or smooth predictions.")

        contract = manifest.get("input_contract", {})
        names = list(contract.get("feature_names", []))
        if names != [
            "phi",
            "h",
            "sigma",
            "alpha_infinity",
            "lambda",
            "lambda_prime",
            "k0_prime",
            "f",
        ]:
            raise ValueError("Formal Global SR raw input order changed.")
        lower, upper = [float(value) for value in contract["frequency_range_hz"]]
        if (lower, upper) != (100.0, 4950.0):
            raise ValueError("Formal Global SR frequency range changed.")

        transform = manifest["feature_transform"]
        transformed_names = list(transform["transformed_feature_names"])
        for target in ("re", "im"):
            target_spec = targets[target]
            policy = validate_stored_feature_policy(
                target_spec["feature_policy"], target, transformed_names
            )
            base_names = list(policy["feature_names"])
            modulation = validate_frequency_modulation(
                target_spec["frequency_modulation"], target, base_names
            )
            expected_names = [
                *base_names,
                *[feature["name"] for feature in modulation["features"]],
            ]
            equation_key = "f1_equation" if target == "re" else "equation"
            equation_names = list(target_spec[equation_key]["variable_names"])
            if equation_names != expected_names:
                raise ValueError(f"Formal {target} equation feature order changed.")

        re_spec = targets["re"]
        base_names = list(re_spec["feature_policy"]["feature_names"])
        f3_spec = validate_f3_local_terminal_spec(
            re_spec["f3_local_terminal_spec"], base_names
        )
        residual = re_spec["f3_residual"]
        if residual["variable_names"] != f3_spec["terminal_feature_names"]:
            raise ValueError("Formal Re F3 residual terminal order changed.")
        if residual.get("formula_safety_passed") is not True:
            raise ValueError("Formal Re F3 residual is not marked formula-safe.")
        if re_spec.get("formal_acceptance_decision") != (
            "accept_frozen_Re_F3_as_formal_Re_model"
        ):
            raise ValueError("Formal Re F3 acceptance decision is missing.")
        if targets["im"].get("validation_selected_candidate") != 14:
            raise ValueError("Formal Im F2 must remain validation-selected candidate 14.")

    def _prepare_raw_inputs(
        self,
        X: np.ndarray,
        feature_names: list[str] | None,
        allow_extrapolation: bool,
    ) -> np.ndarray:
        raw = np.asarray(X, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2 or raw.shape[1] != len(self.input_feature_names):
            raise ValueError(
                f"Expected an [n, {len(self.input_feature_names)}] raw feature matrix."
            )
        supplied_names = self.input_feature_names if feature_names is None else list(feature_names)
        if supplied_names != self.input_feature_names:
            raise ValueError("Raw input feature names/order do not match the formal model contract.")
        if not np.all(np.isfinite(raw)):
            raise ValueError("Formal Global SR inputs contain non-finite values.")

        name_to_index = {name: index for index, name in enumerate(self.input_feature_names)}
        for name, expected in self.manifest["input_contract"]["fixed_features"].items():
            values = raw[:, name_to_index[name]]
            if not np.allclose(values, float(expected), rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Feature {name!r} was constant at {expected} during training and may not vary."
                )

        if not allow_extrapolation:
            frequency = raw[:, name_to_index["f"]]
            lower, upper = self.manifest["input_contract"]["frequency_range_hz"]
            if np.any(frequency < float(lower)) or np.any(frequency > float(upper)):
                raise ValueError(
                    f"Frequency is outside the validated [{lower}, {upper}] Hz model range."
                )
        return raw

    def predict_with_diagnostics(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        *,
        allow_extrapolation: bool = False,
    ) -> dict[str, np.ndarray]:
        """Predict both components and expose frozen Re F3 replay diagnostics."""
        raw = self._prepare_raw_inputs(X, feature_names, allow_extrapolation)
        transformed = apply_feature_transform(
            raw, self.input_feature_names, self.manifest["feature_transform"]
        )
        transformed_names = list(
            self.manifest["feature_transform"]["transformed_feature_names"]
        )

        re_spec = self.manifest["targets"]["re"]
        re_policy = re_spec["feature_policy"]
        re_base = transformed[:, [int(index) for index in re_policy["feature_indices_0based"]]]
        re_base_names = list(re_policy["feature_names"])
        re_f1_features, re_f1_names = apply_frequency_modulation(
            re_base, re_base_names, re_spec["frequency_modulation"]
        )
        if re_f1_names != re_spec["f1_equation"]["variable_names"]:
            raise ValueError("Runtime Re F1 feature order differs from the frozen equation.")
        re_f1 = _predict_expression(
            self._re_f1_predictor, re_f1_features, re_f1_names, "Re F1"
        )

        re_f3_features, re_f3_names = apply_f3_local_terminals(
            re_base, re_base_names, re_spec["f3_local_terminal_spec"]
        )
        terminal_name = re_spec["f3_residual"]["variable_names"][0]
        terminal = re_f3_features[:, re_f3_names.index(terminal_name)].reshape(-1, 1)
        residual = _predict_expression(
            self._re_residual_predictor,
            terminal,
            [terminal_name],
            "Re F3 residual",
        )
        re_prediction = re_f1 + residual

        im_spec = self.manifest["targets"]["im"]
        im_policy = im_spec["feature_policy"]
        im_base = transformed[:, [int(index) for index in im_policy["feature_indices_0based"]]]
        im_base_names = list(im_policy["feature_names"])
        im_features, im_names = apply_frequency_modulation(
            im_base, im_base_names, im_spec["frequency_modulation"]
        )
        if im_names != im_spec["equation"]["variable_names"]:
            raise ValueError("Runtime Im F2 feature order differs from the frozen equation.")
        im_prediction = _predict_expression(
            self._im_predictor, im_features, im_names, "Im F2"
        )

        reflect = re_prediction + 1j * im_prediction
        if not np.all(np.isfinite(reflect)):
            raise ValueError("Combined formal Global SR prediction is non-finite.")
        return {
            "R_real": re_prediction,
            "R_imag": im_prediction,
            "Reflect": reflect,
            "Re_F1": re_f1,
            "Re_F3_residual": residual,
            terminal_name: terminal.reshape(-1),
        }

    def predict_components(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        *,
        allow_extrapolation: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict_with_diagnostics(
            X, feature_names, allow_extrapolation=allow_extrapolation
        )
        return prediction["R_real"], prediction["R_imag"]

    def predict_complex(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        *,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        return self.predict_with_diagnostics(
            X, feature_names, allow_extrapolation=allow_extrapolation
        )["Reflect"]


def _read_cli_input(path: Path, feature_names: list[str]) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != feature_names:
                raise ValueError(
                    "Input CSV columns/order must exactly match: " + ", ".join(feature_names)
                )
            rows = [[float(row[name]) for name in feature_names] for row in reader]
        return np.asarray(rows, dtype=float)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "X" not in archive:
                raise ValueError("Input NPZ must contain an X array.")
            return np.asarray(archive["X"], dtype=float)
    raise ValueError("Formal inference input must be .csv or .npz.")


def _write_cli_output(
    path: Path,
    raw: np.ndarray,
    feature_names: list[str],
    prediction: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fieldnames = [*feature_names, "R_real", "R_imag"]
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(raw.shape[0]):
                row = {name: raw[index, column] for column, name in enumerate(feature_names)}
                row["R_real"] = prediction["R_real"][index]
                row["R_imag"] = prediction["R_imag"][index]
                writer.writerow(row)
        return
    if path.suffix.lower() == ".npz":
        np.savez_compressed(
            path,
            X=raw,
            R_real=prediction["R_real"],
            R_imag=prediction["R_imag"],
            Reflect=prediction["Reflect"],
        )
        return
    raise ValueError("Formal inference output must be .csv or .npz.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw .csv or .npz inputs.")
    parser.add_argument("--output", type=Path, required=True, help="Prediction .csv or .npz.")
    parser.add_argument("--model", type=Path, default=DEFAULT_POINTER)
    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Allow frequency values outside 100-4950 Hz (disabled by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = FormalGlobalSymbolicModel.load(args.model)
    raw = _read_cli_input(args.input.resolve(), model.input_feature_names)
    prediction = model.predict_with_diagnostics(
        raw, allow_extrapolation=args.allow_extrapolation
    )
    _write_cli_output(
        args.output.resolve(), raw, model.input_feature_names, prediction
    )
    print(
        f"Formal Global SR inference complete: {raw.shape[0]} rows, "
        f"Re={model.manifest['targets']['re']['active_model']}, "
        f"Im={model.manifest['targets']['im']['active_model']}"
    )
    print(f"Predictions: {args.output.resolve()}")


if __name__ == "__main__":
    main()
