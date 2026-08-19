"""Serializable local material-frequency terminals for Re Global SR F3."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
TARGET = "re"
FREQUENCY_FEATURE = "log10_f"
FREQUENCY_RANGE_HZ = (100.0, 4950.0)
REFERENCE_GRID_SIZE = 4096


def _logistic_highpass(
    log10_frequency: np.ndarray, center_hz: float, transition_log10: float
) -> np.ndarray:
    if center_hz <= 0.0 or transition_log10 <= 0.0:
        raise ValueError("Band-pass center and transition must be positive.")
    signed = (log10_frequency - np.log10(center_hz)) / transition_log10
    return 1.0 / (1.0 + np.exp(-np.clip(signed, -60.0, 60.0)))


def _raw_envelope(log10_frequency: np.ndarray, envelope: dict[str, Any]) -> np.ndarray:
    if envelope.get("kind") != "smooth_log10_bandpass_product":
        raise ValueError(f"Unsupported F3 envelope kind: {envelope.get('kind')!r}")
    highpass = _logistic_highpass(
        log10_frequency,
        float(envelope["low_hz"]),
        float(envelope["low_transition_log10"]),
    )
    lowpass = 1.0 - _logistic_highpass(
        log10_frequency,
        float(envelope["high_hz"]),
        float(envelope["high_transition_log10"]),
    )
    return highpass * lowpass


def reference_normalization(envelope: dict[str, Any]) -> float:
    reference_log10_frequency = np.linspace(
        np.log10(FREQUENCY_RANGE_HZ[0]),
        np.log10(FREQUENCY_RANGE_HZ[1]),
        REFERENCE_GRID_SIZE,
    )
    value = float(np.max(_raw_envelope(reference_log10_frequency, envelope)))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("F3 envelope has an invalid reference normalization.")
    return value


def build_re_f3b_sigma_spec(
    base_feature_names: list[str],
    *,
    sigma_center: float,
    sigma_scale: float,
    split_hash: str,
    source_f3a_dir: str,
) -> dict[str, Any]:
    """Build the frozen F3-B specification selected by F3-A."""
    envelope = {
        "name": "g_mid",
        "kind": "smooth_log10_bandpass_product",
        "frequency_coordinate": "log10_hz",
        "low_hz": 1250.0,
        "high_hz": 2100.0,
        "low_transition_log10": 0.06,
        "high_transition_log10": 0.06,
        "normalization_method": "maximum_on_fixed_log10_reference_grid",
        "normalization_frequency_range_hz": list(FREQUENCY_RANGE_HZ),
        "normalization_reference_grid_size": REFERENCE_GRID_SIZE,
    }
    envelope["normalization_max"] = reference_normalization(envelope)
    terminal_name = "z_sigma__f3_mid"
    specification = {
        "schema_version": SCHEMA_VERSION,
        "name": "re_f3_sigma_local_bandpass_v1",
        "stage_frozen": "F3-B",
        "source_evidence_stage": "F3-A",
        "source_f3a_dir": source_f3a_dir,
        "target": TARGET,
        "dataset_run": "run1",
        "shared_split_hash": split_hash,
        "frequency_range_hz": list(FREQUENCY_RANGE_HZ),
        "input_feature_names": list(base_feature_names),
        "base_features_retained": True,
        "frequency_feature": FREQUENCY_FEATURE,
        "standardizations": {
            "z_sigma": {
                "source_feature": "log10_sigma",
                "center": float(sigma_center),
                "scale": float(sigma_scale),
                "fitted_partition": "train",
                "fitted_unique_curve_count": 700,
                "method": "training_unique_curve_population_mean_std_ddof0",
                "ddof": 0,
            }
        },
        "envelopes": {"g_mid": envelope},
        "terminals": [
            {
                "name": terminal_name,
                "standardized_feature": "z_sigma",
                "envelope": "g_mid",
                "analytic_definition": "((log10_sigma - center) / scale) * g_mid(log10_f)",
            }
        ],
        "terminal_feature_names": [terminal_name],
        "output_feature_names": [*base_feature_names, terminal_name],
        "terminal_order_contract": "append_after_base_features_in_declared_order",
        "teacher_or_sobol_values_used_as_row_inputs": False,
        "output_postprocessing": {"method": "none"},
    }
    return validate_f3_local_terminal_spec(specification, base_feature_names)


def validate_f3_local_terminal_spec(
    specification: dict[str, Any], base_feature_names: list[str]
) -> dict[str, Any]:
    """Validate a stored F3 local-terminal specification before replay."""
    specification = deepcopy(specification)
    if specification.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported F3 local-terminal schema version.")
    if specification.get("target") != TARGET:
        raise ValueError("F3 local-terminal target must be Re.")
    if list(specification.get("input_feature_names", [])) != list(base_feature_names):
        raise ValueError("Stored F3 input feature order does not match the supplied base view.")
    if specification.get("frequency_feature") != FREQUENCY_FEATURE:
        raise ValueError("Stored F3 frequency feature is incompatible.")
    if FREQUENCY_FEATURE not in base_feature_names:
        raise ValueError("F3 base features omit log10 frequency.")

    standardizations = specification.get("standardizations", {})
    if set(standardizations) != {"z_sigma"}:
        raise ValueError("F3-B must freeze exactly the z_sigma standardization.")
    z_sigma = standardizations["z_sigma"]
    if z_sigma.get("source_feature") != "log10_sigma":
        raise ValueError("z_sigma must use log10_sigma.")
    if z_sigma.get("fitted_partition") != "train":
        raise ValueError("z_sigma normalization must be fitted on training curves only.")
    if float(z_sigma.get("scale", 0.0)) <= 0.0:
        raise ValueError("z_sigma scale must be positive.")
    if not np.isfinite(float(z_sigma["center"])) or not np.isfinite(float(z_sigma["scale"])):
        raise ValueError("z_sigma normalization contains non-finite constants.")

    envelopes = specification.get("envelopes", {})
    if set(envelopes) != {"g_mid"}:
        raise ValueError("F3-B must freeze exactly one g_mid envelope.")
    envelope = envelopes["g_mid"]
    if float(envelope["low_hz"]) != 1250.0 or float(envelope["high_hz"]) != 2100.0:
        raise ValueError("F3-B g_mid frequency bounds changed from the accepted F3-A branch.")
    if (
        float(envelope["low_transition_log10"]) != 0.06
        or float(envelope["high_transition_log10"]) != 0.06
    ):
        raise ValueError("F3-B g_mid transitions changed from the accepted F3-A branch.")
    stored_normalization = float(envelope.get("normalization_max", 0.0))
    recomputed_normalization = reference_normalization(envelope)
    if stored_normalization != recomputed_normalization:
        raise ValueError(
            "Stored g_mid normalization does not replay exactly on the declared reference grid."
        )

    terminals = list(specification.get("terminals", []))
    if len(terminals) != 1:
        raise ValueError("F3-B must contain exactly one sigma-local terminal.")
    terminal = terminals[0]
    if terminal.get("name") != "z_sigma__f3_mid":
        raise ValueError("Unexpected F3-B atomic terminal name.")
    if terminal.get("standardized_feature") != "z_sigma" or terminal.get("envelope") != "g_mid":
        raise ValueError("F3-B terminal references incompatible components.")
    expected_output_names = [*base_feature_names, "z_sigma__f3_mid"]
    if list(specification.get("terminal_feature_names", [])) != ["z_sigma__f3_mid"]:
        raise ValueError("F3-B terminal name order is invalid.")
    if list(specification.get("output_feature_names", [])) != expected_output_names:
        raise ValueError("F3-B output feature order is invalid.")
    return specification


def apply_f3_local_terminals(
    X: np.ndarray,
    base_feature_names: list[str],
    specification: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Append the frozen F3-B atomic terminal without fitting any statistics."""
    specification = validate_f3_local_terminal_spec(specification, base_feature_names)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(base_feature_names):
        raise ValueError("F3 input matrix does not match the declared base feature order.")
    if not np.all(np.isfinite(X)):
        raise ValueError("F3 input matrix contains non-finite values.")

    standardization = specification["standardizations"]["z_sigma"]
    sigma_index = base_feature_names.index(standardization["source_feature"])
    frequency_index = base_feature_names.index(specification["frequency_feature"])
    z_sigma = (X[:, sigma_index] - float(standardization["center"])) / float(
        standardization["scale"]
    )
    envelope_spec = specification["envelopes"]["g_mid"]
    envelope = _raw_envelope(X[:, frequency_index], envelope_spec)
    envelope /= float(envelope_spec["normalization_max"])
    terminal = z_sigma * envelope
    output = np.column_stack([X, terminal])
    if not np.all(np.isfinite(output)):
        raise ValueError("F3 output features contain non-finite values.")
    return output, list(specification["output_feature_names"])
