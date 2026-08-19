"""Smooth teacher-Sobol-guided frequency modulation for Global SR inputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
FREQUENCY_FEATURE = "log10_f"
FREQUENCY_RANGE_HZ = (100.0, 4950.0)

# Every base material feature remains available.  Each additional feature is
# base_feature * envelope(log10_f).  Envelopes use smooth generic functions;
# teacher Sobol values determine only qualitative locations and relative lobes,
# and are never passed to the regressor as input values.
MODULATION_COMPONENTS: dict[str, dict[str, list[dict[str, float | str]]]] = {
    "re": {
        "log10_sigma": [
            {"kind": "gaussian", "center_hz": 1437.0, "width_log10": 0.366, "weight": 1.0},
        ],
        "alpha_infinity": [
            {"kind": "gaussian", "center_hz": 3422.0, "width_log10": 0.220, "weight": 1.0},
        ],
        "log10_lambda": [
            {"kind": "highpass", "center_hz": 2200.0, "transition_log10": 0.140, "weight": 1.0},
        ],
        "log10_lambda_prime": [
            {"kind": "lowpass", "center_hz": 700.0, "transition_log10": 0.120, "weight": 0.30},
            {"kind": "highpass", "center_hz": 3800.0, "transition_log10": 0.100, "weight": 1.00},
        ],
        "log10_k0_prime": [
            {"kind": "lowpass", "center_hz": 850.0, "transition_log10": 0.120, "weight": 1.00},
            {"kind": "highpass", "center_hz": 3400.0, "transition_log10": 0.120, "weight": 0.65},
        ],
    },
    "im": {
        "log10_sigma": [
            {"kind": "lowpass", "center_hz": 1200.0, "transition_log10": 0.120, "weight": 1.00},
            {"kind": "gaussian", "center_hz": 3000.0, "width_log10": 0.250, "weight": 0.90},
        ],
        "alpha_infinity": [
            {"kind": "gaussian", "center_hz": 1550.0, "width_log10": 0.160, "weight": 0.75},
            {"kind": "highpass", "center_hz": 3700.0, "transition_log10": 0.110, "weight": 1.00},
        ],
        "log10_lambda": [
            {"kind": "gaussian", "center_hz": 1500.0, "width_log10": 0.180, "weight": 0.70},
            {"kind": "highpass", "center_hz": 3900.0, "transition_log10": 0.110, "weight": 1.00},
        ],
        "log10_lambda_prime": [
            {"kind": "lowpass", "center_hz": 700.0, "transition_log10": 0.130, "weight": 0.35},
            {"kind": "highpass", "center_hz": 4000.0, "transition_log10": 0.100, "weight": 1.00},
        ],
        "log10_k0_prime": [
            {"kind": "lowpass", "center_hz": 750.0, "transition_log10": 0.120, "weight": 1.00},
            {"kind": "highpass", "center_hz": 3900.0, "transition_log10": 0.100, "weight": 0.40},
        ],
    },
}


def _component_values(log10_f: np.ndarray, component: dict[str, Any]) -> np.ndarray:
    center = np.log10(float(component["center_hz"]))
    kind = component["kind"]
    if kind == "gaussian":
        width = float(component["width_log10"])
        if width <= 0:
            raise ValueError("Gaussian modulation width must be positive.")
        values = np.exp(-0.5 * ((log10_f - center) / width) ** 2)
    elif kind in {"lowpass", "highpass"}:
        transition = float(component["transition_log10"])
        if transition <= 0:
            raise ValueError("Logistic modulation transition must be positive.")
        signed = (log10_f - center) / transition
        signed = np.clip(signed, -60.0, 60.0)
        highpass = 1.0 / (1.0 + np.exp(-signed))
        values = highpass if kind == "highpass" else 1.0 - highpass
    else:
        raise ValueError(f"Unsupported modulation component kind: {kind!r}")
    return float(component["weight"]) * values


def _raw_envelope(log10_f: np.ndarray, components: list[dict[str, Any]]) -> np.ndarray:
    envelope = np.zeros_like(log10_f, dtype=float)
    for component in components:
        envelope += _component_values(log10_f, component)
    return envelope


def build_frequency_modulation(target: str, base_feature_names: list[str]) -> dict[str, Any]:
    """Create a reproducible five-parameter modulation specification."""
    if target not in MODULATION_COMPONENTS:
        raise ValueError(f"Unsupported modulation target: {target!r}")
    required = [FREQUENCY_FEATURE, *MODULATION_COMPONENTS[target]]
    missing = sorted(set(required) - set(base_feature_names))
    if missing:
        raise ValueError(f"Frequency modulation requires unavailable features: {missing}")

    reference_log_f = np.linspace(
        np.log10(FREQUENCY_RANGE_HZ[0]), np.log10(FREQUENCY_RANGE_HZ[1]), 4096
    )
    features: list[dict[str, Any]] = []
    for base_feature, components in MODULATION_COMPONENTS[target].items():
        normalizer = float(np.max(_raw_envelope(reference_log_f, components)))
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError(f"Invalid modulation normalizer for {base_feature}.")
        features.append(
            {
                "name": f"{base_feature}__sobol_mod",
                "base_feature": base_feature,
                "components": deepcopy(components),
                "normalization_max": normalizer,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "teacher_sobol_smooth_parameter_envelopes",
        "source": "teacher Sobol run1 N=1024 frequency and segment ST diagnostics",
        "target": target,
        "frequency_feature": FREQUENCY_FEATURE,
        "frequency_range_hz": list(FREQUENCY_RANGE_HZ),
        "base_features_retained": True,
        "teacher_st_values_used_as_model_inputs": False,
        "features": features,
    }


def build_custom_frequency_modulation(
    target: str,
    name: str,
    feature_components: list[dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    """Build a persisted target-specific modulation specification."""
    reference_log_f = np.linspace(
        np.log10(FREQUENCY_RANGE_HZ[0]), np.log10(FREQUENCY_RANGE_HZ[1]), 4096
    )
    features: list[dict[str, Any]] = []
    for definition in feature_components:
        components = deepcopy(list(definition["components"]))
        normalizer = float(np.max(_raw_envelope(reference_log_f, components)))
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError(f"Invalid modulation normalizer for {definition['name']}.")
        features.append(
            {
                "name": str(definition["name"]),
                "base_feature": str(definition["base_feature"]),
                "components": components,
                "normalization_max": normalizer,
            }
        )
    specification = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "source": source,
        "target": target,
        "frequency_feature": FREQUENCY_FEATURE,
        "frequency_range_hz": list(FREQUENCY_RANGE_HZ),
        "base_features_retained": True,
        "teacher_st_values_used_as_model_inputs": False,
        "features": features,
    }
    base_feature_names = [FREQUENCY_FEATURE, *sorted({
        feature["base_feature"] for feature in features
    })]
    return validate_frequency_modulation(specification, target, base_feature_names)


def validate_frequency_modulation(
    specification: dict[str, Any], target: str, base_feature_names: list[str]
) -> dict[str, Any]:
    """Validate a persisted modulation specification before evaluation."""
    if specification.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Global frequency-modulation schema version.")
    if specification.get("target") != target:
        raise ValueError("Frequency-modulation target does not match evaluation target.")
    if specification.get("frequency_feature") not in base_feature_names:
        raise ValueError("Stored modulation frequency feature is unavailable.")
    features = list(specification.get("features", []))
    if not features:
        raise ValueError("Stored modulation does not contain any modulated features.")
    feature_names = [feature.get("name") for feature in features]
    if any(not name for name in feature_names) or len(feature_names) != len(set(feature_names)):
        raise ValueError("Stored modulation feature names must be non-empty and unique.")
    if specification.get("name") == "teacher_sobol_smooth_parameter_envelopes":
        expected_bases = set(MODULATION_COMPONENTS[target])
        stored_bases = {feature.get("base_feature") for feature in features}
        if stored_bases != expected_bases or len(features) != len(expected_bases):
            raise ValueError("Stored F1 modulation does not contain exactly one feature per parameter.")
    for feature in features:
        if feature["base_feature"] not in base_feature_names:
            raise ValueError(f"Unavailable modulation base feature: {feature['base_feature']}")
        if float(feature.get("normalization_max", 0.0)) <= 0:
            raise ValueError("Stored modulation normalization must be positive.")
        components = list(feature.get("components", []))
        if not components:
            raise ValueError(f"Modulation feature has no envelope components: {feature['name']}")
        reference_log_f = np.linspace(
            np.log10(FREQUENCY_RANGE_HZ[0]), np.log10(FREQUENCY_RANGE_HZ[1]), 4096
        )
        values = _raw_envelope(reference_log_f, components)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite stored envelope: {feature['name']}")
    return specification


def apply_frequency_modulation(
    X: np.ndarray, base_feature_names: list[str], specification: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    """Append one smooth frequency-modulated column for every material input."""
    frequency_index = base_feature_names.index(specification["frequency_feature"])
    log10_f = X[:, frequency_index]
    columns: list[np.ndarray] = [X]
    names = list(base_feature_names)
    for feature in specification["features"]:
        base_index = base_feature_names.index(feature["base_feature"])
        envelope = _raw_envelope(log10_f, feature["components"])
        envelope /= float(feature["normalization_max"])
        modulated = X[:, base_index] * envelope
        if not np.all(np.isfinite(modulated)):
            raise ValueError(f"Non-finite modulated values for {feature['name']}.")
        columns.append(modulated.reshape(-1, 1))
        names.append(feature["name"])
    return np.column_stack(columns), names
