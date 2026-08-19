"""Reproducible local-frequency features for segmented SR experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


ENGINEERING_SCHEMA_VERSION = 1
LOCAL_FREQUENCY_FEATURES = ("local_t", "local_t2", "local_t3")


def build_local_frequency_spec(
    segment_names: list[str], segment_bounds: np.ndarray
) -> dict[str, Any]:
    if segment_bounds.shape != (len(segment_names), 2):
        raise ValueError("Segment bounds do not match segment names.")
    per_segment: dict[str, Any] = {}
    for segment_name, bounds in zip(segment_names, segment_bounds):
        lower_hz, upper_hz = (float(bounds[0]), float(bounds[1]))
        width_hz = upper_hz - lower_hz
        if width_hz <= 0.0:
            raise ValueError(f"Invalid frequency bounds for {segment_name}.")
        per_segment[segment_name] = {
            "lower_hz": lower_hz,
            "upper_hz": upper_hz,
            "center_hz": 0.5 * (lower_hz + upper_hz),
            "width_hz": width_hz,
        }
    return {
        "schema_version": ENGINEERING_SCHEMA_VERSION,
        "name": "segment_local_polynomial_frequency",
        "source_feature": "f",
        "stage": "after_training_fitted_base_transform",
        "definition": "local_t=(f-center_hz)/width_hz; local_t2=local_t^2; local_t3=local_t^3",
        "feature_names": list(LOCAL_FREQUENCY_FEATURES),
        "per_segment": per_segment,
    }


def apply_local_frequency_features(
    X: np.ndarray,
    feature_names: list[str],
    physical_frequency_hz: np.ndarray,
    segment_index: np.ndarray,
    segment_names: list[str],
    spec: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Append persisted local t, t^2, and t^3 without fitting on evaluation data."""
    if spec.get("schema_version") != ENGINEERING_SCHEMA_VERSION:
        raise ValueError("Unsupported local-frequency feature specification.")
    if list(spec.get("feature_names", [])) != list(LOCAL_FREQUENCY_FEATURES):
        raise ValueError("Local-frequency feature names do not match the supported schema.")
    if X.shape[0] != physical_frequency_hz.size or X.shape[0] != segment_index.size:
        raise ValueError("Frequency-feature row arrays are not aligned.")
    if any(name in feature_names for name in LOCAL_FREQUENCY_FEATURES):
        raise ValueError("Local-frequency features already exist in the base feature matrix.")

    local_t = np.empty(X.shape[0], dtype=float)
    stored_segments = spec.get("per_segment", {})
    if set(stored_segments) != set(segment_names):
        raise ValueError("Stored frequency-feature segments do not match the dataset.")
    for segment_id, segment_name in enumerate(segment_names, start=1):
        segment_mask = segment_index == segment_id
        segment_spec = stored_segments[segment_name]
        center_hz = float(segment_spec["center_hz"])
        width_hz = float(segment_spec["width_hz"])
        local_t[segment_mask] = (
            physical_frequency_hz[segment_mask] - center_hz
        ) / width_hz

    engineered = np.column_stack((local_t, local_t**2, local_t**3))
    if not np.all(np.isfinite(engineered)):
        raise ValueError("Local-frequency engineering produced non-finite values.")
    return np.column_stack((X, engineered)), [*feature_names, *LOCAL_FREQUENCY_FEATURES]

