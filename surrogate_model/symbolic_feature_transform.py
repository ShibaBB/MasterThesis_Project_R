"""Training-fitted feature transforms shared by symbolic-regression tools."""

from __future__ import annotations

from typing import Any

import numpy as np


TRANSFORM_SCHEMA_VERSION = 1


def fit_feature_transform(
    X_train: np.ndarray,
    feature_names: list[str],
    recommended_log10_indices_1based: list[int],
) -> dict[str, Any]:
    """Fit a deterministic log10/drop-constant transform on training rows only."""
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2 or X_train.shape[1] != len(feature_names):
        raise ValueError("Training feature matrix does not match feature_names.")
    if not np.all(np.isfinite(X_train)):
        raise ValueError("Training features contain non-finite values.")

    log_indices = sorted({int(index) - 1 for index in recommended_log10_indices_1based})
    if any(index < 0 or index >= X_train.shape[1] for index in log_indices):
        raise ValueError("A recommended log10 feature index is outside the feature matrix.")

    constant_indices = [
        index
        for index in range(X_train.shape[1])
        if np.all(X_train[:, index] == X_train[0, index])
    ]
    retained_indices = [
        index for index in range(X_train.shape[1]) if index not in constant_indices
    ]
    if not retained_indices:
        raise ValueError("All symbolic-regression input features are constant.")

    retained_log_indices = [index for index in log_indices if index in retained_indices]
    for index in retained_log_indices:
        if np.any(X_train[:, index] <= 0.0):
            raise ValueError(
                f"Feature {feature_names[index]!r} contains non-positive training values "
                "and cannot be transformed with log10."
            )

    transformed_names = [
        f"log10_{feature_names[index]}" if index in retained_log_indices else feature_names[index]
        for index in retained_indices
    ]
    return {
        "schema_version": TRANSFORM_SCHEMA_VERSION,
        "method": "recommended_log10_then_drop_training_constants",
        "original_feature_names": list(feature_names),
        "recommended_log10_indices_1based": [index + 1 for index in log_indices],
        "log10_feature_indices_0based": retained_log_indices,
        "log10_feature_names": [feature_names[index] for index in retained_log_indices],
        "constant_feature_indices_0based": constant_indices,
        "constant_feature_names": [feature_names[index] for index in constant_indices],
        "retained_feature_indices_0based": retained_indices,
        "retained_feature_names": [feature_names[index] for index in retained_indices],
        "transformed_feature_names": transformed_names,
    }


def identity_feature_transform(feature_names: list[str]) -> dict[str, Any]:
    indices = list(range(len(feature_names)))
    return {
        "schema_version": TRANSFORM_SCHEMA_VERSION,
        "method": "identity",
        "original_feature_names": list(feature_names),
        "recommended_log10_indices_1based": [],
        "log10_feature_indices_0based": [],
        "log10_feature_names": [],
        "constant_feature_indices_0based": [],
        "constant_feature_names": [],
        "retained_feature_indices_0based": indices,
        "retained_feature_names": list(feature_names),
        "transformed_feature_names": list(feature_names),
    }


def apply_feature_transform(
    X: np.ndarray,
    feature_names: list[str],
    transform: dict[str, Any],
) -> np.ndarray:
    """Apply a previously fitted transform without re-estimating any decisions."""
    X = np.asarray(X, dtype=float)
    expected_names = list(transform["original_feature_names"])
    if list(feature_names) != expected_names:
        raise ValueError(
            "Dataset feature names do not match the feature transform fitted during training."
        )
    if X.ndim != 2 or X.shape[1] != len(expected_names):
        raise ValueError("Feature matrix does not match the fitted feature transform.")

    transformed = X.copy()
    for index in transform["log10_feature_indices_0based"]:
        index = int(index)
        if np.any(transformed[:, index] <= 0.0):
            raise ValueError(
                f"Feature {expected_names[index]!r} contains non-positive values and "
                "cannot be transformed with log10."
            )
        transformed[:, index] = np.log10(transformed[:, index])

    retained_indices = [int(index) for index in transform["retained_feature_indices_0based"]]
    transformed = transformed[:, retained_indices]
    if not np.all(np.isfinite(transformed)):
        raise ValueError("Transformed features contain non-finite values.")
    return transformed
