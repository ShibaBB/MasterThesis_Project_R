"""Sobol-guided feature policies for full-range Global SR experiments."""

from __future__ import annotations

from typing import Any


POLICY_SCHEMA_VERSION = 1
SUPPORTED_BRANCHES = ("all", "sobol_subset", "sobol_modulated")
FREQUENCY_FEATURE = "log10_f"

# Teacher Sobol run1 N=1024 ranks lambda_prime last over the complete
# 100--4950 Hz response for both targets.  The other four varying material
# inputs are retained because they have meaningful full-range or local effects.
SOBOL_FEATURES = {
    "re": (
        "log10_sigma",
        "alpha_infinity",
        "log10_lambda",
        "log10_k0_prime",
    ),
    "im": (
        "log10_sigma",
        "alpha_infinity",
        "log10_lambda",
        "log10_k0_prime",
    ),
}


def build_feature_policy(
    target: str,
    branch: str,
    transformed_feature_names: list[str],
) -> dict[str, Any]:
    """Build a single full-range feature view after the fitted base transform."""
    if target not in SOBOL_FEATURES:
        raise ValueError(f"Unsupported target for Global Sobol policy: {target!r}")
    if branch not in SUPPORTED_BRANCHES:
        raise ValueError(f"Unsupported Global feature branch: {branch!r}")
    if FREQUENCY_FEATURE not in transformed_feature_names:
        raise ValueError(
            f"Required frequency feature {FREQUENCY_FEATURE!r} is missing after transformation."
        )

    requested = (
        list(transformed_feature_names)
        if branch in {"all", "sobol_modulated"}
        else [*SOBOL_FEATURES[target], FREQUENCY_FEATURE]
    )
    missing = sorted(set(requested) - set(transformed_feature_names))
    if missing:
        raise ValueError(f"Global feature policy requests unavailable features: {missing}")

    selected_names = [name for name in transformed_feature_names if name in requested]
    selected_indices = [transformed_feature_names.index(name) for name in selected_names]
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "name": (
            "teacher_sobol_smooth_modulation_base_features"
            if branch == "sobol_modulated"
            else "teacher_sobol_full_range_global_features"
        ),
        "source": "teacher Sobol run1 N=1024, variance-weighted 100-4950 Hz ST",
        "branch": branch,
        "target": target,
        "frequency_range_hz": [100.0, 4950.0],
        "frequency_always_retained": True,
        "selection_stage": "after_training_fitted_base_transform",
        "feature_names": selected_names,
        "feature_indices_0based": selected_indices,
        "excluded_transformed_features": [
            name for name in transformed_feature_names if name not in selected_names
        ],
    }


def validate_stored_feature_policy(
    policy: dict[str, Any],
    target: str,
    transformed_feature_names: list[str],
) -> dict[str, Any]:
    """Fail if evaluation cannot reproduce the training feature view exactly."""
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing Global feature-policy schema version.")
    if policy.get("target") != target:
        raise ValueError("Training feature policy target does not match evaluation target.")
    if policy.get("branch") not in SUPPORTED_BRANCHES:
        raise ValueError("Training metadata contains an unsupported Global feature branch.")

    names = list(policy.get("feature_names", []))
    indices = [int(value) for value in policy.get("feature_indices_0based", [])]
    try:
        expected = [transformed_feature_names[index] for index in indices]
    except IndexError as error:
        raise ValueError("Stored Global feature indices exceed transformed input width.") from error
    if not names or names != expected:
        raise ValueError("Stored Global feature names and indices are inconsistent.")
    if FREQUENCY_FEATURE not in names:
        raise ValueError("Stored Global feature policy omits the required frequency feature.")
    return policy
