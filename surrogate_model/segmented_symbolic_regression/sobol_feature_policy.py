"""Sobol-guided feature policies for phase-one segmented SR experiments."""

from __future__ import annotations

from typing import Any


POLICY_SCHEMA_VERSION = 1
FREQUENCY_ENHANCED_BRANCH = "frequency_enhanced_f1"
SUPPORTED_BRANCHES = (
    "all",
    "sobol_subset",
    "sobol_plus_high_lambda_prime",
    FREQUENCY_ENHANCED_BRANCH,
)

EXPECTED_SEGMENTS = (
    "low_100_700",
    "midlow_700_1000",
    "midhigh_1000_1300",
    "highlow_1300_1650",
    "high_1650_2000",
    "upperlow_2000_3000",
    "uppermid_3000_4000",
    "upperhigh_4000_4950",
)

FREQUENCY_FEATURE = "log10_f"
LOCAL_FREQUENCY_FEATURES = ("local_t", "local_t2", "local_t3")

PHASE1_VALIDATION_WINNERS = {
    "re": {
        "low_100_700": "sobol_subset",
        **{segment: "all" for segment in EXPECTED_SEGMENTS[1:]},
    },
    "im": {
        **{segment: "all" for segment in EXPECTED_SEGMENTS},
        "highlow_1300_1650": "sobol_subset",
        "upperhigh_4000_4950": "sobol_subset",
    },
}

# Feature names refer to the persisted, training-fitted base transform. The
# order in each set expresses Sobol priority; the final matrix order follows
# the base transform so matched branches cannot differ merely by column order.
SOBOL_FEATURES = {
    "re": {
        "low_100_700": ("log10_sigma", "log10_k0_prime", "alpha_infinity"),
        "midlow_700_1000": (
            "log10_sigma", "log10_k0_prime", "alpha_infinity", "log10_lambda"
        ),
        "midhigh_1000_1300": (
            "log10_sigma", "log10_k0_prime", "log10_lambda", "alpha_infinity"
        ),
        "highlow_1300_1650": ("log10_sigma", "log10_lambda", "alpha_infinity"),
        "high_1650_2000": ("log10_sigma", "log10_lambda", "alpha_infinity"),
        "upperlow_2000_3000": ("log10_sigma", "alpha_infinity", "log10_lambda"),
        "uppermid_3000_4000": (
            "alpha_infinity", "log10_sigma", "log10_lambda", "log10_k0_prime"
        ),
        "upperhigh_4000_4950": (
            "log10_sigma", "alpha_infinity", "log10_k0_prime", "log10_lambda"
        ),
    },
    "im": {
        "low_100_700": ("log10_sigma", "log10_k0_prime"),
        "midlow_700_1000": ("log10_sigma", "alpha_infinity", "log10_lambda"),
        "midhigh_1000_1300": ("log10_sigma", "alpha_infinity", "log10_lambda"),
        "highlow_1300_1650": ("alpha_infinity", "log10_sigma", "log10_lambda"),
        "high_1650_2000": ("log10_sigma", "alpha_infinity", "log10_lambda"),
        "upperlow_2000_3000": ("log10_sigma", "alpha_infinity", "log10_lambda"),
        "uppermid_3000_4000": ("log10_sigma", "alpha_infinity", "log10_lambda"),
        "upperhigh_4000_4950": (
            "log10_sigma", "alpha_infinity", "log10_lambda", "log10_k0_prime"
        ),
    },
}


def build_feature_policy(
    target: str,
    branch: str,
    transformed_feature_names: list[str],
    segment_names: list[str],
) -> dict[str, Any]:
    """Build and validate a per-segment feature view for one training run."""
    if target not in SOBOL_FEATURES:
        raise ValueError(f"Unsupported target for Sobol feature policy: {target!r}")
    if branch not in SUPPORTED_BRANCHES:
        raise ValueError(f"Unsupported feature branch: {branch!r}")
    if tuple(segment_names) != EXPECTED_SEGMENTS:
        raise ValueError(
            "Sobol feature policy requires the configured run1 eight-segment layout; "
            f"received {segment_names!r}."
        )
    if FREQUENCY_FEATURE not in transformed_feature_names:
        raise ValueError(
            f"Required frequency feature {FREQUENCY_FEATURE!r} is missing after transformation."
        )

    per_segment: dict[str, Any] = {}
    for segment_name in segment_names:
        phase1_material_branch = branch
        if branch == "all":
            requested = list(transformed_feature_names)
        elif branch == FREQUENCY_ENHANCED_BRANCH:
            phase1_material_branch = PHASE1_VALIDATION_WINNERS[target][segment_name]
            if phase1_material_branch == "all":
                requested = [
                    name
                    for name in transformed_feature_names
                    if name not in LOCAL_FREQUENCY_FEATURES
                ]
            else:
                requested = [*SOBOL_FEATURES[target][segment_name], FREQUENCY_FEATURE]
            requested.extend(LOCAL_FREQUENCY_FEATURES)
        else:
            requested = [*SOBOL_FEATURES[target][segment_name], FREQUENCY_FEATURE]
            if (
                branch == "sobol_plus_high_lambda_prime"
                and segment_name == "upperhigh_4000_4950"
            ):
                requested.append("log10_lambda_prime")

        missing = sorted(set(requested) - set(transformed_feature_names))
        if missing:
            raise ValueError(
                f"Segment {segment_name} requests unavailable transformed features: {missing}"
            )
        selected_names = [name for name in transformed_feature_names if name in requested]
        selected_indices = [transformed_feature_names.index(name) for name in selected_names]
        per_segment[segment_name] = {
            "feature_names": selected_names,
            "feature_indices_0based": selected_indices,
            "phase1_material_branch": phase1_material_branch,
        }

    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "name": (
            "phase1_material_policy_plus_local_frequency_f1"
            if branch == FREQUENCY_ENHANCED_BRANCH
            else "teacher_sobol_phase1_segment_features"
        ),
        "source": (
            "phase1 validation winners plus local frequency basis"
            if branch == FREQUENCY_ENHANCED_BRANCH
            else "teacher Sobol run1 N=1024"
        ),
        "branch": branch,
        "target": target,
        "frequency_always_retained": True,
        "selection_stage": "after_training_fitted_base_transform",
        "per_segment": per_segment,
    }


def validate_stored_feature_policy(
    policy: dict[str, Any],
    target: str,
    transformed_feature_names: list[str],
    segment_names: list[str],
) -> dict[str, Any]:
    """Fail loudly if evaluation cannot reproduce a stored feature policy."""
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing segmented feature-policy schema version.")
    if policy.get("target") != target:
        raise ValueError("Training feature policy target does not match evaluation target.")
    stored_segments = policy.get("per_segment", {})
    if set(stored_segments) != set(segment_names):
        raise ValueError("Training feature policy segments do not match the evaluation dataset.")

    for segment_name in segment_names:
        spec = stored_segments[segment_name]
        names = list(spec.get("feature_names", []))
        indices = [int(value) for value in spec.get("feature_indices_0based", [])]
        expected = [transformed_feature_names[index] for index in indices]
        if not names or names != expected:
            raise ValueError(
                f"Stored feature names/indices are inconsistent for segment {segment_name}."
            )
        if FREQUENCY_FEATURE not in names:
            raise ValueError(f"Segment {segment_name} omits required frequency feature.")
    return policy
