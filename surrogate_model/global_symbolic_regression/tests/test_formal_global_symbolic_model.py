"""System and regression tests for the accepted Re F3 / Im F2 model pair."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = TEST_DIR.parent
SURROGATE_ROOT = SCRIPT_DIR.parent
for import_path in (SCRIPT_DIR, SURROGATE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from evaluate_formal_global_symbolic_model import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_RE_REFERENCE,
    DEFAULT_SPLIT,
    evaluate_partition,
    load_paired_symbolic_dataset,
)
from formal_global_symbolic_model import FormalGlobalSymbolicModel  # noqa: E402
from shared_split_utils import load_shared_split, source_curve_mask  # noqa: E402


class FormalGlobalSymbolicModelSystemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = FormalGlobalSymbolicModel.load()
        cls.paired = load_paired_symbolic_dataset(DEFAULT_DATASET)
        cls.split = load_shared_split(
            DEFAULT_SPLIT, expected_dataset_run="run1", expected_curve_count=1000
        )
        cls.test_mask = source_curve_mask(
            cls.paired["source_curve_index"], cls.split, "test"
        )
        cls.test_X = cls.paired["X"][cls.test_mask]
        cls.test_prediction = cls.model.predict_with_diagnostics(
            cls.test_X, cls.paired["feature_names"]
        )

    def test_active_pointer_and_frozen_contract(self) -> None:
        self.assertEqual(self.model.manifest["targets"]["re"]["active_model"], "Re_F3")
        self.assertEqual(self.model.manifest["targets"]["im"]["active_model"], "Im_F2")
        self.assertEqual(self.model.split_hash, self.split["split_hash"])
        self.assertEqual(self.model.manifest["output_postprocessing"], {"method": "none"})
        self.assertFalse(
            self.model.manifest["freeze_contract"]["training_performed_by_export"]
        )
        self.assertFalse(
            self.model.manifest["freeze_contract"]["candidate_selection_performed_by_export"]
        )

    def test_exact_saved_prediction_and_metric_regression(self) -> None:
        summary, _ = evaluate_partition(
            self.model,
            self.paired,
            self.split,
            "test",
            DEFAULT_RE_REFERENCE,
        )
        self.assertTrue(summary["exact_re_prediction_regression"]["passed"])
        self.assertEqual(
            summary["exact_re_prediction_regression"][
                "maximum_absolute_prediction_difference"
            ],
            0.0,
        )
        self.assertTrue(summary["reference_metric_regression"]["re"]["passed"])
        self.assertTrue(summary["reference_metric_regression"]["im"]["passed"])

    def test_combined_output_and_frequency_boundaries(self) -> None:
        reflect = self.test_prediction["Reflect"]
        self.assertTrue(np.array_equal(reflect.real, self.test_prediction["R_real"]))
        self.assertTrue(np.array_equal(reflect.imag, self.test_prediction["R_imag"]))

        sample = np.repeat(self.test_X[:1], 4, axis=0)
        sample[:, self.model.input_feature_names.index("f")] = [
            100.0,
            1250.0,
            2100.0,
            4950.0,
        ]
        boundary_prediction = self.model.predict_with_diagnostics(sample)
        self.assertTrue(np.all(np.isfinite(boundary_prediction["Reflect"])))

        below = sample[:1].copy()
        below[0, self.model.input_feature_names.index("f")] = 99.0
        with self.assertRaisesRegex(ValueError, "outside the validated"):
            self.model.predict_components(below)
        above = sample[:1].copy()
        above[0, self.model.input_feature_names.index("f")] = 4951.0
        with self.assertRaisesRegex(ValueError, "outside the validated"):
            self.model.predict_components(above)

    def test_fixed_inputs_and_serialization_replay(self) -> None:
        changed_phi = self.test_X[:1].copy()
        changed_phi[0, self.model.input_feature_names.index("phi")] = 0.91
        with self.assertRaisesRegex(ValueError, "constant"):
            self.model.predict_components(changed_phi)

        sample = self.test_X[::997]
        expected = self.model.predict_with_diagnostics(sample)
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "model_manifest.json"
            self.model.serialize(manifest)
            reloaded = FormalGlobalSymbolicModel.load(manifest)
            actual = reloaded.predict_with_diagnostics(sample)
        for key in (
            "R_real",
            "R_imag",
            "Reflect",
            "Re_F1",
            "Re_F3_residual",
            "z_sigma__f3_mid",
        ):
            self.assertTrue(np.array_equal(expected[key], actual[key]), key)


if __name__ == "__main__":
    unittest.main()
