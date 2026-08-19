"""Export the accepted run1 Global SR Re F3 / Im F2 pair as one model bundle.

This command performs no fitting and no model selection.  It verifies the
persisted acceptance decisions, copies the frozen transforms and equations into
a self-contained JSON manifest, and refuses to overwrite by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SURROGATE_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from formal_global_symbolic_model import FormalGlobalSymbolicModel  # noqa: E402


ARTIFACT_ROOT = SCRIPT_DIR / "artifacts"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "formal_models" / "run1_f3_f2"
RE_F1_METADATA = ARTIFACT_ROOT / "re" / "train" / "20260816_run1_sobol_modulated" / "training_metadata.json"
RE_F1_SELECTION = ARTIFACT_ROOT / "re" / "eval" / "20260816_run1_sobol_modulated_best_loss" / "selected_candidate.csv"
RE_F3_FREEZE_DIR = ARTIFACT_ROOT / "F3" / "F3-F1S" / "20260818T211720"
RE_F3_TEST_DIR = ARTIFACT_ROOT / "F3" / "F3-F2" / "20260818T213528"
IM_F2_METADATA = ARTIFACT_ROOT / "im" / "train" / "20260816_f2_im_f2_formal" / "training_metadata.json"
IM_F2_EVAL_DIR = ARTIFACT_ROOT / "f2_evaluation_20260816" / "eval" / "im_formal" / "20260816_f2_im_f2_formal_v"
IM_F2_ACCEPTANCE = ARTIFACT_ROOT / "f2_manifests" / "20260816_f2_im" / "comparisons" / "im" / "formal_acceptance_decision.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_single_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one frozen equation row: {path}")
    return rows[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def source_record(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen source artifact is missing: {path}")
    return {"path": repo_relative(path), "sha256": sha256_file(path)}


def build_manifest() -> dict[str, Any]:
    re_metadata = load_json(RE_F1_METADATA)
    re_selection = load_single_csv_row(RE_F1_SELECTION)
    re_frozen = load_json(RE_F3_FREEZE_DIR / "frozen_candidate_spec.json")
    re_acceptance = load_json(RE_F3_TEST_DIR / "comparison_decision.json")
    im_metadata = load_json(IM_F2_METADATA)
    im_selection = load_single_csv_row(IM_F2_EVAL_DIR / "selected_candidate.csv")
    im_summary = load_json(IM_F2_EVAL_DIR / "selected_global_summary.json")
    im_acceptance = load_json(IM_F2_ACCEPTANCE)

    split_hash = str(re_metadata["shared_split_hash"])
    observed_hashes = {
        split_hash,
        str(re_frozen["shared_split_hash"]),
        str(im_metadata["shared_split_hash"]),
        str(im_summary["shared_split_hash"]),
    }
    if observed_hashes != {split_hash}:
        raise ValueError(f"Frozen Global SR sources use different split hashes: {observed_hashes}")
    if re_metadata["feature_transform"] != im_metadata["feature_transform"]:
        raise ValueError("Accepted Re and Im models do not share one raw feature transform.")
    if re_acceptance.get("decision") != "accept_frozen_Re_F3_as_formal_Re_model":
        raise ValueError("Persisted Re F3 formal acceptance decision is missing.")
    if re_acceptance.get("formal_model_replacement_decision") is not True:
        raise ValueError("Persisted Re F3 replacement flag is not true.")
    frozen_spec_hash = sha256_file(RE_F3_FREEZE_DIR / "frozen_candidate_spec.json")
    if re_acceptance.get("frozen_candidate_spec_sha256") != frozen_spec_hash:
        raise ValueError("Persisted Re F3 candidate changed after formal acceptance.")
    if im_acceptance.get("selected_checkpoint") != "F2":
        raise ValueError("Persisted Im formal checkpoint is not F2.")
    if im_acceptance.get("test_metrics_used_for_decision") is not False:
        raise ValueError("Im F2 acceptance did not preserve validation-only selection.")
    if im_acceptance.get("regional_guardrails_passed") is not True:
        raise ValueError("Im F2 acceptance guardrails did not pass.")
    if int(im_selection["candidate_index"]) != 14:
        raise ValueError("Accepted Im F2 equation is no longer candidate 14.")
    if re_frozen.get("formula_domain_safety_passed") is not True:
        raise ValueError("Frozen Re F3 equation is not marked domain-safe.")

    re_f1_names = list(re_metadata["feature_names"])
    im_names = list(im_metadata["feature_names"])
    f3_terminal_spec = re_frozen["f3b_terminal_spec"]
    manifest = {
        "schema_version": 1,
        "model_id": "run1_global_sr_re_f3_im_f2",
        "model_family": "global_symbolic_regression",
        "dataset": {
            "run_id": "run1",
            "complex_source": "Reflect",
            "shared_split_hash": split_hash,
            "frequency_grid_hz": {"min": 100.0, "max": 4950.0, "points": 128},
        },
        "input_contract": {
            "feature_names": list(re_metadata["original_feature_names"]),
            "fixed_features": {"phi": 0.92, "h": 0.03},
            "frequency_feature": "f",
            "frequency_range_hz": [100.0, 4950.0],
            "out_of_range_policy": "fail_unless_allow_extrapolation_is_explicit",
        },
        "feature_transform": re_metadata["feature_transform"],
        "targets": {
            "re": {
                "target_name": "R_real",
                "definition": "real(Reflect)",
                "active_model": "Re_F3",
                "model_definition": "Re_F3 = Re_F1 + delta_Re",
                "feature_policy": re_metadata["feature_policy"],
                "frequency_modulation": re_metadata["frequency_modulation"],
                "f1_equation": {
                    "source_model": "Re_F1",
                    "validation_selected_candidate": int(re_selection["candidate_index"]),
                    "complexity": int(re_selection["complexity"]),
                    "equation": re_selection["equation"],
                    "expression_for_eval": re_selection["expression_for_eval"],
                    "variable_names": re_f1_names,
                },
                "f3_local_terminal_spec": f3_terminal_spec,
                "f3_residual": {
                    "source_stage": "F3-F1S",
                    "validation_selected_candidate": int(re_frozen["source_candidate_index"]),
                    "complexity": int(re_frozen["source_candidate_complexity"]),
                    "equation": re_frozen["canonical_equation"],
                    "expression_for_eval": re_frozen["canonical_expression_for_eval"],
                    "variable_names": list(re_frozen["input_feature_names"]),
                    "formula_safety_passed": True,
                    "f1_fallback_at_terminal_zero": True,
                },
                "formal_acceptance_decision": re_acceptance["decision"],
                "reference_test_metrics": re_acceptance["frozen_f3_test_metrics"],
                "output_postprocessing": {"method": "none"},
            },
            "im": {
                "target_name": "R_imag",
                "definition": "imag(Reflect)",
                "active_model": "Im_F2",
                "feature_policy": im_metadata["feature_policy"],
                "frequency_modulation": im_metadata["frequency_modulation"],
                "equation": {
                    "validation_selected_candidate": int(im_selection["candidate_index"]),
                    "complexity": int(im_selection["complexity"]),
                    "equation": im_selection["equation"],
                    "expression_for_eval": im_selection["expression_for_eval"],
                    "variable_names": im_names,
                },
                "validation_selected_candidate": int(im_selection["candidate_index"]),
                "formal_acceptance_decision": "accept_Im_F2_as_formal_Im_model",
                "reference_test_metrics": im_acceptance["f2_test_metrics_final_reporting"],
                "output_postprocessing": {"method": "none"},
            },
        },
        "combined_output": {
            "name": "Reflect",
            "definition": "R_real + 1j * R_imag",
            "row_order_contract": "Re and Im are predicted from the same raw input rows",
        },
        "output_postprocessing": {"method": "none"},
        "provenance": {
            "re_f1_training_metadata": source_record(RE_F1_METADATA),
            "re_f1_selected_candidate": source_record(RE_F1_SELECTION),
            "re_f3_frozen_candidate": source_record(
                RE_F3_FREEZE_DIR / "frozen_candidate_spec.json"
            ),
            "re_f3_formal_acceptance": source_record(
                RE_F3_TEST_DIR / "comparison_decision.json"
            ),
            "re_f3_reference_predictions": source_record(
                RE_F3_TEST_DIR / "test_predictions.npz"
            ),
            "im_f2_training_metadata": source_record(IM_F2_METADATA),
            "im_f2_selected_candidate": source_record(
                IM_F2_EVAL_DIR / "selected_candidate.csv"
            ),
            "im_f2_evaluation_summary": source_record(
                IM_F2_EVAL_DIR / "selected_global_summary.json"
            ),
            "im_f2_formal_acceptance": source_record(IM_F2_ACCEPTANCE),
        },
        "freeze_contract": {
            "training_performed_by_export": False,
            "candidate_selection_performed_by_export": False,
            "test_data_used_for_export_decision": False,
            "re_formula_and_constants_frozen": True,
            "im_formula_and_constants_frozen": True,
        },
    }
    FormalGlobalSymbolicModel(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_file = output_dir / "model_manifest.json"
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite formal model bundle: {output_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    FormalGlobalSymbolicModel.load(output_file)
    print(f"Exported immutable formal Global SR model: {output_file}")
    print("Active targets: Re_F3 / Im_F2; no training or candidate selection was run.")


if __name__ == "__main__":
    main()
