import copy
import json
import unittest
from pathlib import Path

from validation.score_spatial_covariance import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    expected_cell_ids,
    load_preregistration,
    preregistration_digest,
    score_receipt,
    seed_schedule_digest,
    validate_preregistration,
    sha256_json,
    SchemaError,
)


VALIDATION = Path(__file__).parents[1]
PREREGISTRATION = VALIDATION / "spatial_covariance_preregistration.json"
PR61_FIXTURE = VALIDATION / "fixtures" / "spatial_covariance_validation" / "pr61_bookkeeping_receipt.json"


class SpatialCovariancePreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preregistration = load_preregistration(PREREGISTRATION)

    def test_v2_full_matrix_and_protocol_are_explicit(self):
        self.assertEqual(self.preregistration["schema_version"], 2)
        ids = expected_cell_ids(self.preregistration)
        self.assertEqual(len(ids), 89100)
        self.assertIn("hw_7x14|stride_4|ks_frozen|masked|disjoint_after_depth_4|four_blocks_partial_final|evd|near_tie|spatial_correlation_stress", ids)
        self.assertEqual(self.preregistration["generator"]["binary"]["input_schema"], "dolphinrust.spatial-covariance.attempt/2")
        self.assertTrue(self.preregistration["generator"]["binary"]["one_input_one_output"])

    def test_generator_is_dimensioned_and_mirrors_ministack_planner(self):
        generator = self.preregistration["generator"]
        raw = generator["raw_proper_complex"]
        self.assertEqual(raw["covariance_shape"], "N_by_N_per_topology")
        self.assertIn("C_ab", raw["covariance_formula"])
        self.assertIn("lower_hermitian_cholesky", raw["sampler"])
        self.assertEqual(raw["hermitian_rule"], "C_ba=conjugate(C_ab)")
        self.assertEqual(raw["spatial_correlation"]["distance_scale_pixels"], 1.5)
        self.assertEqual(generator["source_centered_empirical"]["mean"], "zero; no sample-mean subtraction")
        planner = generator["acquisition"]["planner"]
        for topology in generator["acquisition"]["topologies"].values():
            starts = list(range(0, topology["acquisition_count"], planner["ministack_size"]))
            expected = [
                {"block_id": block, "num_compressed": min(block, topology["max_num_compressed"]), "real_start": start, "num_real": min(planner["ministack_size"], topology["acquisition_count"] - start)}
                for block, start in enumerate(starts)
            ]
            self.assertEqual(topology["expected_blocks"], expected)
            self.assertEqual(len(topology["date_axis"]), topology["acquisition_count"])

    def test_every_window_stride_has_realized_coordinates_and_exact_overlap_contract(self):
        coordinates = self.preregistration["generator"]["coordinates"]
        # The labels, rather than a single fallback delta, identify every production support.
        expected = {f"{item['id']}|{stride['id']}" for item in self.preregistration["dimensions"]["half_window"] for stride in self.preregistration["dimensions"]["stride"]}
        self.assertEqual(set(coordinates["window_stride"]), expected)
        self.assertEqual(coordinates["overlap_counts"], {"shared_75": 3, "shared_50": 2, "shared_25": 1})
        self.assertEqual(set(coordinates["overlap_fixture"]["reference_units_by_geometry"]), {item["id"] for item in self.preregistration["dimensions"]["pair_geometry"]})
        for spec in coordinates["window_stride"].values():
            self.assertEqual(spec["support_shape"], [2 * spec["half_window"][0] + 1, 2 * spec["half_window"][1] + 1])
            self.assertEqual(set(spec["reference_delta_by_pair_geometry"]), set(item["id"] for item in self.preregistration["dimensions"]["pair_geometry"]))

    def test_production_support_and_status_rules_are_fail_closed(self):
        generator = self.preregistration["generator"]
        self.assertTrue(generator["neighbor_generation"]["full_half_window"])
        self.assertEqual(generator["neighbor_generation"]["offset_order"], "neighbor_grid_row_major_from_clamped_start")
        self.assertEqual(generator["neighbor_generation"]["glrt"]["alpha"], 0.001)
        self.assertEqual(generator["neighbor_generation"]["ks"]["alpha"], 0.001)
        self.assertEqual(generator["supported"]["not_evaluable_if"], ["tied_eigenvalue"])
        self.assertIn("missing_attempt_record", generator["supported"]["receipt_failure_if"])
        changed = copy.deepcopy(self.preregistration)
        changed["generator"]["supported"]["not_evaluable_if"].append("missing_attempt_record")
        with self.assertRaises(SchemaError):
            validate_preregistration(changed)

    def test_cell_overlap_and_sign_drift_is_rejected(self):
        receipt = self._receipt_shell()
        cell_id = expected_cell_ids(self.preregistration)[0]
        labels = cell_id.split("|")
        cell = dict(zip(("half_window", "stride", "support", "position", "pair_geometry", "block_topology", "estimator", "eigen_stress", "source_process"), labels))
        cell.update({"cell_id": cell_id, "status": NOT_EVALUABLE, "not_evaluable_reason": "test-only", "attempted_seeds": 5000, "emitted_seeds": 0, "top_up_seeds": 0, "realized_overlap_percent": 25, "signed_influence_sign": "negative", "attempts": []})
        receipt["cells"] = [cell]
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("realized overlap/sign" in error for error in report["errors"]))

    def test_frozen_threshold_and_generator_changes_are_rejected(self):
        changed = copy.deepcopy(self.preregistration)
        changed["thresholds"]["coverage_absolute_error_max"] = 0.03
        with self.assertRaises(SchemaError):
            validate_preregistration(changed)
        mutations = (
            ("binary", "input_schema", "drift/2"),
            ("acquisition", "cadence_days", 24),
            ("raw_proper_complex", "noise_scale", 2.0),
            ("source_centered_empirical", "shrinkage_alpha", 0.1),
            ("effective_looks", "distance_scale_pixels", 2.0),
            ("estimators", "emi", {"beta": 0.1}),
            ("neighbor_generation", "fixed_support_reuse", False),
            ("coordinates", "rounding", "float"),
            ("truth", "z_95", 1.96),
            ("supported", "supported_if", ["finite_raw_source"]),
            ("identity", "protocol_version", 3),
        )
        for section, field, value in mutations:
            candidate = copy.deepcopy(self.preregistration)
            candidate["generator"][section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(SchemaError):
                validate_preregistration(candidate)

    def _receipt_shell(self):
        generator = self.preregistration["generator"]
        return {
            "schema": "dolphinrust.spatial_covariance.receipt",
            "schema_version": 2,
            "preregistration_sha256": preregistration_digest(self.preregistration),
            "seed_schedule_sha256": seed_schedule_digest(self.preregistration),
            "protocol": {key: generator["binary"][key] for key in ("input_schema", "output_schema", "one_input_one_output")},
            "binary": {"release_invocation": generator["binary"]["release_invocation"], "release_only": True},
            "hashes": {field: "0" * 64 for field in ("code_sha256", "fixture_sha256", "operator_sha256", "variance_sha256", "resource_sha256", "generator_protocol_sha256", "config_sha256", "source_model_sha256", "result_sha256", "binary_sha256")},
            "cells": [],
            "resources": [],
        }

    def test_pr61_bookkeeping_receipt_cannot_satisfy_v2_schema(self):
        with PR61_FIXTURE.open(encoding="utf-8") as handle:
            receipt = json.load(handle)
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("version 2" in error for error in report["errors"]))
        self.assertTrue(any("aggregate-only" in error for error in report["errors"]))

    def test_missing_cells_and_aggregate_receipt_are_rejected(self):
        report = score_receipt(self.preregistration, self._receipt_shell())
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("missing 89100" in error for error in report["errors"]))
        self.assertTrue(any("resource receipts" in error for error in report["errors"]))

    def test_receipt_identity_hash_drift_is_rejected(self):
        receipt = self._receipt_shell()
        receipt["hashes"]["generator_protocol_sha256"] = sha256_json({"protocol": "drift"})
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("generator protocol hash" in error for error in report["errors"]))

    def test_incomplete_per_attempt_evidence_is_rejected(self):
        receipt = self._receipt_shell()
        cell_id = expected_cell_ids(self.preregistration)[0]
        receipt["cells"] = [{"cell_id": cell_id, "attempts": []}]
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("per-attempt" in error for error in report["errors"]))

    def test_top_up_is_rejected_even_for_a_known_cell(self):
        receipt = self._receipt_shell()
        cell_id = expected_cell_ids(self.preregistration)[0]
        labels = cell_id.split("|")
        cell = dict(zip(("half_window", "stride", "support", "position", "pair_geometry", "block_topology", "estimator", "eigen_stress", "source_process"), labels))
        cell.update({"cell_id": cell_id, "status": NOT_EVALUABLE, "not_evaluable_reason": "test-only", "attempted_seeds": 5000, "emitted_seeds": 0, "top_up_seeds": 1, "attempts": []})
        receipt["cells"] = [cell]
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("zero top-up" in error for error in report["errors"]))

    def test_status_vocabulary_remains_distinct(self):
        self.assertEqual({PASS, FAIL, NOT_EVALUABLE}, {"pass", "fail", "not_evaluable"})


if __name__ == "__main__":
    unittest.main()
