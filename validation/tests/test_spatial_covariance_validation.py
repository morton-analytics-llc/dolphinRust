import copy
import json
import unittest
from pathlib import Path

from validation.score_spatial_covariance import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    SchemaError,
    expected_cell_ids,
    load_preregistration,
    preregistration_digest,
    score_receipt,
    seed_schedule_digest,
    validate_preregistration,
)


VALIDATION = Path(__file__).parents[1]
PREREGISTRATION = VALIDATION / "spatial_covariance_preregistration.json"
PR61_FIXTURE = VALIDATION / "fixtures" / "spatial_covariance_validation" / "pr61_bookkeeping_receipt.json"


class SpatialCovariancePreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preregistration = load_preregistration(PREREGISTRATION)

    def test_full_matrix_is_explicit_and_not_the_pr61_36_label_loop(self):
        ids = expected_cell_ids(self.preregistration)
        self.assertEqual(len(ids), 89100)
        self.assertIn(
            "hw_7x14|stride_4|ks_frozen|masked|disjoint_after_depth_4|four_blocks_partial_final|evd|near_tie|spatial_correlation_stress",
            ids,
        )
        self.assertEqual(len(set(ids)), len(ids))

    def test_frozen_threshold_change_is_rejected(self):
        changed = copy.deepcopy(self.preregistration)
        changed["thresholds"]["coverage_absolute_error_max"] = 0.03
        with self.assertRaises(SchemaError):
            validate_preregistration(changed)

    def test_pr61_bookkeeping_receipt_cannot_satisfy_schema(self):
        with PR61_FIXTURE.open(encoding="utf-8") as handle:
            receipt = json.load(handle)
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("receipt schema" in error for error in report["errors"]))
        self.assertTrue(any("aggregate-only" in error for error in report["errors"]))

    def test_missing_cells_and_aggregate_receipt_are_rejected(self):
        receipt = {
            "schema": "dolphinrust.spatial_covariance.receipt",
            "schema_version": 1,
            "preregistration_sha256": preregistration_digest(self.preregistration),
            "seed_schedule_sha256": seed_schedule_digest(self.preregistration),
            "hashes": {field: "0" * 64 for field in ("code_sha256", "fixture_sha256", "operator_sha256", "variance_sha256", "resource_sha256")},
            "cells": [],
            "resources": [],
        }
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("missing 89100" in error for error in report["errors"]))
        self.assertTrue(any("resource receipts" in error for error in report["errors"]))

    def test_preregistration_hash_mismatch_is_rejected(self):
        receipt = {
            "schema": "dolphinrust.spatial_covariance.receipt",
            "schema_version": 1,
            "preregistration_sha256": "f" * 64,
            "seed_schedule_sha256": seed_schedule_digest(self.preregistration),
            "hashes": {field: "0" * 64 for field in ("code_sha256", "fixture_sha256", "operator_sha256", "variance_sha256", "resource_sha256")},
            "cells": [],
            "resources": [],
        }
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("preregistration_sha256" in error for error in report["errors"]))

    def test_top_up_is_rejected_even_for_a_known_cell(self):
        cell_id = expected_cell_ids(self.preregistration)[0]
        labels = cell_id.split("|")
        cell = dict(zip(("half_window", "stride", "support", "position", "pair_geometry", "block_topology", "estimator", "eigen_stress", "source_process"), labels))
        cell.update(
            {
                "cell_id": cell_id,
                "status": NOT_EVALUABLE,
                "not_evaluable_reason": "test-only",
                "attempted_seeds": 5000,
                "emitted_seeds": 5000,
                "top_up_seeds": 1,
                "operator_hash": "0" * 64,
                "variance_hash": "0" * 64,
                "psd_hash": "0" * 64,
                "coverage_hash": "0" * 64,
                "emission_hash": "0" * 64,
            }
        )
        receipt = {
            "schema": "dolphinrust.spatial_covariance.receipt",
            "schema_version": 1,
            "preregistration_sha256": preregistration_digest(self.preregistration),
            "seed_schedule_sha256": seed_schedule_digest(self.preregistration),
            "hashes": {field: "0" * 64 for field in ("code_sha256", "fixture_sha256", "operator_sha256", "variance_sha256", "resource_sha256")},
            "cells": [cell],
            "resources": [],
        }
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertTrue(any("top-up seeds" in error for error in report["errors"]))

    def test_status_vocabulary_remains_distinct(self):
        self.assertEqual({PASS, FAIL, NOT_EVALUABLE}, {"pass", "fail", "not_evaluable"})


if __name__ == "__main__":
    unittest.main()
