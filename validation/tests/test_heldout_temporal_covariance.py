import copy
import json
import unittest
from pathlib import Path

from validation.heldout_temporal_covariance.cohort import (
    CohortValidationError,
    build_manifest,
    canonical_digest,
    discover_candidates,
    validate_manifest,
)
from validation.heldout_temporal_covariance.scorer import (
    exact_binomial_noninferiority,
    score_receipt,
    score_slope_difference,
)


VALIDATION = Path(__file__).parents[1]
PREREGISTRATION_PATH = VALIDATION / "temporal_covariance_heldout_preregistration.json"


def candidate(index: int, **overrides):
    value = {
        "candidate_id": "candidate-%03d" % index,
        "source_kind": "catalog_metadata",
        "burst_id": "T999_%06d_IW1" % index,
        "orbit_id": "orbit-%03d" % index,
        "footprint_id": "footprint-%03d" % index,
        "site_id": "site-%03d" % index,
        "frame_id": "frame-%03d" % index,
        "station_ids": ["A%03d" % index, "B%03d" % index],
        "date_start": "2020-01-01",
        "date_end": "2021-01-01",
        "epoch_count": 24,
        "metadata_hashes": {
            "catalog_sha256": "0" * 64,
            "burst_metadata_sha256": "1" * 64,
            "gnss_station_metadata_sha256": "2" * 64,
        },
        "query_digest": "f53-06-metadata-query-v1",
    }
    value.update(overrides)
    return value


class HeldoutCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))

    def test_metadata_discovery_excludes_exposed_and_outcome_records(self):
        records = [
            candidate(0),
            candidate(1, site_id="fresno"),
            candidate(2, station_ids=["A002", "MMX1"]),
            candidate(3, displacement=[1.0]),
        ]
        discovered = discover_candidates(records, self.preregistration)
        self.assertTrue(discovered["metadata_only"])
        self.assertFalse(discovered["bulk_fetch_performed"])
        self.assertEqual([item["candidate_id"] for item in discovered["candidates"]], ["candidate-000"])
        self.assertEqual(len(discovered["rejected"]), 3)

    def test_manifest_freezes_lexical_disjoint_primary_and_surplus_clusters(self):
        discovered = discover_candidates([candidate(index) for index in range(116)], self.preregistration)
        manifest = build_manifest(discovered, self.preregistration)
        self.assertEqual(manifest["status"], "frozen_metadata_only")
        self.assertTrue(manifest["selection_outcome_blind"])
        self.assertEqual(len(manifest["frozen_clusters"]), 96)
        self.assertEqual(len(manifest["surplus_clusters"]), 20)
        validate_manifest(manifest, self.preregistration)

    def test_manifest_stale_or_overlapping_clusters_fail_closed(self):
        discovered = discover_candidates([candidate(index) for index in range(116)], self.preregistration)
        manifest = build_manifest(discovered, self.preregistration)
        stale = copy.deepcopy(manifest)
        stale["preregistration_sha256"] = "f" * 64
        with self.assertRaises(CohortValidationError):
            validate_manifest(stale, self.preregistration)
        overlapping = copy.deepcopy(manifest)
        overlapping["surplus_clusters"][0]["station_ids"] = overlapping["frozen_clusters"][0]["station_ids"]
        with self.assertRaises(CohortValidationError):
            validate_manifest(overlapping, self.preregistration)

    def test_exact_power_contract_is_frozen(self):
        expected = {"68": 96, "90": 72, "95": 62}
        self.assertEqual(self.preregistration["power"]["required_evaluable_clusters"], expected)
        result = exact_binomial_noninferiority(61, 96, 0.68, self.preregistration["power"]["per_level_alpha"])
        self.assertEqual(result["status"], "pass")

    def test_combined_slope_difference_uses_direct_factor_variance_and_levels(self):
        result = score_slope_difference(
            {
                "insar_slope_difference": 1.25,
                "gnss_slope_difference": 0.75,
                "insar_difference_variance": 4.0,
                "gnss_slope_variance": 9.0,
                "sensor_cross_covariance": 0.0,
                "baseline_sigma": {"68": 5.0, "90": 5.0, "95": 5.0},
            }
        )
        self.assertEqual(result["difference"], 0.5)
        self.assertEqual(result["variance"], 13.0)
        self.assertEqual(set(result["levels"]), {"68", "90", "95"})

    def test_nonzero_sensor_cross_covariance_is_rejected(self):
        observation = {
            "insar_slope_difference": 1.0,
            "gnss_slope_difference": 0.0,
            "insar_difference_variance": 1.0,
            "gnss_slope_variance": 1.0,
            "sensor_cross_covariance": 0.1,
            "baseline_sigma": {"68": 1.0, "90": 1.0, "95": 1.0},
        }
        with self.assertRaises(CohortValidationError):
            score_slope_difference(observation)

    def test_receipt_stale_generation_and_identity_fail_closed(self):
        manifest = build_manifest(
            discover_candidates([candidate(index) for index in range(116)], self.preregistration),
            self.preregistration,
        )
        receipt = {
            "schema": "dolphinrust.temporal_covariance.heldout_receipt",
            "schema_version": 1,
            "outcomes_present": True,
            "one_shot_unblinding": True,
            "generation_id": "stale-generation",
            "preregistration_sha256": canonical_digest(self.preregistration),
            "manifest_sha256": canonical_digest(manifest),
            "scope_hash": canonical_digest(self.preregistration["field_scope"]),
            "factor_binding": self.preregistration["factor_binding"],
            "hashes": {field: "0" * 64 for field in ("binary_sha256", "scorer_sha256", "preregistration_sha256", "manifest_sha256", "factor_scope_sha256", "gnss_catalog_sha256")},
        }
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("stale" in error for error in report["errors"]))

    def test_candidate_pool_shortfall_is_not_evaluable_not_a_pass(self):
        manifest = build_manifest(
            discover_candidates([candidate(index) for index in range(10)], self.preregistration),
            self.preregistration,
        )
        self.assertEqual(manifest["status"], "not_evaluable_candidate_pool")
        receipt = {"schema": "not-used"}
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "not_evaluable")


if __name__ == "__main__":
    unittest.main()
