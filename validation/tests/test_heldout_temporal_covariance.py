import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validation.heldout_temporal_covariance.cohort import (
    CohortValidationError,
    build_manifest,
    canonical_digest,
    discover_candidates,
    validate_freeze_receipt,
    validate_manifest,
)
from validation.heldout_temporal_covariance.scorer import (
    exact_binomial_noninferiority,
    holm_step_down,
    score_receipt,
    score_slope_difference,
)
from validation.score_temporal_covariance_holdout import bind_factor_files, build_scored_result
from validation.run_temporal_covariance_holdout_cluster import scorer_source_sha256, validate_completed_receipt


VALIDATION = Path(__file__).parents[1]
PREREGISTRATION_PATH = VALIDATION / "temporal_covariance_heldout_preregistration.json"
FROZEN_MANIFEST_PATH = VALIDATION / "temporal_covariance_heldout_cohort_manifest.json"
FREEZE_RECEIPT_PATH = VALIDATION / "temporal_covariance_heldout_cohort_freeze_receipt.json"


def candidate(index: int, query_digest="dc4268d915af73dbab9d6cde90ffb8dbb5953d8df8841a0cd66d415fa534f74b", **overrides):
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
        "query_digest": query_digest,
    }
    value.update(overrides)
    return value


def outcome_cluster(candidate_value, preregistration, status="pass", difference=0.0):
    scope = {
        "target_station_id": candidate_value["station_ids"][0],
        "control_station_id": candidate_value["station_ids"][1],
        "target_station_pixel": [10, 10],
        "control_station_pixel": [20, 20],
        "schema_version": 4,
        "method_version": 1,
        "method": "reference_specific_influence_v1",
        "calibration_scope": "calibrated_scope_match",
        "common_dates_sha256": "4" * 64,
        "acquisition_days_sha256": "5" * 64,
        "geotransform_sha256": "6" * 64,
        "window": "frozen",
        "overlap": "coincident",
        "distance": "same_frame",
        "reference_signature_sha256": "9" * 64,
        "source_replay_sha256": "a" * 64,
        "l2_map_sha256": "b" * 64,
        "mask_sha256": "c" * 64,
        "source_model_sha256": "e" * 64,
        "effective_looks_sha256": "f" * 64,
        "support_sha256": "0" * 64,
        "correction_order_sha256": "1" * 64,
        "unwrap_branch_sha256": "2" * 64,
        "burst_ownership_sha256": "3" * 64,
        "runtime_resource_receipt_sha256": "7" * 64,
        "approximation_receipt_sha256": "8" * 64,
        "resource_receipt_sha256": "9" * 64,
        "review_receipt_sha256": "a" * 64,
        "method_manifest_sha256": "b" * 64,
        "calibration_scope_sha256": "c" * 64,
        "burst_id": candidate_value["burst_id"],
        "grid_sha256": "d" * 64,
        "units": "meters",
    }
    cluster = {
        "cluster_id": candidate_value["candidate_id"],
        "station_ids": candidate_value["station_ids"],
        "burst_id": candidate_value["burst_id"],
        "site_id": candidate_value["site_id"],
        "status": status,
    }
    if status == "not_evaluable":
        cluster["reason_code"] = "gnss_solution_missing"
        return cluster
    if status == "not_used":
        return cluster
    cluster.update(
        {
            "difference_covariance": {
                **copy.deepcopy(preregistration["factor_binding"]),
                "calibrated_scope_match": "calibrated_scope_match",
                "operator_sha256": "0" * 64,
                "operator_manifest_sha256": "1" * 64,
                "persisted_factor_sha256": "2" * 64,
                "persisted_factor_manifest_sha256": "3" * 64,
                "factor_sha256": "2" * 64,
                "scope_sha256": canonical_digest(scope),
                "scope": scope,
            },
            "gnss_provenance": {
                "solution_sources": {
                    candidate_value["station_ids"][0]: {"sha256": "6" * 64},
                    candidate_value["station_ids"][1]: {"sha256": "6" * 64},
                },
                "solution_sha256": "6" * 64,
                "coordinate_frame": "ENU",
                "los_source": "run_specific_sourced_los_components",
                "los_sha256": "7" * 64,
                "station_los_vectors": {
                    candidate_value["station_ids"][0]: [0.0, 0.0, 1.0],
                    candidate_value["station_ids"][1]: [0.0, 0.0, 1.0],
                },
                "projection_convention": "signed_ground_to_sensor_los_dot_enu",
                "epoch_zero_reference_sha256": "8" * 64,
                "covariance_projection": "u_transpose_C_u",
            },
            "observation": {
                "insar_slope_difference": difference,
                "gnss_slope_difference": 0.0,
                "insar_difference_variance": 1.0,
                "gnss_slope_variance": 1.0,
                "sensor_cross_covariance": 0.0,
                "baseline_sigma": {"68": 10.0, "90": 10.0, "95": 10.0},
            },
            "estimator": {"binary_sha256": "d" * 64},
        }
    )
    return cluster


def receipt_for_manifest(preregistration, manifest, primary_not_evaluable=(), surplus_not_evaluable=(), difference=0.0):
    primary_ids = {candidate_value["candidate_id"] for candidate_value in manifest["frozen_clusters"]}
    surplus_ids = {candidate_value["candidate_id"] for candidate_value in manifest["surplus_clusters"]}
    clusters = []
    for candidate_value in manifest["frozen_clusters"]:
        status = "not_evaluable" if candidate_value["candidate_id"] in primary_not_evaluable else "pass"
        clusters.append(outcome_cluster(candidate_value, preregistration, status, difference))
    surplus_order = sorted(candidate_value["candidate_id"] for candidate_value in manifest["surplus_clusters"])
    usable = [candidate_id for candidate_id in surplus_order if candidate_id not in surplus_not_evaluable]
    used = usable[: len(primary_not_evaluable)]
    for candidate_value in manifest["surplus_clusters"]:
        candidate_id = candidate_value["candidate_id"]
        status = "not_evaluable" if candidate_id in surplus_not_evaluable else ("pass" if candidate_id in used else "not_used")
        clusters.append(outcome_cluster(candidate_value, preregistration, status, difference))
    attrited = sorted(primary_not_evaluable)
    unused = [candidate_id for candidate_id in surplus_order if candidate_id not in used]
    receipt = {
        "schema": "dolphinrust.temporal_covariance.heldout_receipt",
        "schema_version": 1,
        "outcomes_present": True,
        "one_shot_unblinding": True,
        "cohort_id": preregistration["cohort_id"],
        "generation_id": preregistration["generation_id"],
        "preregistration_sha256": canonical_digest(preregistration),
        "manifest_sha256": canonical_digest(manifest),
        "scope_hash": canonical_digest(preregistration["field_scope"]),
        "calibrated_scope_match": True,
        "factor_binding": preregistration["factor_binding"],
        "hashes": {
            "binary_sha256": "0" * 64,
            "scorer_sha256": "1" * 64,
            "preregistration_sha256": canonical_digest(preregistration),
            "manifest_sha256": canonical_digest(manifest),
            "factor_scope_sha256": canonical_digest(preregistration["factor_binding"]),
            "gnss_catalog_sha256": "2" * 64,
            "approximation_receipt_sha256": "3" * 64,
            "resource_receipt_sha256": "4" * 64,
            "calibration_scope_receipt_sha256": "5" * 64,
            "review_receipt_sha256": "6" * 64,
            "operator_sha256": "0" * 64,
            "operator_manifest_sha256": "1" * 64,
            "persisted_factor_sha256": "2" * 64,
            "persisted_factor_manifest_sha256": "3" * 64,
        },
        "cluster_counts": {
            "primary": len(manifest["frozen_clusters"]),
            "surplus": len(manifest["surplus_clusters"]),
            "executed": sum(cluster["status"] != "not_used" for cluster in clusters),
            "evaluable": sum(
                cluster["status"] in {"pass", "fail"} for cluster in clusters
            ),
        },
        "attrition": {
            "attrited_primary_ids": attrited,
            "used_surplus_ids": used,
            "unused_surplus_ids": unused,
            "reasons_by_cluster": {candidate_id: "gnss_solution_missing" for candidate_id in attrited + sorted(surplus_not_evaluable)},
        },
        "clusters": clusters,
    }
    passing = [cluster for cluster in clusters if cluster["status"] in {"pass", "fail"}]
    for field in (
        "operator_sha256",
        "operator_manifest_sha256",
        "persisted_factor_sha256",
        "persisted_factor_manifest_sha256",
    ):
        receipt["hashes"][field] = canonical_digest(
            {
                cluster["cluster_id"]: cluster["difference_covariance"][field]
                for cluster in passing
            }
        )
    receipt["hashes"]["gnss_catalog_sha256"] = canonical_digest(
        {
            cluster["cluster_id"]: cluster["gnss_provenance"]["solution_sha256"]
            for cluster in passing
        }
    )
    for output, source in (
        ("approximation_receipt_sha256", "approximation_receipt_sha256"),
        ("resource_receipt_sha256", "resource_receipt_sha256"),
        ("calibration_scope_receipt_sha256", "method_manifest_sha256"),
        ("review_receipt_sha256", "review_receipt_sha256"),
    ):
        receipt["hashes"][output] = canonical_digest(
            {
                cluster["cluster_id"]: cluster["difference_covariance"]["scope"][source]
                for cluster in passing
            }
        )
    return receipt


class HeldoutCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))

    def test_factor_binding_matches_current_production_artifact(self):
        output = self.preregistration["factor_binding"]["output_factor"]
        self.assertEqual(output["schema_version"], 4)
        self.assertEqual(output["artifact_hdf5"], "referenced_displacement_covariance_factor.h5")
        self.assertEqual(output["artifact_manifest"], "referenced_displacement_covariance_provenance.json")

    def test_completed_receipt_requires_exact_run_product_and_binary_identity(self):
        manifest = build_manifest(
            discover_candidates(
                [
                    candidate(
                        index,
                        query_digest=self.preregistration["candidate_query"][
                            "query_digest"
                        ],
                    )
                    for index in range(116)
                ],
                self.preregistration,
            ),
            self.preregistration,
        )
        receipt = receipt_for_manifest(self.preregistration, manifest)
        product_identities = {
            cluster["cluster_id"]: canonical_digest(cluster["cluster_id"])
            for cluster in receipt["clusters"]
        }
        identity = {
            "binary_sha256": "d" * 64,
            "run_plan_sha256": "e" * 64,
        }
        identity_digest = canonical_digest(identity)
        for cluster in receipt["clusters"]:
            if cluster["status"] != "not_used":
                cluster["run_identity_sha256"] = identity_digest
                cluster["product_identity_sha256"] = product_identities[
                    cluster["cluster_id"]
                ]
        receipt["run_identity"] = identity
        receipt["run_identity_sha256"] = identity_digest
        receipt["hashes"]["binary_sha256"] = identity["binary_sha256"]
        receipt["hashes"]["scorer_sha256"] = scorer_source_sha256()
        validate_completed_receipt(
            receipt,
            self.preregistration,
            manifest,
            identity,
            product_identities,
        )
        stale = copy.deepcopy(receipt)
        stale["clusters"][0]["product_identity_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "product/run identity"):
            validate_completed_receipt(
                stale,
                self.preregistration,
                manifest,
                identity,
                product_identities,
            )

    def test_receipt_scores_to_exact_promotion_evidence_shape(self):
        manifest = build_manifest(
            discover_candidates(
                [
                    candidate(
                        index,
                        query_digest=self.preregistration["candidate_query"][
                            "query_digest"
                        ],
                    )
                    for index in range(116)
                ],
                self.preregistration,
            ),
            self.preregistration,
        )
        receipt = receipt_for_manifest(self.preregistration, manifest)
        receipt["run_identity"] = {"freeze_receipt_sha256": "f" * 64}
        receipt["run_identity_sha256"] = canonical_digest(receipt["run_identity"])
        score = score_receipt(self.preregistration, manifest, receipt)
        shaped = build_scored_result(
            score,
            receipt,
            self.preregistration,
            manifest,
            receipt_file_sha256="a" * 64,
            manifest_file_sha256="b" * 64,
        )
        self.assertEqual(shaped["schema_version"], 1)
        self.assertEqual(shaped["cohort_id"], self.preregistration["cohort_id"])
        self.assertEqual(shaped["primary_cluster_count"], 96)
        self.assertEqual(shaped["surplus_cluster_count"], 20)
        self.assertEqual(shaped["evaluated_clusters"], 96)
        self.assertEqual(shaped["heldout_receipt_sha256"], "a" * 64)
        self.assertEqual(
            set(shaped),
            {
                "schema", "schema_version", "cohort_id", "manifest_file_sha256",
                "manifest_sha256", "freeze_receipt_sha256", "factor_scope_sha256",
                "heldout_receipt_sha256", "primary_cluster_count", "surplus_cluster_count",
                "status", "errors", "levels", "evaluated_clusters", "emission_rate",
                "attrited_primary_ids", "used_surplus_ids", "unused_surplus_ids",
                "reasons_by_cluster",
            },
        )

    def test_supplied_factor_bytes_are_bound_before_heldout_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factor = root / "referenced_displacement_covariance_factor.h5"
            provenance = root / "referenced_displacement_covariance_provenance.json"
            factor.write_bytes(b"production-factor-v4")
            factor_sha256 = hashlib.sha256(factor.read_bytes()).hexdigest()
            manifest = {
                "schema_version": 3,
                "method": "reference_specific_influence_v1",
                "method_version": 1,
                "hdf5_file": factor.name,
                "hdf5_bytes": factor.stat().st_size,
                "hdf5_sha256": factor_sha256,
                "calibration_scope": "calibrated_scope_match",
            }
            provenance.write_text(json.dumps(manifest), encoding="utf-8")
            receipt = {"hashes": {
                "persisted_factor_sha256": factor_sha256,
                "persisted_factor_manifest_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
            }}
            bind_factor_files(receipt, factor, provenance)
            factor.write_bytes(b"tampered-factor-v4")
            with self.assertRaisesRegex(ValueError, "factor hash differs"):
                bind_factor_files(receipt, factor, provenance)

    def test_metadata_discovery_excludes_exposed_and_outcome_records(self):
        records = [
            candidate(0, query_digest=self.preregistration["candidate_query"]["query_digest"]),
            candidate(1, query_digest=self.preregistration["candidate_query"]["query_digest"], site_id="fresno"),
            candidate(2, query_digest=self.preregistration["candidate_query"]["query_digest"], station_ids=["A002", "MMX1"]),
            candidate(3, query_digest=self.preregistration["candidate_query"]["query_digest"], displacement=[1.0]),
        ]
        discovered = discover_candidates(records, self.preregistration)
        self.assertTrue(discovered["metadata_only"])
        self.assertFalse(discovered["bulk_fetch_performed"])
        self.assertEqual([item["candidate_id"] for item in discovered["candidates"]], ["candidate-000"])
        self.assertEqual(len(discovered["rejected"]), 3)

    def test_query_identity_is_a_real_sha256(self):
        self.assertEqual(
            self.preregistration["candidate_query"]["query_digest"],
            canonical_digest(self.preregistration["candidate_query"]["query"]),
        )
        rejected = discover_candidates([candidate(0, query_digest="f53-06-metadata-query-v1")], self.preregistration)
        self.assertEqual(rejected["candidates"], [])

    def test_manifest_freezes_lexical_disjoint_primary_and_surplus_clusters(self):
        discovered = discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(116)], self.preregistration)
        manifest = build_manifest(discovered, self.preregistration)
        self.assertEqual(manifest["status"], "frozen_metadata_only")
        self.assertTrue(manifest["selection_outcome_blind"])
        self.assertEqual(len(manifest["frozen_clusters"]), 96)
        self.assertEqual(len(manifest["surplus_clusters"]), 20)
        validate_manifest(manifest, self.preregistration)

    def test_manifest_stale_or_overlapping_clusters_fail_closed(self):
        discovered = discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(116)], self.preregistration)
        manifest = build_manifest(discovered, self.preregistration)
        stale = copy.deepcopy(manifest)
        stale["preregistration_sha256"] = "f" * 64
        with self.assertRaises(CohortValidationError):
            validate_manifest(stale, self.preregistration)
        overlapping = copy.deepcopy(manifest)
        overlapping["surplus_clusters"][0]["station_ids"] = overlapping["frozen_clusters"][0]["station_ids"]
        with self.assertRaises(CohortValidationError):
            validate_manifest(overlapping, self.preregistration)

    def test_manifest_rejects_outcome_fields_and_skips_overlapping_surplus(self):
        records = [candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(117)]
        records[1]["station_ids"] = records[0]["station_ids"]
        discovered = discover_candidates(records, self.preregistration)
        manifest = build_manifest(discovered, self.preregistration)
        self.assertEqual(len(manifest["frozen_clusters"]), 96)
        self.assertEqual(len(manifest["surplus_clusters"]), 20)
        self.assertEqual(manifest["excluded_after_selection"], ["candidate-001"])
        validate_manifest(manifest, self.preregistration)
        exposed = copy.deepcopy(manifest)
        exposed["gnss_series"] = [0.0]
        with self.assertRaisesRegex(CohortValidationError, "schema"):
            validate_manifest(exposed, self.preregistration)

    def test_relative_orbit_is_composite_provenance_not_a_global_exclusion(self):
        records = [
            candidate(
                index,
                query_digest=self.preregistration["candidate_query"]["query_digest"],
                orbit_id="ascending-r001",
            )
            for index in range(116)
        ]
        manifest = build_manifest(
            discover_candidates(records, self.preregistration), self.preregistration
        )
        self.assertEqual(manifest["status"], "frozen_metadata_only")
        self.assertEqual(
            manifest["selection_algorithm"],
            "lexical_candidate_id_greedy_independent_burst_site_v2",
        )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(len(manifest["frozen_clusters"]), 96)
        self.assertEqual(len(manifest["surplus_clusters"]), 20)
        validate_manifest(manifest, self.preregistration)

        old_v1 = copy.deepcopy(manifest)
        old_v1["schema_version"] = 1
        old_v1["selection_algorithm"] = "lexical_candidate_id_greedy_disjoint_v1"
        with self.assertRaisesRegex(CohortValidationError, "schema/version mismatch"):
            validate_manifest(old_v1, self.preregistration)

    def test_burst_footprint_site_and_station_overlap_remain_excluded(self):
        query_digest = self.preregistration["candidate_query"]["query_digest"]
        for field in ("burst_id", "footprint_id", "site_id", "station_ids"):
            with self.subTest(field=field):
                records = [candidate(index, query_digest=query_digest) for index in range(117)]
                records[1][field] = records[0][field]
                manifest = build_manifest(
                    discover_candidates(records, self.preregistration),
                    self.preregistration,
                )
                self.assertEqual(manifest["excluded_after_selection"], ["candidate-001"])
                validate_manifest(manifest, self.preregistration)

    def test_persisted_freeze_receipt_recomputes_local_hashes_and_rejects_mutation(self):
        receipt = json.loads(FREEZE_RECEIPT_PATH.read_text(encoding="utf-8"))
        validate_freeze_receipt(
            receipt,
            FROZEN_MANIFEST_PATH,
            PREREGISTRATION_PATH,
        )

        changed_receipt = copy.deepcopy(receipt)
        changed_receipt["outcomes_present"] = True
        with self.assertRaisesRegex(CohortValidationError, "outcome-blind"):
            validate_freeze_receipt(
                changed_receipt,
                FROZEN_MANIFEST_PATH,
                PREREGISTRATION_PATH,
            )

        with tempfile.TemporaryDirectory() as directory:
            changed_manifest = Path(directory) / FROZEN_MANIFEST_PATH.name
            changed_manifest.write_bytes(FROZEN_MANIFEST_PATH.read_bytes() + b"\n")
            with self.assertRaisesRegex(CohortValidationError, "manifest file hash"):
                validate_freeze_receipt(
                    receipt,
                    changed_manifest,
                    PREREGISTRATION_PATH,
                )

    def test_exact_power_contract_is_frozen(self):
        expected = {"68": 96, "90": 72, "95": 62}
        self.assertEqual(self.preregistration["power"]["required_evaluable_clusters"], expected)
        result = exact_binomial_noninferiority(61, 96, 0.68, self.preregistration["power"]["familywise_alpha"] / 3)
        self.assertEqual(result["status"], "pass")

    def test_holm_step_down_preserves_order_and_stops_after_nonrejection(self):
        result = holm_step_down({"68": 0.001, "90": 0.03, "95": 0.03}, 0.05)
        self.assertEqual(result["ordered_levels"], ["68", "90", "95"])
        self.assertTrue(result["reject"]["68"])
        self.assertFalse(result["reject"]["90"])
        self.assertFalse(result["reject"]["95"])

    def test_deterministic_surplus_fill_can_keep_power_evaluable(self):
        manifest = build_manifest(
            discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(116)], self.preregistration),
            self.preregistration,
        )
        primary_id = manifest["frozen_clusters"][0]["candidate_id"]
        receipt = receipt_for_manifest(self.preregistration, manifest, primary_not_evaluable={primary_id})
        for cluster in receipt["clusters"][:24]:
            if cluster["status"] == "pass":
                cluster["observation"]["insar_slope_difference"] = 2.0
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["attrited_primary_ids"], [primary_id])
        self.assertEqual(len(report["used_surplus_ids"]), 1)
        self.assertEqual(report["emission_rate"], 1.0)

    def test_insufficient_surplus_is_not_evaluable(self):
        manifest = build_manifest(
            discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(116)], self.preregistration),
            self.preregistration,
        )
        primary_id = manifest["frozen_clusters"][0]["candidate_id"]
        surplus_ids = {candidate_value["candidate_id"] for candidate_value in manifest["surplus_clusters"]}
        receipt = receipt_for_manifest(self.preregistration, manifest, primary_not_evaluable={primary_id}, surplus_not_evaluable=surplus_ids)
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "not_evaluable")
        self.assertIn("insufficient", report["errors"][0])

    def test_absolute_coverage_gate_fails_even_when_intervals_emit(self):
        manifest = build_manifest(
            discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(116)], self.preregistration),
            self.preregistration,
        )
        receipt = receipt_for_manifest(self.preregistration, manifest, difference=1.0)
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any(not level.get("coverage_absolute_gate", True) for level in report["levels"].values()))

    def test_uncalibrated_factor_is_rejected(self):
        manifest = build_manifest(
            discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(116)], self.preregistration),
            self.preregistration,
        )
        receipt = receipt_for_manifest(self.preregistration, manifest)
        receipt["clusters"][0]["difference_covariance"]["calibrated_scope_match"] = "uncalibrated"
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("calibrated_scope_match" in error for error in report["errors"]))

    def test_stale_factor_scope_receipt_hash_is_rejected(self):
        manifest = build_manifest(
            discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(116)], self.preregistration),
            self.preregistration,
        )
        receipt = receipt_for_manifest(self.preregistration, manifest)
        receipt["hashes"]["approximation_receipt_sha256"] = "f" * 64
        receipt["calibrated_scope_match"] = False
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("calibrated scope" in error for error in report["errors"]))

    def test_actual_factor_evidence_bundle_hashes_reject_surrogate_mapping(self):
        manifest = build_manifest(
            discover_candidates(
                [
                    candidate(
                        index,
                        query_digest=self.preregistration["candidate_query"][
                            "query_digest"
                        ],
                    )
                    for index in range(116)
                ],
                self.preregistration,
            ),
            self.preregistration,
        )
        receipt = receipt_for_manifest(self.preregistration, manifest)
        receipt["hashes"]["review_receipt_sha256"] = canonical_digest(
            {
                cluster["cluster_id"]: cluster["difference_covariance"]["scope"][
                    "reference_signature_sha256"
                ]
                for cluster in receipt["clusters"]
                if cluster["status"] in {"pass", "fail"}
            }
        )
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(
            any("review_receipt_sha256" in error for error in report["errors"])
        )

    def test_evaluable_fail_is_included_in_factor_and_gnss_bundles(self):
        manifest = build_manifest(
            discover_candidates(
                [
                    candidate(
                        index,
                        query_digest=self.preregistration["candidate_query"][
                            "query_digest"
                        ],
                    )
                    for index in range(116)
                ],
                self.preregistration,
            ),
            self.preregistration,
        )
        receipt = receipt_for_manifest(self.preregistration, manifest)
        receipt["clusters"][0]["status"] = "fail"
        for field in (
            "operator_sha256",
            "operator_manifest_sha256",
            "persisted_factor_sha256",
            "persisted_factor_manifest_sha256",
        ):
            receipt["hashes"][field] = canonical_digest(
                {
                    cluster["cluster_id"]: cluster["difference_covariance"][field]
                    for cluster in receipt["clusters"]
                    if cluster["status"] in {"pass", "fail"}
                }
            )
        receipt["hashes"]["gnss_catalog_sha256"] = canonical_digest(
            {
                cluster["cluster_id"]: cluster["gnss_provenance"]["solution_sha256"]
                for cluster in receipt["clusters"]
                if cluster["status"] in {"pass", "fail"}
            }
        )
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["errors"], ["at least one frozen primary cluster reported fail"])

    def test_cross_wired_output_method_or_manifest_digest_is_rejected(self):
        manifest = build_manifest(
            discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(116)], self.preregistration),
            self.preregistration,
        )
        receipt = receipt_for_manifest(self.preregistration, manifest)
        receipt["clusters"][0]["difference_covariance"]["output_factor"]["method"] = "sequential_source_dag_v1"
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("persistence identity" in error for error in report["errors"]))
        receipt = receipt_for_manifest(self.preregistration, manifest)
        receipt["hashes"]["persisted_factor_manifest_sha256"] = "f" * 64
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("cross-wired" in error for error in report["errors"]))

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
            discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(116)], self.preregistration),
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
            discover_candidates([candidate(index, query_digest=self.preregistration["candidate_query"]["query_digest"]) for index in range(10)], self.preregistration),
            self.preregistration,
        )
        self.assertEqual(manifest["status"], "not_evaluable_candidate_pool")
        receipt = {"schema": "not-used"}
        report = score_receipt(self.preregistration, manifest, receipt)
        self.assertEqual(report["status"], "not_evaluable")


if __name__ == "__main__":
    unittest.main()
