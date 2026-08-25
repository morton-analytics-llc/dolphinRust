import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from validation.score_spatial_covariance import (
    ATTEMPT_KEYS,
    FROZEN_ATTEMPT_COUNT,
    FROZEN_CELL_COUNT,
    FROZEN_CELL_SUMMARY_COMPONENT_BYTES,
    FROZEN_MAX_SHARD_BYTES,
    FROZEN_MAX_RUN_MANIFEST_BYTES,
    FROZEN_MAX_RECORD_BYTES,
    FROZEN_MAX_SHARD_MANIFEST_BYTES,
    FROZEN_RETAINED_SIZE_BOUND_BYTES,
    FROZEN_SEED_COUNT,
    FROZEN_SHARD_COUNT,
    PASS,
    CellAccumulator,
    SchemaError,
    ShardSpec,
    _CellSummarySink,
    _dgp_cell_ordinal,
    _expected_seed_hash,
    _fixed_l2_reconstruction_bound,
    _generate_complex_source,
    _growth_exponent,
    _raw_source_digest,
    _read_hashed_json_record,
    _read_single_json_record,
    _validate_resources,
    derive_dense_joint_oracle,
    expected_cell_ids,
    expected_empty_support,
    expected_seed_count,
    expected_production_artifact_provenance,
    independently_recompute_metrics,
    load_preregistration,
    numeric_digest,
    portable_dgp_key_sha256,
    portable_normal,
    portable_table_coverage,
    preregistration_digest,
    regenerate_frozen_attempt_inputs,
    sha256_json,
    validate_cell_summary,
    validate_direct_pair_variance_order,
    validate_production_parity_fixture,
    validate_preregistration,
)
from validation.spatial_covariance_simulation import (
    _load_bounded_json,
    build_run_manifest,
    commit_cell_transport,
    commit_output_shard,
    committed_shard_matches,
    compact_json_line,
    capture_benchmark_stdout,
    iter_attempt_requests,
    prepare_input_shard,
)


VALIDATION = Path(__file__).parents[1]
PREREGISTRATION = VALIDATION / "spatial_covariance_preregistration.json"
CODE = "a" * 64
BINARY = "b" * 64
CELL = "hw_1x1|stride_1|rect|interior|coincident|one_block|emi|well_separated|independent_complex_looks"
SUPPORTED_CELL = "hw_1x1|stride_1|rect|interior|shared_75_positive|one_block|emi|well_separated|independent_complex_looks"
RISK_CELL = "hw_7x14|stride_4|glrt_frozen|interior|shared_75_positive|four_blocks|evd|near_tie|spatial_correlation_stress"
MASKED_CELL = "hw_1x1|stride_1|rect|masked|coincident|one_block|emi|well_separated|independent_complex_looks"


class SpatialCovarianceValidationV4Tests(unittest.TestCase):
    benchmark_stdout_by_scope = {}
    @classmethod
    def setUpClass(cls):
        cls.preregistration = load_preregistration(PREREGISTRATION)
        cls.artifact_directory = tempfile.TemporaryDirectory()
        cls.artifact_root = Path(cls.artifact_directory.name)
        cls.addClassCleanup(cls.artifact_directory.cleanup)

    def _accumulator(self, cell_id, ordinal=0, seeds=1):
        return CellAccumulator(
            self.preregistration, cell_id, ordinal, seeds, CODE, BINARY,
            artifact_root=self.artifact_root,
        )

    def _attempt(self, cell_id, ordinal, seed_index, masked=False, artifact_root=None):
        artifact_root = self.artifact_root if artifact_root is None else Path(artifact_root)
        labels = dict(zip(self.preregistration["matrix_contract"]["dimension_order"], cell_id.split("|")))
        generator = self.preregistration["generator"]
        frozen = regenerate_frozen_attempt_inputs(self.preregistration, cell_id, seed_index)
        tied = labels["eigen_stress"] == "tied_eigenvalue" and labels["position"] != "masked"
        emitted = not masked and not tied
        sign = "zero"
        influence = 0.0
        if labels["pair_geometry"].endswith("_positive"):
            sign = "positive"
        elif labels["pair_geometry"].endswith("_negative"):
            sign = "negative"
        elif labels["pair_geometry"].startswith("disjoint_"):
            sign = "none"
        if sign in {"positive", "negative"}:
            influence = (
                frozen["target_global_loading_mean"]
                * frozen["reference_global_loading_mean"]
            )
        attempt = {
            "schema": "dolphinrust.spatial-covariance.attempt-evidence/4",
            "cell_id": cell_id,
            "cell_ordinal": ordinal,
            "seed_index": seed_index,
            "seed_sha256": _expected_seed_hash(self.preregistration, cell_id, seed_index),
            "status": "masked_target" if masked else "singular_local_information" if tied else "valid",
            "emitted": emitted,
            "factor_emitted": emitted,
            "raw_input_shape": frozen["raw_input_shape"],
            "raw_input_value_count": frozen["raw_input_value_count"],
            "raw_input_sha256": frozen["raw_input_sha256"],
            "target_raw_input_sha256": frozen["target_raw_input_sha256"],
            "reference_raw_input_sha256": frozen["reference_raw_input_sha256"],
            "sequential_ancestry_sha256": frozen["sequential_ancestry_sha256"],
            "raw_dgp_identity_sha256": frozen["raw_dgp_identity_sha256"],
            "latent_history_sha256": frozen["latent_history_sha256"],
            "estimate_sha256": "0" * 64,
            "predicted_covariance_sha256": "0" * 64,
            "date_axis_sha256": frozen["date_axis_sha256"],
            "generator_hash": sha256_json(generator),
            "config_hash": sha256_json(generator),
            "source_model_hash": sha256_json(generator["source_centered_empirical"]),
            "target_coordinate": frozen["target_coordinate"],
            "reference_coordinate": frozen["reference_coordinate"],
            "target_support_sha256": frozen["target_support_sha256"],
            "reference_support_sha256": frozen["reference_support_sha256"],
            "target_source_count": frozen["target_source_count"],
            "reference_source_count": frozen["reference_source_count"],
            "intersection_source_count": frozen["intersection_source_count"],
            "union_source_count": frozen["union_source_count"],
            "realized_overlap_jaccard": (
                frozen["intersection_source_count"] / frozen["union_source_count"]
                if frozen["union_source_count"]
                else 0.0
            ),
            "signed_cross_influence": None if masked or tied else influence,
            "signed_influence_sign": sign,
            "effective_looks_fraction": frozen["effective_looks_fraction"],
            "effective_looks_application": "source_influence_joint_contraction_v1",
            "source_correlation_model": frozen["source_correlation_model"],
            "source_correlation_distance_scale_pixels": frozen[
                "source_correlation_distance_scale_pixels"
            ],
            "estimator_branch": "evd" if tied else labels["estimator"],
            "target_estimate_history": None if not emitted else copy.deepcopy(frozen["latent_target_history"]),
            "reference_estimate_history": None if not emitted else copy.deepcopy(frozen["latent_reference_history"]),
            "predicted_difference_covariance": None,
            "production_operator_matrix": None,
            "contrast_weights": None,
            "operator_sha256": "0" * 64,
        }
        if emitted:
            dates = len(frozen["latent_target_history"])
            covariance = copy.deepcopy(frozen["oracle_difference_covariance"])
            attempt["predicted_difference_covariance"] = covariance
            joint = copy.deepcopy(frozen["dense_joint_oracle"])
            attempt["production_operator_matrix"] = copy.deepcopy(joint)
            attempt["contrast_weights"] = [0.0] * (2 * dates)
            attempt["contrast_weights"][dates - 1] = 1.0
            attempt["contrast_weights"][-1] = -1.0
            attempt["estimate_sha256"] = numeric_digest(
                "estimate-history-v4",
                [*attempt["target_estimate_history"], *attempt["reference_estimate_history"]],
            )
            attempt["predicted_covariance_sha256"] = numeric_digest(
                "predicted-difference-covariance-v4",
                [value for row in covariance for value in row],
            )
            attempt["operator_sha256"] = numeric_digest(
                "production-operator-v4", [value for row in joint for value in row]
            )
        self.assertEqual(set(attempt), ATTEMPT_KEYS)
        return attempt

    def _nondifferentiable_attempt(self, cell_id, ordinal, seed_index):
        attempt = self._attempt(cell_id, ordinal, seed_index)
        attempt["status"] = "nondifferentiable_node"
        attempt["emitted"] = False
        attempt["factor_emitted"] = False
        for name in (
            "target_estimate_history", "reference_estimate_history",
            "predicted_difference_covariance", "production_operator_matrix",
            "contrast_weights",
        ):
            attempt[name] = None
        for name in ("estimate_sha256", "predicted_covariance_sha256", "operator_sha256"):
            attempt[name] = "0" * 64
        return attempt

    def _set_operator_and_difference(self, attempt, difference):
        dates = len(difference)
        joint = [[0.0] * (2 * dates) for _ in range(2 * dates)]
        for row in range(dates):
            for column in range(dates):
                joint[row][column] = difference[row][column]
        attempt["production_operator_matrix"] = joint
        attempt["predicted_difference_covariance"] = copy.deepcopy(difference)
        attempt["predicted_covariance_sha256"] = numeric_digest(
            "predicted-difference-covariance-v4", [value for row in difference for value in row]
        )
        attempt["operator_sha256"] = numeric_digest(
            "production-operator-v4", [value for row in joint for value in row]
        )

    def _sync_difference_from_operator(self, attempt):
        joint = attempt["production_operator_matrix"]
        dates = len(joint) // 2
        raw_difference = [
            [
                joint[row][column] + joint[dates + row][dates + column]
                - joint[row][dates + column] - joint[dates + row][column]
                for column in range(dates)
            ]
            for row in range(dates)
        ]
        difference = [
            [0.0 if row == 0 or column == 0 else 0.5 * (raw_difference[row][column] + raw_difference[column][row])
             for column in range(dates)]
            for row in range(dates)
        ]
        attempt["predicted_difference_covariance"] = difference
        attempt["predicted_covariance_sha256"] = numeric_digest(
            "predicted-difference-covariance-v4", [value for row in difference for value in row]
        )

    def _resource_receipts(self):
        matrix = {item["id"]: item for item in self.preregistration["resource_matrix"]}
        peaks = {name: 100_000_000 + matrix[name]["tile_pixels"] * 100 + matrix[name]["dates"] * 10_000 for name in matrix}
        area_names = ("area_128_dates_26", "area_256_dates_26", "area_512_dates_26")
        date_names = ("area_256_dates_13", "area_256_dates_26", "area_256_dates_52")
        area = _growth_exponent([(matrix[name]["tile_pixels"], peaks[name]) for name in area_names])
        dates = _growth_exponent([(matrix[name]["dates"], peaks[name]) for name in date_names])
        sampling = self.preregistration["resource_sampling"]
        result = []
        for name in matrix:
            tile_pixels = matrix[name]["tile_pixels"]
            date_count = matrix[name]["dates"]
            microbatch_pixels = min(tile_pixels, 4096)
            allocation_model = {
                "model": "production-runtime-resource-receipt-v1",
                "source": "spatial_covariance_bench captured stdout",
            }
            command = ["cargo", "run", "--release", "-p", "dolphin-workflows", "--example", "spatial_covariance_bench", "--", "--tile-pixels", str(tile_pixels), "--dates", str(date_count)]
            scope = (tile_pixels, date_count)
            if scope not in self.benchmark_stdout_by_scope:
                completed = subprocess.run(
                    command, check=True, capture_output=True, text=True,
                    cwd=Path(__file__).parents[2],
                )
                self.benchmark_stdout_by_scope[scope] = completed.stdout
            stdout_json = self.benchmark_stdout_by_scope[scope]
            allocation_measurement = json.loads(stdout_json)
            dependency_cone = {
                "model": "spatial-query-cone-v1", "tile_pixels": tile_pixels,
                "date_count": date_count,
                "maximum_sources": allocation_measurement["maximum_sources_per_block"],
                "block_count": allocation_measurement["block_count"],
                "maximum_dependency_depth": allocation_measurement["maximum_dependency_depth"],
                "reference_cone_sources": allocation_measurement["reference_cone_sources"],
            }
            microbatch = {"model": "bounded-microbatch-v1", "microbatch_pixels": microbatch_pixels, "batch_count": (tile_pixels + microbatch_pixels - 1) // microbatch_pixels}
            components = copy.deepcopy(allocation_measurement["allocation_components"])
            observations = []
            for repetition in range(3):
                raw_measurement = {
                    "command": command,
                    "exit_status": 0,
                    "wall_seconds": 1.0,
                    "max_rss_bytes": peaks[name] - 2 + repetition,
                    "rss_sampler": sampling["rss_sampler"],
                    "rss_field": sampling["rss_field"],
                    "os": sampling["os"],
                    "hardware_class": sampling["hardware_class"],
                    "ram_bytes": sampling["ram_bytes"],
                    "tool_versions": {"rustc": "rustc test", "cargo": "cargo test", "uname": "Darwin test"},
                    "stdout_bytes": len(stdout_json.encode()),
                    "stdout_sha256": hashlib.sha256(stdout_json.encode()).hexdigest(),
                    "stdout_json": stdout_json,
                }
                observations.append({"repetition": repetition, "tile_pixels": matrix[name]["tile_pixels"], "date_count": matrix[name]["dates"], "peak_rss_bytes": peaks[name] - 2 + repetition, "wall_seconds": 1.0, "raw_measurement": raw_measurement, "raw_measurement_sha256": sha256_json(raw_measurement)})
            item = {"resource_id": name, "status": PASS, "rss_bytes": peaks[name], "growth_class": "linear", "resource_hash": "", "config_hash": sha256_json(self.preregistration["generator"]), "binary_hash": BINARY, "os": sampling["os"], "hardware_class": sampling["hardware_class"], "ram_bytes": sampling["ram_bytes"], "rss_sampler": sampling["rss_sampler"], "rss_field": sampling["rss_field"], "warmup_runs": sampling["warmup_runs"], "measured_repetitions": sampling["measured_repetitions"], "tool_versions": sampling["tool_versions"], "growth_observation": observations, "area_growth_exponent": area, "date_growth_exponent": dates, "acceptance": sampling["acceptance"], "allocation_model": allocation_model, "allocation_model_sha256": sha256_json(allocation_model), "dependency_cone": dependency_cone, "dependency_cone_sha256": sha256_json(dependency_cone), "microbatch": microbatch, "microbatch_sha256": sha256_json(microbatch), "allocation_components": components}
            item["resource_hash"] = sha256_json({key: value for key, value in item.items() if key != "resource_hash"})
            result.append(item)
        return result

    def test_v4_supersedes_v3_outcome_free_without_changing_matrix(self):
        validate_preregistration(self.preregistration)
        self.assertEqual(self.preregistration["schema_version"], 4)
        self.assertEqual(self.preregistration["supersedes"]["schema_version"], 3)
        self.assertFalse(self.preregistration["outcomes_present"])
        self.assertEqual(len(expected_cell_ids(self.preregistration)), FROZEN_CELL_COUNT)
        self.assertEqual(FROZEN_ATTEMPT_COUNT, 665_220)
        counts = [expected_seed_count(cell_id) for cell_id in expected_cell_ids(self.preregistration)]
        self.assertEqual(counts.count(FROZEN_SEED_COUNT), 133)
        self.assertEqual(counts.count(1), 220)
        self.assertEqual(FROZEN_SHARD_COUNT, 4)

    def test_compact_design_covers_every_pairwise_level_combination(self):
        cells = [cell_id.split("|") for cell_id in expected_cell_ids(self.preregistration)]
        dimensions = self.preregistration["matrix_contract"]["dimension_order"]
        values = [[item["id"] for item in self.preregistration["dimensions"][name]] for name in dimensions]
        for first in range(len(dimensions)):
            for second in range(first + 1, len(dimensions)):
                observed = {(cell[first], cell[second]) for cell in cells}
                expected = {(left, right) for left in values[first] for right in values[second]}
                self.assertTrue(expected <= observed, (dimensions[first], dimensions[second]))
        self.assertEqual(sum(expected_seed_count("|".join(cell)) for cell in cells), FROZEN_ATTEMPT_COUNT)

    def test_supported_interior_disjoint_cells_retain_5000_seeds_except_empty_ks(self):
        for cell_id in expected_cell_ids(self.preregistration):
            labels = dict(zip(self.preregistration["matrix_contract"]["dimension_order"], cell_id.split("|")))
            if (
                labels["position"] == "interior"
                and labels["eigen_stress"] != "tied_eigenvalue"
                and labels["pair_geometry"].startswith("disjoint_")
            ):
                self.assertEqual(
                    expected_seed_count(cell_id),
                    1 if expected_empty_support(cell_id) else FROZEN_SEED_COUNT,
                )

    def test_ephemeral_evidence_is_estimator_history_not_realized_truth(self):
        attempt = self._attempt(SUPPORTED_CELL, 0, 0)
        self.assertEqual(attempt["estimator_branch"], "emi")
        self.assertIn("target_estimate_history", attempt)
        self.assertIn("reference_estimate_history", attempt)
        self.assertIn("predicted_difference_covariance", attempt)
        self.assertIn("latent_history_sha256", attempt)
        self.assertNotIn("truth_matrix", ATTEMPT_KEYS)
        self.assertNotIn("truth_value", ATTEMPT_KEYS)

    def test_estimator_branch_and_latent_history_drift_fail_closed(self):
        attempt = self._attempt(SUPPORTED_CELL, 0, 0)
        attempt["estimator_branch"] = "evd"
        with self.assertRaisesRegex(SchemaError, "estimator"):
            self._accumulator(SUPPORTED_CELL).add(attempt)
        attempt = self._attempt(SUPPORTED_CELL, 0, 0)
        attempt["latent_history_sha256"] = "f" * 64
        with self.assertRaisesRegex(SchemaError, "latent"):
            self._accumulator(SUPPORTED_CELL).add(attempt)

    def test_operator_and_contrast_variance_errors_are_independently_gated(self):
        attempt = self._attempt(SUPPORTED_CELL, 0, 0)
        attempt["production_operator_matrix"][-1][-1] = 2.0
        self._sync_difference_from_operator(attempt)
        attempt["operator_sha256"] = numeric_digest(
            "production-operator-v4",
            [value for row in attempt["production_operator_matrix"] for value in row],
        )
        accumulator = self._accumulator(SUPPORTED_CELL)
        accumulator.add(attempt)
        self.assertEqual(accumulator.finalize()["status"], "fail")

    def test_emi_and_evd_estimator_histories_are_both_exercised_and_bound(self):
        for cell_id, branch in ((SUPPORTED_CELL, "emi"), (RISK_CELL, "evd")):
            attempt = self._attempt(cell_id, 0, 0)
            self.assertEqual(attempt["estimator_branch"], branch)
            accumulator = self._accumulator(cell_id)
            accumulator.add(attempt)
            self.assertEqual(accumulator.finalize()["emitted_seeds"], 1)

    def test_retained_bound_is_derived_and_below_32_gib(self):
        execution = self.preregistration["execution_protocol"]
        derived = (
            FROZEN_CELL_COUNT * execution["max_encoded_cell_summary_bytes"]
            + FROZEN_SHARD_COUNT * execution["max_encoded_shard_manifest_bytes"]
            + execution["max_encoded_run_manifest_bytes"]
            + execution["max_production_hdf5_bytes"]
            + execution["max_production_sidecar_bytes"]
        )
        self.assertEqual(derived, FROZEN_RETAINED_SIZE_BOUND_BYTES)
        self.assertLess(derived, 32 << 30)
        self.assertFalse(execution["retained_attempt_records"])
        self.assertFalse(execution["request_files_retained"])

    def test_final_summary_sink_accepts_exact_full_component_and_rejects_one_byte_over(self):
        self.assertEqual(FROZEN_CELL_SUMMARY_COMPONENT_BYTES, 2_891_776)
        self.assertEqual(FROZEN_RETAINED_SIZE_BOUND_BYTES, 21_831_680)
        encoded = compact_json_line({"cell": 1})
        with tempfile.TemporaryDirectory() as directory:
            exact = _CellSummarySink(Path(directory) / "exact.jsonl")
            self.assertEqual(exact.byte_limit, FROZEN_CELL_SUMMARY_COMPONENT_BYTES)
            exact.open()
            exact.byte_count = FROZEN_CELL_SUMMARY_COMPONENT_BYTES - len(encoded)
            exact.add({"cell": 1})
            self.assertEqual(exact.byte_count, FROZEN_CELL_SUMMARY_COMPONENT_BYTES)
            exact.abort()
            over = _CellSummarySink(Path(directory) / "over.jsonl")
            over.open()
            over.byte_count = FROZEN_CELL_SUMMARY_COMPONENT_BYTES - len(encoded) + 1
            with self.assertRaisesRegex(SchemaError, "full retained cell-summary cap"):
                over.add({"cell": 1})
            over.abort()
        self.assertEqual(FROZEN_MAX_SHARD_BYTES, 819_200)

    def test_exact_request_regeneration_is_ordered_and_stable(self):
        spec = ShardSpec(0, 0, 1, (SUPPORTED_CELL,), (expected_seed_count(SUPPORTED_CELL),))
        generator = iter_attempt_requests(self.preregistration, spec)
        first = [next(generator), next(generator)]
        repeated_generator = iter_attempt_requests(self.preregistration, spec)
        repeated = [next(repeated_generator), next(repeated_generator)]
        self.assertEqual(first, repeated)
        self.assertEqual([item["seed_index"] for item in first], [0, 1])
        self.assertEqual(first[0]["schema"], "dolphinrust.spatial-covariance.attempt/4")

    def test_portable_sha256_lut_normal_has_frozen_cross_language_golden(self):
        key = portable_dgp_key_sha256(
            self.preregistration, 17, 23, -4, 19, 7, "local-signal-real", 3
        )
        value = portable_normal(
            self.preregistration, 17, 23, -4, 19, 7, "local-signal-real", 3
        )
        self.assertEqual(key, "b2abe5cf7624b7311597354b1be7db9934d145cd2cc42f9706eb42a44bb61a7d")
        self.assertEqual(value.hex(), "-0x1.b995f4321d0dep-1")

    def test_portable_complex_source_has_frozen_cross_language_golden(self):
        values = _generate_complex_source(
            self.preregistration, 17, 23, (-4, 19), 4, True, "well_separated", -1.0
        )
        self.assertEqual(
            [(value.real.hex(), value.imag.hex()) for value in values],
            [
                ("-0x1.108cd80000000p-1", "0x1.48532a0000000p-2"),
                ("-0x1.fa28d00000000p-2", "0x1.7a341a0000000p-6"),
                ("-0x1.160ec80000000p-3", "0x1.1d79fe0000000p-1"),
                ("0x1.c2563c0000000p-4", "0x1.4821f00000000p-1"),
            ],
        )
        self.assertEqual(
            _raw_source_digest("portable-golden-v1", [(-4, 19)], {(-4, 19): values}),
            "ba3bdd27980b5e4848b1988aeb6080b0363a0bd7d1340121c424e7948ae5fe1d",
        )

    def test_signed_pair_uses_positive_cell_ordinal_for_matched_draws(self):
        negative = SUPPORTED_CELL.replace("shared_75_positive", "shared_75_negative")
        self.assertEqual(
            _dgp_cell_ordinal(self.preregistration, SUPPORTED_CELL),
            _dgp_cell_ordinal(self.preregistration, negative),
        )

    def test_adaptive_signed_pair_selects_identical_support_from_signed_stack(self):
        positive_cell = SUPPORTED_CELL.replace("rect", "glrt_frozen")
        negative_cell = positive_cell.replace("shared_75_positive", "shared_75_negative")
        positive = regenerate_frozen_attempt_inputs(self.preregistration, positive_cell, 0)
        negative = regenerate_frozen_attempt_inputs(self.preregistration, negative_cell, 0)
        self.assertEqual(positive["target_support_sha256"], negative["target_support_sha256"])
        self.assertEqual(positive["reference_support_sha256"], negative["reference_support_sha256"])

    def test_portable_tables_cover_every_frozen_coordinate_and_date(self):
        coverage = portable_table_coverage(self.preregistration)
        self.assertEqual(coverage["cell_count"], FROZEN_CELL_COUNT)
        self.assertEqual(coverage["coordinate_count"], 29007)
        self.assertEqual(coverage["amplitude_argument_count"], 1106)
        self.assertEqual(coverage["slope_argument_count"], 397)
        self.assertEqual(coverage["date_count"], 20)

    def test_portable_table_asset_is_relative_bounded_and_exact_byte_bound(self):
        contract = json.loads(PREREGISTRATION.read_bytes())
        self.assertNotIn("portable_dgp_tables", contract)
        asset = contract["portable_dgp_asset"]
        self.assertEqual(asset["path"], "spatial_covariance_portable_tables.json")
        raw = (VALIDATION / asset["path"]).read_bytes()
        self.assertEqual(len(raw), asset["byte_count"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), asset["sha256"])
        self.assertEqual(preregistration_digest(self.preregistration), sha256_json(contract))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / PREREGISTRATION.name).write_bytes(PREREGISTRATION.read_bytes())
            (root / asset["path"]).write_bytes(raw + b" ")
            with self.assertRaisesRegex(SchemaError, "exact byte identity"):
                load_preregistration(root / PREREGISTRATION.name)

    def test_receipt_metadata_cannot_change_frozen_raw_cube(self):
        baseline = regenerate_frozen_attempt_inputs(self.preregistration, SUPPORTED_CELL, 0)
        metadata_only = copy.deepcopy(self.preregistration)
        metadata_only["receipt_review_note"] = "does not participate in DGP generation"
        mutated = regenerate_frozen_attempt_inputs(metadata_only, SUPPORTED_CELL, 0)
        self.assertEqual(
            metadata_only["determinism"]["dgp_generator_identity_sha256"],
            self.preregistration["determinism"]["dgp_generator_identity_sha256"],
        )
        self.assertEqual(mutated["raw_input_sha256"], baseline["raw_input_sha256"])
        self.assertEqual(mutated["raw_dgp_identity_sha256"], baseline["raw_dgp_identity_sha256"])

    def test_numeric_encoding_canonicalizes_negative_zero(self):
        self.assertEqual(numeric_digest("truth-v4", [0.0, 1.0]), numeric_digest("truth-v4", [-0.0, 1.0]))
        with self.assertRaises(SchemaError):
            numeric_digest("truth-v4", [float("nan")])

    def test_python_independently_recomputes_all_numeric_claims(self):
        attempt = self._attempt(SUPPORTED_CELL, 0, 0)
        frozen = regenerate_frozen_attempt_inputs(self.preregistration, SUPPORTED_CELL, 0)
        computed = independently_recompute_metrics(
            attempt, frozen["latent_target_history"], frozen["latent_reference_history"],
            derive_dense_joint_oracle(self.preregistration, SUPPORTED_CELL, frozen),
        )
        self.assertTrue(all(value == 0.0 for value in computed["error"]))
        self.assertEqual(int(computed["covered"].sum()), len(frozen["latent_target_history"]) - 1)
        self.assertGreater(float(computed["interval_score"].sum()), 0.0)

    def test_fixed_l2_psd_reconstruction_accepts_only_derived_roundoff_bound(self):
        within = self._attempt(SUPPORTED_CELL, 0, 0)
        expected = np.asarray(within["predicted_difference_covariance"])
        bound = _fixed_l2_reconstruction_bound(expected)
        within["predicted_difference_covariance"][1][1] += 0.5 * bound
        within["predicted_covariance_sha256"] = numeric_digest(
            "predicted-difference-covariance-v4",
            [value for row in within["predicted_difference_covariance"] for value in row],
        )
        self._accumulator(SUPPORTED_CELL).add(within)

        above = self._attempt(SUPPORTED_CELL, 0, 0)
        above["predicted_difference_covariance"][1][1] += 1.01 * bound
        above["predicted_covariance_sha256"] = numeric_digest(
            "predicted-difference-covariance-v4",
            [value for row in above["predicted_difference_covariance"] for value in row],
        )
        with self.assertRaisesRegex(SchemaError, "fixed-L2 PSD reconstruction"):
            self._accumulator(SUPPORTED_CELL).add(above)

    def test_effective_looks_accepts_only_frozen_roundoff_bound(self):
        within = self._attempt(SUPPORTED_CELL, 0, 0)
        expected = within["effective_looks_fraction"]
        within["effective_looks_fraction"] += 0.5e-12 * max(1.0, abs(expected))
        self._accumulator(SUPPORTED_CELL).add(within)

        above = self._attempt(SUPPORTED_CELL, 0, 0)
        expected = above["effective_looks_fraction"]
        above["effective_looks_fraction"] += 1.01e-12 * max(1.0, abs(expected))
        with self.assertRaisesRegex(SchemaError, "effective-look realization differs"):
            self._accumulator(SUPPORTED_CELL).add(above)

    def test_fabricated_zero_and_self_attested_hashes_are_rejected(self):
        attempt = self._attempt(SUPPORTED_CELL, 0, 0)
        attempt["target_estimate_history"][-1] += 0.5
        accumulator = self._accumulator(SUPPORTED_CELL)
        with self.assertRaisesRegex(SchemaError, "digest mismatch"):
            accumulator.add(attempt)

    def test_producer_replaced_raw_input_or_latent_history_fails_regeneration(self):
        raw = self._attempt(CELL, 0, 0)
        raw["raw_input_sha256"] = "f" * 64
        with self.assertRaisesRegex(SchemaError, "raw DGP"):
            self._accumulator(CELL).add(raw)
        latent = self._attempt(CELL, 0, 0)
        latent["latent_history_sha256"] = "e" * 64
        with self.assertRaisesRegex(SchemaError, "latent history"):
            self._accumulator(CELL).add(latent)

    def test_full_production_shaped_dgp_binds_support_ancestry_and_raw_content(self):
        frozen = regenerate_frozen_attempt_inputs(self.preregistration, RISK_CELL, 0)
        dates = len(self.preregistration["generator"]["acquisition"]["topologies"]["four_blocks"]["date_axis"])
        self.assertEqual(frozen["raw_input_shape"], [frozen["raw_cube_source_count"], dates, 2])
        self.assertEqual(frozen["raw_input_value_count"], 2 * dates * frozen["raw_cube_source_count"])
        self.assertGreater(frozen["raw_cube_source_count"], frozen["union_source_count"])
        self.assertGreater(frozen["raw_input_value_count"], 10_000)
        self.assertEqual(len(frozen["latent_target_history"]), dates)
        self.assertEqual(len(frozen["latent_reference_history"]), dates)
        self.assertLessEqual(len(compact_json_line(self._attempt(RISK_CELL, 0, 0))), FROZEN_MAX_RECORD_BYTES)
        for name in (
            "raw_input_sha256", "target_raw_input_sha256", "reference_raw_input_sha256",
            "target_support_sha256", "reference_support_sha256", "sequential_ancestry_sha256",
            "raw_dgp_identity_sha256",
        ):
            self.assertEqual(len(frozen[name]), 64)

        attempt = self._attempt(CELL, 0, 0)
        attempt["sequential_ancestry_sha256"] = "f" * 64
        attempt["raw_dgp_identity_sha256"] = "e" * 64
        attempt["target_support_sha256"] = "d" * 64
        attempt["reference_support_sha256"] = "d" * 64
        with self.assertRaisesRegex(SchemaError, "raw DGP|support|ancestry"):
            self._accumulator(CELL).add(attempt)

    def test_source_factor_halo_tamper_changes_expected_raw_identity(self):
        baseline = regenerate_frozen_attempt_inputs(self.preregistration, CELL, 0)
        target = baseline["target_coordinate"]
        halo_only = (target[0] - 2, target[1] - 2)
        original = _generate_complex_source

        def tamper(*args, **kwargs):
            values = original(*args, **kwargs)
            if args[3] == halo_only:
                values[0] = complex(values[0].real + 1.0, values[0].imag)
            return values

        with mock.patch(
            "validation.score_spatial_covariance._generate_complex_source", side_effect=tamper
        ):
            changed = regenerate_frozen_attempt_inputs(self.preregistration, CELL, 0)
        self.assertEqual(changed["target_support_sha256"], baseline["target_support_sha256"])
        self.assertNotEqual(changed["raw_input_sha256"], baseline["raw_input_sha256"])
        self.assertNotEqual(changed["raw_dgp_identity_sha256"], baseline["raw_dgp_identity_sha256"])

    def test_coincident_pair_is_the_same_source_and_exact_zero_contrast(self):
        frozen = regenerate_frozen_attempt_inputs(self.preregistration, CELL, 0)
        self.assertEqual(frozen["target_support_sha256"], frozen["reference_support_sha256"])
        self.assertEqual(frozen["target_raw_input_sha256"], frozen["reference_raw_input_sha256"])
        self.assertEqual(frozen["latent_target_history"], frozen["latent_reference_history"])
        attempt = self._attempt(CELL, 0, 0)
        self.assertEqual(attempt["target_estimate_history"], attempt["reference_estimate_history"])
        self.assertTrue(all(value == 0.0 for row in attempt["predicted_difference_covariance"] for value in row))

    def test_online_reducer_uses_latent_estimator_errors_without_retaining_attempts(self):
        accumulator = self._accumulator(SUPPORTED_CELL, seeds=2)
        first = self._attempt(SUPPORTED_CELL, 0, 0)
        second = self._attempt(SUPPORTED_CELL, 0, 1)
        first["target_estimate_history"][-1] += 1.0
        second["target_estimate_history"][-1] -= 1.0
        for attempt in (first, second):
            dates = len(attempt["target_estimate_history"])
            difference = [[0.0] * dates for _ in range(dates)]
            difference[-1][-1] = 1.0
            self._set_operator_and_difference(attempt, difference)
            attempt["estimate_sha256"] = numeric_digest(
                "estimate-history-v4",
                [*attempt["target_estimate_history"], *attempt["reference_estimate_history"]],
            )
            accumulator.add(attempt)
        summary = accumulator.finalize()
        self.assertEqual(summary["error_bias_norm"], 0.0)
        self.assertEqual(summary["covariance_calibration_relative_error"], 0.0)
        self.assertGreater(summary["empirical_error_covariance_trace"], 0.0)
        self.assertIn("empirical_error_covariance_digest", summary)
        self.assertFalse(hasattr(accumulator, "attempts"))

    def test_nondifferentiable_attempt_is_retained_and_counts_against_emission_gate(self):
        accumulator = self._accumulator(SUPPORTED_CELL, seeds=2)
        accumulator.add(self._nondifferentiable_attempt(SUPPORTED_CELL, 0, 0))
        accumulator.add(self._attempt(SUPPORTED_CELL, 0, 1))
        summary = accumulator.finalize()
        self.assertEqual(summary["status_histogram"]["nondifferentiable_node"], 1)
        self.assertEqual(summary["failure_histogram"], {"nondifferentiable_node": 1})
        self.assertEqual(summary["emitted_seeds"], 1)
        self.assertEqual(summary["status"], "fail")

    def test_coverage_is_gated_per_date_and_names_final_date(self):
        accumulator = self._accumulator(SUPPORTED_CELL, seeds=2)
        for seed_index in range(2):
            attempt = self._attempt(SUPPORTED_CELL, 0, seed_index)
            difference = [
                [0.0] * len(attempt["target_estimate_history"])
                for _ in attempt["target_estimate_history"]
            ]
            self._set_operator_and_difference(attempt, difference)
            if seed_index == 0:
                attempt["target_estimate_history"][1] += 1.0
            attempt["estimate_sha256"] = numeric_digest(
                "estimate-history-v4",
                [*attempt["target_estimate_history"], *attempt["reference_estimate_history"]],
            )
            accumulator.add(attempt)
        summary = accumulator.finalize()
        self.assertEqual(summary["coverage_95_by_date"][1], 0.5)
        self.assertEqual(summary["final_date_coverage_95"], 1.0)
        self.assertEqual(summary["status"], "fail")

    def test_deterministic_signed_and_invalid_contracts_are_separate_from_coverage(self):
        for geometry, expected_sign in (
            ("shared_75_positive", "positive"),
            ("shared_75_negative", "negative"),
            ("disjoint_immediate", "none"),
            ("coincident", "zero"),
        ):
            cell_id = CELL.replace("coincident", geometry)
            attempt = self._attempt(cell_id, 0, 0)
            self.assertEqual(attempt["signed_influence_sign"], expected_sign)
            accumulator = self._accumulator(cell_id)
            accumulator.add(attempt)
            summary = accumulator.finalize()
            self.assertEqual(summary["attempted_seeds"], 1)
        masked = self._attempt(MASKED_CELL, 0, 0, masked=True)
        accumulator = self._accumulator(MASKED_CELL)
        accumulator.add(masked)
        self.assertIsNone(accumulator.finalize()["coverage_95_by_date"])

        masked_ks = MASKED_CELL.replace("rect", "ks_frozen")
        masked_attempt = self._attempt(masked_ks, 0, 0, masked=True)
        masked_accumulator = self._accumulator(masked_ks)
        masked_accumulator.add(masked_attempt)
        self.assertEqual(masked_accumulator.finalize()["status"], "pass")

    def test_tied_eigen_stress_binds_actual_singular_production_probe(self):
        tied = CELL.replace("well_separated", "tied_eigenvalue")
        frozen = regenerate_frozen_attempt_inputs(self.preregistration, tied, 0)
        self.assertEqual(frozen["raw_input_shape"], [9, 3, 2])
        self.assertEqual(
            frozen["raw_input_sha256"],
            "6cb68cc5e0957b86cbdbcd1f65ffb753c4193292e8994dbd34b881a29b497ebe",
        )
        attempt = self._attempt(tied, 0, 0)
        self.assertEqual(attempt["status"], "singular_local_information")
        self.assertEqual(attempt["estimator_branch"], "evd")
        self.assertFalse(attempt["emitted"])
        accumulator = self._accumulator(tied)
        accumulator.add(attempt)
        summary = accumulator.finalize()
        self.assertEqual(summary["status"], "not_evaluable")
        self.assertEqual(summary["status_histogram"]["singular_local_information"], 1)

    def test_negative_pair_dgp_has_negative_target_reference_error_loading(self):
        negative = SUPPORTED_CELL.replace("shared_75_positive", "shared_75_negative").replace(
            "independent_complex_looks", "spatial_correlation_stress"
        )
        positive = negative.replace("shared_75_negative", "shared_75_positive")
        negative_dgp = regenerate_frozen_attempt_inputs(self.preregistration, negative, 0)
        positive_dgp = regenerate_frozen_attempt_inputs(self.preregistration, positive, 0)
        self.assertLess(
            negative_dgp["target_global_loading_mean"]
            * negative_dgp["reference_global_loading_mean"],
            0.0,
        )
        self.assertGreater(
            positive_dgp["target_global_loading_mean"]
            * positive_dgp["reference_global_loading_mean"],
            0.0,
        )

    def test_source_process_selects_identity_or_exponential_correlation(self):
        independent = regenerate_frozen_attempt_inputs(self.preregistration, CELL, 0)
        spatial = regenerate_frozen_attempt_inputs(
            self.preregistration,
            CELL.replace("independent_complex_looks", "spatial_correlation_stress"),
            0,
        )
        self.assertEqual(independent["source_correlation_model"], "identity_v1")
        self.assertEqual(independent["effective_looks_fraction"], 1.0)
        self.assertEqual(spatial["source_correlation_model"], "exponential_euclidean_v1")
        self.assertEqual(spatial["source_correlation_distance_scale_pixels"], 1.5)
        self.assertLess(spatial["effective_looks_fraction"], 1.0)

    def test_effective_looks_accepts_libm_ulp_but_rejects_material_drift(self):
        cell_id = expected_cell_ids(self.preregistration)[0]
        accepted = self._attempt(cell_id, 0, 0)
        accepted["effective_looks_fraction"] = math.nextafter(
            accepted["effective_looks_fraction"], 0.0
        )
        self._accumulator(cell_id).add(accepted)
        rejected = self._attempt(cell_id, 0, 0)
        rejected["effective_looks_fraction"] += 1e-12
        with self.assertRaisesRegex(SchemaError, "effective-look realization"):
            self._accumulator(cell_id).add(rejected)

    def test_attempt_contract_restores_operator_error_and_production_artifact_bindings(self):
        for field in (
            "production_operator_matrix", "contrast_weights", "operator_sha256",
        ):
            self.assertIn(field, ATTEMPT_KEYS)
        self.assertNotIn("production_artifact", ATTEMPT_KEYS)
        self.assertNotIn("production_artifact_sha256", ATTEMPT_KEYS)
        self.assertNotIn("dense_oracle_matrix", ATTEMPT_KEYS)
        self.assertNotIn("oracle_sha256", ATTEMPT_KEYS)
        for threshold in (
            "deterministic_operator_relative_error_max",
            "stochastic_operator_relative_error_max",
            "contrast_variance_relative_error_max",
        ):
            self.assertIn(threshold, self.preregistration["thresholds"])

    def test_attempt_stream_does_not_retain_production_hdf5_paths(self):
        attempt = self._attempt(SUPPORTED_CELL, 0, 0)
        self.assertNotIn("production_artifact", attempt)
        self.assertNotIn("production_artifact_sha256", attempt)

    def test_rust_batch_writes_and_rereads_one_actual_production_parity_fixture(self):
        request = {
            "schema": "dolphinrust.spatial-covariance.attempt/4",
            "cell_id": CELL,
            "cell_ordinal": 0,
            "seed_index": 0,
            "seed_sha256": "0" * 64,
            **dict(zip(self.preregistration["matrix_contract"]["dimension_order"], CELL.split("|"))),
        }
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    "cargo", "run", "--quiet", "-p", "dolphin-workflows",
                    "--no-default-features", "--features", "no-gpu", "--example",
                    "spatial_covariance_batch", "--", "--parity-fixture",
                    "--artifact-directory", directory,
                ],
                input=json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n",
                check=True,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parents[2],
            )
            evidence = json.loads(completed.stdout)
            binding = {
                "schema": "dolphinrust.spatial-covariance.production-parity-fixture/4",
                "hdf5_path": evidence["hdf5_path"],
                "sidecar_path": evidence["sidecar_path"],
                "hdf5_sha256": evidence["hdf5_sha256"],
                "sidecar_sha256": evidence["sidecar_sha256"],
                "hdf5_schema_version": evidence["hdf5_schema_version"],
                "manifest_schema_version": evidence["manifest_schema_version"],
                "coupling": evidence["coupling"],
                "seed_index": evidence["seed_index"],
                "factor_digest": evidence["factor_digest"],
                "persisted_factor_digest": evidence["persisted_factor_digest"],
                "estimator_branch": evidence["estimator"],
            }
            validate_production_parity_fixture(
                self.preregistration, Path(directory), binding, sha256_json(binding)
            )
            (Path(directory) / binding["hdf5_path"]).write_bytes(b"tamper")
            with self.assertRaisesRegex(SchemaError, "production parity"):
                validate_production_parity_fixture(
                    self.preregistration, Path(directory), binding, sha256_json(binding)
                )

    def test_parity_driver_cannot_masquerade_as_full_cell_generator(self):
        completed = subprocess.run(
            [
                "cargo", "run", "--quiet", "-p", "dolphin-workflows",
                "--no-default-features", "--features", "no-gpu", "--example",
                "spatial_covariance_batch",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("full-cell mode requires --preregistration", completed.stderr)

    def test_direct_pair_contract_requires_actual_positive_independent_negative_order(self):
        identities = {
            "marginal_dgp_digest": "d" * 64,
            "target_support_digest": "a" * 64,
            "reference_support_digest": "b" * 64,
            "latent_history_digest": "c" * 64,
            "phase_orientation_digest": "e" * 64,
        }
        margins = {
            "target_predicted_covariance_trace": 1.0,
            "reference_predicted_covariance_trace": 1.0,
            "target_empirical_error_covariance_trace": 1.05,
            "reference_empirical_error_covariance_trace": 1.05,
        }
        positive = {**identities, **margins, "predicted_covariance_trace": 1.0, "empirical_error_covariance_trace": 1.1}
        negative = {**identities, **margins, "predicted_covariance_trace": 3.0, "empirical_error_covariance_trace": 3.1}
        independent = {**identities, **margins, "predicted_covariance_trace": 2.0, "empirical_error_covariance_trace": 2.1}
        validate_direct_pair_variance_order(positive, independent, negative)
        with self.assertRaisesRegex(SchemaError, "positive < independent < negative"):
            validate_direct_pair_variance_order(negative, independent, positive)

    def test_stochastic_operator_error_is_derived_from_online_empirical_joint_covariance(self):
        accumulator = self._accumulator(SUPPORTED_CELL, seeds=2)
        for seed_index, signed_error in enumerate((1.0, -1.0)):
            attempt = self._attempt(SUPPORTED_CELL, 0, seed_index)
            attempt["target_estimate_history"][-1] += signed_error
            dates = len(attempt["target_estimate_history"])
            joint = [[0.0] * (2 * dates) for _ in range(2 * dates)]
            joint[dates - 1][dates - 1] = 1.0
            attempt["production_operator_matrix"] = joint
            self._sync_difference_from_operator(attempt)
            attempt["operator_sha256"] = numeric_digest(
                "production-operator-v4", [value for row in joint for value in row]
            )
            attempt["estimate_sha256"] = numeric_digest(
                "estimate-history-v4",
                [*attempt["target_estimate_history"], *attempt["reference_estimate_history"]],
            )
            accumulator.add(attempt)
        summary = accumulator.finalize()
        self.assertEqual(summary["operator_relative_error"], 0.0)
        self.assertEqual(summary["contrast_variance_relative_error"], 0.0)

    def test_python_derives_joint_oracle_from_regenerated_dgp(self):
        frozen = regenerate_frozen_attempt_inputs(self.preregistration, SUPPORTED_CELL, 0)
        oracle = derive_dense_joint_oracle(self.preregistration, SUPPORTED_CELL, frozen)
        dates = len(frozen["latent_target_history"])
        self.assertEqual(oracle.shape, (2 * dates, 2 * dates))
        self.assertEqual(
            numeric_digest("dense-oracle-v4", oracle.flat),
            frozen["dense_oracle_sha256"],
        )

    def test_matched_signed_pair_differs_only_by_coupling(self):
        positive = regenerate_frozen_attempt_inputs(self.preregistration, SUPPORTED_CELL, 0)
        negative = regenerate_frozen_attempt_inputs(
            self.preregistration,
            SUPPORTED_CELL.replace("shared_75_positive", "shared_75_negative"),
            0,
        )
        for field in (
            "target_support_sha256", "reference_support_sha256",
            "latent_target_history", "latent_reference_history",
            "target_marginal_oracle_sha256", "reference_marginal_oracle_sha256",
        ):
            self.assertEqual(positive[field], negative[field])
        self.assertLess(
            positive["oracle_difference_covariance_trace"],
            positive["oracle_independent_covariance_trace"],
        )
        self.assertLess(
            positive["oracle_independent_covariance_trace"],
            negative["oracle_difference_covariance_trace"],
        )

    def test_cell_summary_binds_attempt_digest_and_scope(self):
        accumulator = self._accumulator(CELL)
        accumulator.add(self._attempt(CELL, 0, 0))
        summary = accumulator.finalize()
        self.assertEqual(summary["attempted_seeds"], 1)
        self.assertNotEqual(summary["request_digest"], summary["attempt_digest"])

    def test_cell_boundary_commit_deletes_transport_only_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transport = root / "attempts.jsonl"
            transport.write_bytes(b"".join(compact_json_line(self._attempt(MASKED_CELL, 0, seed, masked=True, artifact_root=root)) for seed in range(expected_seed_count(MASKED_CELL))))
            destination = root / "cell-00000.jsonl"
            commit_cell_transport(
                self.preregistration, MASKED_CELL, 0, transport, destination, CODE, BINARY,
                artifact_root=root,
            )
            self.assertFalse(transport.exists())
            validate_cell_summary(self.preregistration, json.loads(destination.read_text()), MASKED_CELL, 0, CODE, BINARY)

    def test_crash_preserves_prior_boundary_and_malformed_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / "cell-00000.jsonl"
            prior.write_text("committed\n")
            transport = root / "broken.jsonl"
            transport.write_text("{\n")
            with self.assertRaises(SchemaError):
                commit_cell_transport(self.preregistration, CELL, 1, transport, root / "cell-00001.jsonl", CODE, BINARY)
            self.assertEqual(prior.read_text(), "committed\n")
            self.assertTrue(transport.exists())
            self.assertFalse((root / "cell-00001.jsonl").exists())

    def test_compact_shard_commit_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = root / "cells"
            cells.mkdir()
            transport = root / "attempts.jsonl"
            transport.write_bytes(b"".join(compact_json_line(self._attempt(MASKED_CELL, 0, seed, masked=True, artifact_root=root)) for seed in range(expected_seed_count(MASKED_CELL))))
            commit_cell_transport(
                self.preregistration, MASKED_CELL, 0, transport, cells / "cell-00000.jsonl", CODE, BINARY,
                artifact_root=root,
            )
            spec = ShardSpec(0, 0, 1, (MASKED_CELL,), (expected_seed_count(MASKED_CELL),))
            manifest = root / "manifest.jsonl"
            commit_output_shard(self.preregistration, spec, root, cells, manifest, CODE, BINARY, 1.0, 1_000_000)
            replay = lambda cell_id, ordinal: (self._attempt(cell_id, ordinal, seed, masked=True, artifact_root=root) for seed in range(expected_seed_count(cell_id)))
            self.assertTrue(committed_shard_matches(self.preregistration, spec, root, manifest, CODE, BINARY, replay))
            cells.joinpath("cell-00000.jsonl").write_text("{}\n")
            self.assertFalse(committed_shard_matches(self.preregistration, spec, root, manifest, CODE, BINARY, replay))

    def test_resume_rejects_self_consistent_replaced_summary_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = root / "cells"
            cells.mkdir()
            transport = root / "attempts.jsonl"
            transport.write_bytes(b"".join(compact_json_line(self._attempt(MASKED_CELL, 0, seed, masked=True, artifact_root=root)) for seed in range(expected_seed_count(MASKED_CELL))))
            cell_path = cells / "cell-00000.jsonl"
            commit_cell_transport(
                self.preregistration, MASKED_CELL, 0, transport, cell_path, CODE, BINARY,
                artifact_root=root,
            )
            summary = json.loads(cell_path.read_text())
            summary["effective_looks_fraction"] = 0.5
            cell_path.write_bytes(compact_json_line(summary))
            spec = ShardSpec(0, 0, 1, (MASKED_CELL,), (expected_seed_count(MASKED_CELL),))
            manifest = root / "manifest.jsonl"
            commit_output_shard(self.preregistration, spec, root, cells, manifest, CODE, BINARY, 1.0, 1_000_000)
            replay = lambda cell_id, ordinal: (self._attempt(cell_id, ordinal, seed, masked=True, artifact_root=root) for seed in range(expected_seed_count(cell_id)))
            self.assertFalse(committed_shard_matches(self.preregistration, spec, root, manifest, CODE, BINARY, replay))
            self.assertFalse(committed_shard_matches(self.preregistration, spec, root, manifest, CODE, BINARY))

    def test_prepare_retains_one_descriptor_not_attempt_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = ShardSpec(0, 0, 1, (CELL,), (expected_seed_count(CELL),))
            destination = Path(directory) / "descriptor.jsonl"
            receipt = prepare_input_shard(self.preregistration, spec, destination)
            self.assertEqual(receipt["records"], 1)
            descriptor = json.loads(destination.read_text())
            self.assertFalse(descriptor["retained"])
            self.assertEqual(descriptor["expected_attempts"], expected_seed_count(CELL))
            self.assertEqual(descriptor["seed_counts"], [expected_seed_count(CELL)])

    def test_committing_shard_deletes_preparation_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = root / "cells"
            cells.mkdir()
            spec = ShardSpec(0, 0, 1, (MASKED_CELL,), (expected_seed_count(MASKED_CELL),))
            descriptor = root / "requests" / "shard-00000.jsonl"
            prepare_input_shard(self.preregistration, spec, descriptor)
            transport = root / "attempts.jsonl"
            transport.write_bytes(compact_json_line(self._attempt(MASKED_CELL, 0, 0, masked=True, artifact_root=root)))
            commit_cell_transport(
                self.preregistration, MASKED_CELL, 0, transport,
                cells / "cell-00000.jsonl", CODE, BINARY,
                artifact_root=root,
            )
            commit_output_shard(
                self.preregistration, spec, root, cells, root / "manifest.jsonl",
                CODE, BINARY, 1.0, 1_000_000,
            )
            self.assertFalse(descriptor.exists())

    def test_area_and_date_sweeps_are_independently_identifiable(self):
        matrix = self.preregistration["resource_matrix"]
        self.assertEqual(len({item["tile_pixels"] for item in matrix if item["dates"] == 26}), 3)
        self.assertEqual(len({item["dates"] for item in matrix if item["tile_pixels"] == 65536}), 3)
        resources = self._resource_receipts()
        self.assertEqual(_validate_resources(self.preregistration, resources, BINARY), [PASS] * 5)
        resources[0]["growth_observation"][0]["raw_measurement_sha256"] = "bad"
        with self.assertRaises(SchemaError):
            _validate_resources(self.preregistration, resources, BINARY)
        resources = self._resource_receipts()
        raw = resources[0]["growth_observation"][0]["raw_measurement"]
        raw["command"][-1] = "999"
        resources[0]["growth_observation"][0]["raw_measurement_sha256"] = sha256_json(raw)
        resources[0]["resource_hash"] = sha256_json({key: value for key, value in resources[0].items() if key != "resource_hash"})
        with self.assertRaisesRegex(SchemaError, "raw resource measurement"):
            _validate_resources(self.preregistration, resources, BINARY)

    def test_resource_receipt_rejects_unbound_component_source(self):
        resources = self._resource_receipts()
        component = resources[0]["allocation_components"][0]
        component["source"] = ""
        component["component_sha256"] = sha256_json({key: value for key, value in component.items() if key != "component_sha256"})
        resources[0]["resource_hash"] = sha256_json({key: value for key, value in resources[0].items() if key != "resource_hash"})
        with self.assertRaisesRegex(SchemaError, "allocation receipt"):
            _validate_resources(self.preregistration, resources, BINARY)

    def test_resource_receipt_binds_dependency_microbatch_and_named_formulas(self):
        for field, digest_field in (
            ("dependency_cone", "dependency_cone_sha256"),
            ("microbatch", "microbatch_sha256"),
        ):
            resources = self._resource_receipts()
            resources[0][field][next(name for name in resources[0][field] if name.endswith("count") or name.endswith("sources"))] += 1
            resources[0][digest_field] = sha256_json(resources[0][field])
            resources[0]["resource_hash"] = sha256_json({key: value for key, value in resources[0].items() if key != "resource_hash"})
            with self.assertRaisesRegex(SchemaError, "identity mismatch"):
                _validate_resources(self.preregistration, resources, BINARY)
        resources = self._resource_receipts()
        component = resources[0]["allocation_components"][0]
        component["bytes"] += 1
        component["component_sha256"] = sha256_json({key: value for key, value in component.items() if key != "component_sha256"})
        resources[0]["resource_hash"] = sha256_json({key: value for key, value in resources[0].items() if key != "resource_hash"})
        with self.assertRaisesRegex(SchemaError, "allocation scope drifted"):
            _validate_resources(self.preregistration, resources, BINARY)

    def test_allocation_receipt_is_benchmark_emitted_and_hash_bound(self):
        resources = self._resource_receipts()
        raw = resources[0]["growth_observation"][0]["raw_measurement"]
        self.assertNotIn("allocation_measurement", raw)
        allocation = json.loads(raw["stdout_json"])
        allocation["maximum_simultaneously_retained_bytes"] += 1
        raw["stdout_json"] = json.dumps(allocation, sort_keys=True, separators=(",", ":")) + "\n"
        raw["stdout_bytes"] = len(raw["stdout_json"].encode())
        raw["stdout_sha256"] = hashlib.sha256(raw["stdout_json"].encode()).hexdigest()
        resources[0]["growth_observation"][0]["raw_measurement_sha256"] = sha256_json(raw)
        resources[0]["resource_hash"] = sha256_json(
            {key: value for key, value in resources[0].items() if key != "resource_hash"}
        )
        with self.assertRaisesRegex(SchemaError, "allocation arithmetic"):
            _validate_resources(self.preregistration, resources, BINARY)

    def test_benchmark_driver_captures_stdout_and_nested_fabrication_is_rejected(self):
        allocation = json.loads(
            self._resource_receipts()[0]["growth_observation"][0]["raw_measurement"]["stdout_json"]
        )
        command = [
            sys.executable, "-c",
            "import json,sys;sys.stdout.write(json.dumps(" + repr(allocation) + ",sort_keys=True,separators=(',',':'))+'\\n')",
        ]
        captured = capture_benchmark_stdout(command)
        self.assertEqual(captured["stdout_sha256"], hashlib.sha256(captured["stdout_json"].encode()).hexdigest())
        self.assertNotIn("allocation_measurement", captured)

        resources = self._resource_receipts()
        raw = resources[0]["growth_observation"][0]["raw_measurement"]
        raw["allocation_measurement"] = allocation
        resources[0]["growth_observation"][0]["raw_measurement_sha256"] = sha256_json(raw)
        resources[0]["resource_hash"] = sha256_json(
            {key: value for key, value in resources[0].items() if key != "resource_hash"}
        )
        with self.assertRaisesRegex(SchemaError, "raw resource measurement"):
            _validate_resources(self.preregistration, resources, BINARY)

    def test_manifest_hash_and_parse_use_one_identical_bounded_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            raw = b'{"schema":"test"}\n'
            path.write_bytes(raw)
            original_open = Path.open
            opens = 0

            def count_open(open_path, *args, **kwargs):
                nonlocal opens
                if open_path == path:
                    opens += 1
                return original_open(open_path, *args, **kwargs)

            with mock.patch.object(Path, "open", count_open):
                value, accepted, digest = _read_hashed_json_record(path, len(raw), "manifest")
            self.assertEqual(opens, 1)
            self.assertEqual(value, {"schema": "test"})
            self.assertEqual(accepted, raw)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    def test_raw_resource_measurement_accepts_exact_bound_and_rejects_one_byte_over(self):
        resources = self._resource_receipts()
        sampling = self.preregistration["resource_sampling"]
        raw = resources[0]["growth_observation"][0]["raw_measurement"]
        encoded = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        raw["tool_versions"]["uname"] += "x" * (sampling["max_encoded_raw_measurement_bytes"] - len(encoded(raw)))
        observation = resources[0]["growth_observation"][0]
        observation["raw_measurement_sha256"] = sha256_json(raw)
        resources[0]["resource_hash"] = sha256_json({key: value for key, value in resources[0].items() if key != "resource_hash"})
        self.assertEqual(len(encoded(raw)), sampling["max_encoded_raw_measurement_bytes"])
        self.assertEqual(_validate_resources(self.preregistration, resources, BINARY), [PASS] * 5)
        raw["tool_versions"]["uname"] += "x"
        observation["raw_measurement_sha256"] = sha256_json(raw)
        resources[0]["resource_hash"] = sha256_json({key: value for key, value in resources[0].items() if key != "resource_hash"})
        with self.assertRaisesRegex(SchemaError, "raw resource measurement"):
            _validate_resources(self.preregistration, resources, BINARY)

    def test_untrusted_single_record_is_sized_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.jsonl"
            with path.open("wb") as handle:
                handle.truncate(129)
            with self.assertRaisesRegex(SchemaError, "before read"):
                _read_single_json_record(path, 128, "test manifest")

    def test_bounded_json_readers_accept_exact_bound_and_reject_one_byte_over(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = "raw resource"
            limit = self.preregistration["resource_sampling"]["max_encoded_raw_measurement_bytes"]
            exact = root / "raw-resource-exact.json"
            exact.write_bytes(b'{"value":1}' + b" " * (limit - len(b'{"value":1}')))
            self.assertEqual(_load_bounded_json(exact, limit, label), {"value": 1})
            over = root / "raw-resource-over.json"
            over.write_bytes(exact.read_bytes() + b" ")
            with self.assertRaisesRegex(SchemaError, "byte cap"):
                _load_bounded_json(over, limit, label)
            for label, limit in (("shard manifest", FROZEN_MAX_SHARD_MANIFEST_BYTES), ("run manifest", FROZEN_MAX_RUN_MANIFEST_BYTES)):
                exact = root / f"{label.replace(' ', '-')}-exact.json"
                exact.write_bytes(b'{"value":1}' + b" " * (limit - len(b'{"value":1}') - 1) + b"\n")
                self.assertEqual(_read_single_json_record(exact, limit, label)[0], {"value": 1})
                over = root / f"{label.replace(' ', '-')}-over.json"
                over.write_bytes(exact.read_bytes() + b"\n")
                with self.assertRaisesRegex(SchemaError, "byte cap"):
                    _read_single_json_record(over, limit, label)

    def test_bounded_json_rejects_path_replacement_between_stat_and_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            replacement = Path(directory) / "replacement.json"
            path.write_text('{"value":1}')
            replacement.write_text('{"value":2}')
            original_open = Path.open

            def replace_then_open(open_path, *args, **kwargs):
                if open_path == path:
                    os.replace(replacement, path)
                return original_open(open_path, *args, **kwargs)

            with mock.patch.object(Path, "open", replace_then_open), self.assertRaisesRegex(SchemaError, "changed before"):
                _load_bounded_json(path, 128, "receipt")

    def test_preregistration_drift_fails_closed(self):
        for section, field, value in (("thresholds", "coverage_absolute_error_max", 0.03), ("determinism", "prng", "unspecified"), ("numeric_contract", "operator_relative_error", "trust Rust"), ("execution_protocol", "retained_attempt_records", True)):
            changed = copy.deepcopy(self.preregistration)
            changed[section][field] = value
            with self.subTest(section=section), self.assertRaises(SchemaError):
                validate_preregistration(changed)

    def test_cli_exposes_compact_lifecycle(self):
        completed = subprocess.run([sys.executable, str(VALIDATION / "spatial_covariance_simulation.py"), "--help"], check=True, capture_output=True, text=True)
        for command in ("capture-resource", "prepare", "reduce-cell", "commit", "resume", "assemble"):
            self.assertIn(command, completed.stdout)

    def test_assembly_fails_closed_until_rust_replay_executable_is_available(self):
        with self.assertRaisesRegex(SchemaError, "Rust spatial_covariance_batch replay executable"):
            build_run_manifest(
                self.preregistration, Path.cwd(), (), CODE, BINARY, {}, self._resource_receipts()
            )


if __name__ == "__main__":
    unittest.main()
