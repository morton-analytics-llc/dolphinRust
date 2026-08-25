import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validation.score_spatial_covariance import (
    ATTEMPT_KEYS,
    FROZEN_ATTEMPT_COUNT,
    FROZEN_CELL_COUNT,
    FROZEN_CELL_SUMMARY_COMPONENT_BYTES,
    FROZEN_MAX_SHARD_BYTES,
    FROZEN_RETAINED_SIZE_BOUND_BYTES,
    FROZEN_SEED_COUNT,
    FROZEN_SHARD_COUNT,
    PASS,
    CellAccumulator,
    SchemaError,
    ShardSpec,
    _CellSummarySink,
    _expected_seed_hash,
    _growth_exponent,
    _read_single_json_record,
    _validate_resources,
    deterministic_normals,
    expected_cell_ids,
    independently_recompute_metrics,
    load_preregistration,
    numeric_digest,
    regenerate_frozen_attempt_inputs,
    sha256_json,
    validate_cell_summary,
    validate_preregistration,
)
from validation.spatial_covariance_simulation import (
    build_run_manifest,
    commit_cell_transport,
    commit_output_shard,
    committed_shard_matches,
    compact_json_line,
    iter_attempt_requests,
    prepare_input_shard,
)


VALIDATION = Path(__file__).parents[1]
PREREGISTRATION = VALIDATION / "spatial_covariance_preregistration.json"
CODE = "a" * 64
BINARY = "b" * 64
CELL = "hw_1x1|stride_1|rect|interior|coincident|one_block|emi|well_separated|independent_complex_looks"
MASKED_CELL = "hw_1x1|stride_1|rect|masked|coincident|one_block|emi|well_separated|independent_complex_looks"


class SpatialCovarianceValidationV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preregistration = load_preregistration(PREREGISTRATION)

    def _attempt(self, cell_id, ordinal, seed_index, masked=False):
        labels = dict(zip(self.preregistration["matrix_contract"]["dimension_order"], cell_id.split("|")))
        generator = self.preregistration["generator"]
        window = generator["coordinates"]["window_stride"][f"{labels['half_window']}|{labels['stride']}"]
        target = window["target_by_position"][labels["position"]]
        delta = window["reference_delta_by_pair_geometry"][labels["pair_geometry"]]
        frozen = regenerate_frozen_attempt_inputs(self.preregistration, cell_id, seed_index)
        attempt = {
            "schema": "dolphinrust.spatial-covariance.attempt-evidence/4",
            "cell_id": cell_id,
            "cell_ordinal": ordinal,
            "seed_index": seed_index,
            "seed_sha256": _expected_seed_hash(self.preregistration, cell_id, seed_index),
            "status": "masked_target" if masked else "valid",
            "emitted": not masked,
            "factor_emitted": not masked,
            "raw_input_values": frozen["raw_input_values"],
            "raw_input_sha256": frozen["raw_input_sha256"],
            "truth_sha256": frozen["truth_sha256"],
            "operator_hash": "3" * 64,
            "variance_hash": "4" * 64,
            "emission_hash": "5" * 64,
            "date_axis_sha256": sha256_json(generator["acquisition"]["topologies"][labels["block_topology"]]["date_axis"]),
            "generator_hash": sha256_json(generator),
            "config_hash": sha256_json(generator),
            "source_model_hash": sha256_json(generator["source_centered_empirical"]),
            "target_coordinate": target,
            "reference_coordinate": [target[0] + delta[0], target[1] + delta[1]],
            "target_support_sha256": "6" * 64,
            "reference_support_sha256": "6" * 64,
            "target_source_count": 4,
            "reference_source_count": 4,
            "intersection_source_count": 4,
            "union_source_count": 4,
            "realized_overlap_jaccard": 1.0,
            "signed_cross_influence": None if masked else 0.0,
            "signed_influence_sign": "zero",
            "effective_looks_fraction": 1.0,
            "effective_looks_application": "source_factor_divided_by_sqrt_fraction",
            "operator_matrix": None if masked else copy.deepcopy(frozen["truth_matrix"]),
            "truth_matrix": None if masked else copy.deepcopy(frozen["truth_matrix"]),
            "contrast_weights": None if masked else frozen["contrast_weights"],
            "estimate_value": None if masked else 0.0,
            "truth_value": None if masked else frozen["truth_value"],
            "operator_relative_error": None,
            "contrast_variance_reference": None,
            "contrast_variance_relative_error": None,
            "psd_min_eigenvalue": None,
            "covered_95": None,
            "interval_score": None,
            "interval_width": None,
        }
        if not masked:
            attempt.update(independently_recompute_metrics(attempt))
        self.assertEqual(set(attempt), ATTEMPT_KEYS)
        return attempt

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
            observations = []
            for repetition in range(3):
                raw_measurement = {
                    "command": ["cargo", "run", "--release", "-p", "dolphin-workflows", "--example", "spatial_covariance_bench", "--", "--tile-pixels", str(matrix[name]["tile_pixels"]), "--dates", str(matrix[name]["dates"])],
                    "exit_status": 0,
                    "wall_seconds": 1.0,
                    "max_rss_bytes": peaks[name] - 2 + repetition,
                    "rss_sampler": sampling["rss_sampler"],
                    "rss_field": sampling["rss_field"],
                    "os": sampling["os"],
                    "hardware_class": sampling["hardware_class"],
                    "ram_bytes": sampling["ram_bytes"],
                    "tool_versions": {"rustc": "rustc test", "cargo": "cargo test", "uname": "Darwin test"},
                }
                observations.append({"repetition": repetition, "tile_pixels": matrix[name]["tile_pixels"], "date_count": matrix[name]["dates"], "peak_rss_bytes": peaks[name] - 2 + repetition, "wall_seconds": 1.0, "raw_measurement": raw_measurement, "raw_measurement_sha256": sha256_json(raw_measurement)})
            item = {"resource_id": name, "status": PASS, "rss_bytes": peaks[name], "growth_class": "linear", "resource_hash": "", "config_hash": sha256_json(self.preregistration["generator"]), "binary_hash": BINARY, "os": sampling["os"], "hardware_class": sampling["hardware_class"], "ram_bytes": sampling["ram_bytes"], "rss_sampler": sampling["rss_sampler"], "rss_field": sampling["rss_field"], "warmup_runs": sampling["warmup_runs"], "measured_repetitions": sampling["measured_repetitions"], "tool_versions": sampling["tool_versions"], "growth_observation": observations, "area_growth_exponent": area, "date_growth_exponent": dates, "acceptance": sampling["acceptance"]}
            item["resource_hash"] = sha256_json({key: value for key, value in item.items() if key != "resource_hash"})
            result.append(item)
        return result

    def test_v4_supersedes_v3_outcome_free_without_changing_matrix(self):
        validate_preregistration(self.preregistration)
        self.assertEqual(self.preregistration["schema_version"], 4)
        self.assertEqual(self.preregistration["supersedes"]["schema_version"], 3)
        self.assertFalse(self.preregistration["outcomes_present"])
        self.assertEqual(len(expected_cell_ids(self.preregistration)), FROZEN_CELL_COUNT)
        self.assertEqual(FROZEN_ATTEMPT_COUNT, 445_500_000)
        self.assertEqual(FROZEN_SHARD_COUNT, 891)

    def test_retained_bound_is_derived_and_below_32_gib(self):
        execution = self.preregistration["execution_protocol"]
        derived = FROZEN_CELL_COUNT * execution["max_encoded_cell_summary_bytes"] + FROZEN_SHARD_COUNT * execution["max_encoded_shard_manifest_bytes"] + execution["max_encoded_run_manifest_bytes"]
        self.assertEqual(derived, FROZEN_RETAINED_SIZE_BOUND_BYTES)
        self.assertLess(derived, 32 << 30)
        self.assertFalse(execution["retained_attempt_records"])
        self.assertFalse(execution["request_files_retained"])

    def test_final_summary_sink_accepts_exact_full_component_and_rejects_one_byte_over(self):
        self.assertEqual(FROZEN_CELL_SUMMARY_COMPONENT_BYTES, 729_907_200)
        self.assertEqual(FROZEN_RETAINED_SIZE_BOUND_BYTES, 761_282_560)
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
        spec = ShardSpec(0, 0, 1, (CELL,))
        generator = iter_attempt_requests(self.preregistration, spec)
        first = [next(generator), next(generator)]
        repeated_generator = iter_attempt_requests(self.preregistration, spec)
        repeated = [next(repeated_generator), next(repeated_generator)]
        self.assertEqual(first, repeated)
        self.assertEqual([item["seed_index"] for item in first], [0, 1])
        self.assertEqual(first[0]["schema"], "dolphinrust.spatial-covariance.attempt/4")

    def test_sha256_ctr_box_muller_has_frozen_golden_values(self):
        values = deterministic_normals("00" * 32, 4)
        self.assertEqual([value.hex() for value in values], ["-0x1.24d403dd03f46p+0", "0x1.7c127730e66f0p+0", "0x1.d80472ced3428p-2", "0x1.ab0b83421b7ecp-2"])
        self.assertNotEqual(values, deterministic_normals("01" * 32, 4))

    def test_numeric_encoding_canonicalizes_negative_zero(self):
        self.assertEqual(numeric_digest("truth-v4", [0.0, 1.0]), numeric_digest("truth-v4", [-0.0, 1.0]))
        with self.assertRaises(SchemaError):
            numeric_digest("truth-v4", [float("nan")])

    def test_python_independently_recomputes_all_numeric_claims(self):
        computed = independently_recompute_metrics(self._attempt(CELL, 0, 0))
        self.assertEqual(computed["operator_relative_error"], 0.0)
        self.assertGreater(computed["contrast_variance_reference"], 0.0)
        self.assertTrue(computed["covered_95"])
        self.assertGreater(computed["interval_score"], 0.0)

    def test_fabricated_zero_and_self_attested_hashes_are_rejected(self):
        attempt = self._attempt(CELL, 0, 0)
        attempt["operator_matrix"][0][0] += 0.5
        attempt["operator_relative_error"] = 0.0
        accumulator = CellAccumulator(self.preregistration, CELL, 0, 1, CODE, BINARY)
        with self.assertRaisesRegex(SchemaError, "fabricated|digest mismatch"):
            accumulator.add(attempt)

    def test_producer_replaced_raw_input_or_truth_fails_regeneration(self):
        raw = self._attempt(CELL, 0, 0)
        raw["raw_input_values"][0] += 1.0
        raw["raw_input_sha256"] = numeric_digest("raw-input-v4", raw["raw_input_values"])
        with self.assertRaisesRegex(SchemaError, "raw DGP"):
            CellAccumulator(self.preregistration, CELL, 0, 1, CODE, BINARY).add(raw)
        truth = self._attempt(CELL, 0, 0)
        truth["truth_matrix"][0][0] += 0.5
        truth["operator_matrix"] = copy.deepcopy(truth["truth_matrix"])
        truth.update(independently_recompute_metrics(truth))
        with self.assertRaisesRegex(SchemaError, "frozen truth"):
            CellAccumulator(self.preregistration, CELL, 0, 1, CODE, BINARY).add(truth)

    def test_cell_summary_binds_attempt_digest_and_scope(self):
        accumulator = CellAccumulator(self.preregistration, CELL, 0, 1, CODE, BINARY)
        accumulator.add(self._attempt(CELL, 0, 0))
        summary = accumulator.finalize()
        self.assertEqual(summary["attempted_seeds"], 1)
        self.assertNotEqual(summary["request_digest"], summary["attempt_digest"])

    def test_cell_boundary_commit_deletes_transport_only_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transport = root / "attempts.jsonl"
            transport.write_bytes(b"".join(compact_json_line(self._attempt(MASKED_CELL, 0, seed, masked=True)) for seed in range(FROZEN_SEED_COUNT)))
            destination = root / "cell-00000.jsonl"
            commit_cell_transport(self.preregistration, MASKED_CELL, 0, transport, destination, CODE, BINARY)
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
            transport.write_bytes(b"".join(compact_json_line(self._attempt(MASKED_CELL, 0, seed, masked=True)) for seed in range(FROZEN_SEED_COUNT)))
            commit_cell_transport(self.preregistration, MASKED_CELL, 0, transport, cells / "cell-00000.jsonl", CODE, BINARY)
            spec = ShardSpec(0, 0, 1, (MASKED_CELL,))
            manifest = root / "manifest.jsonl"
            commit_output_shard(self.preregistration, spec, root, cells, manifest, CODE, BINARY, 1.0, 1_000_000)
            replay = lambda cell_id, ordinal: (self._attempt(cell_id, ordinal, seed, masked=True) for seed in range(FROZEN_SEED_COUNT))
            self.assertTrue(committed_shard_matches(self.preregistration, spec, root, manifest, CODE, BINARY, replay))
            cells.joinpath("cell-00000.jsonl").write_text("{}\n")
            self.assertFalse(committed_shard_matches(self.preregistration, spec, root, manifest, CODE, BINARY, replay))

    def test_resume_rejects_self_consistent_replaced_summary_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = root / "cells"
            cells.mkdir()
            transport = root / "attempts.jsonl"
            transport.write_bytes(b"".join(compact_json_line(self._attempt(MASKED_CELL, 0, seed, masked=True)) for seed in range(FROZEN_SEED_COUNT)))
            cell_path = cells / "cell-00000.jsonl"
            commit_cell_transport(self.preregistration, MASKED_CELL, 0, transport, cell_path, CODE, BINARY)
            summary = json.loads(cell_path.read_text())
            summary["effective_looks_fraction"] = 0.5
            cell_path.write_bytes(compact_json_line(summary))
            spec = ShardSpec(0, 0, 1, (MASKED_CELL,))
            manifest = root / "manifest.jsonl"
            commit_output_shard(self.preregistration, spec, root, cells, manifest, CODE, BINARY, 1.0, 1_000_000)
            replay = lambda cell_id, ordinal: (self._attempt(cell_id, ordinal, seed, masked=True) for seed in range(FROZEN_SEED_COUNT))
            self.assertFalse(committed_shard_matches(self.preregistration, spec, root, manifest, CODE, BINARY, replay))
            self.assertFalse(committed_shard_matches(self.preregistration, spec, root, manifest, CODE, BINARY))

    def test_prepare_retains_one_descriptor_not_attempt_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = ShardSpec(0, 0, 1, (CELL,))
            destination = Path(directory) / "descriptor.jsonl"
            receipt = prepare_input_shard(self.preregistration, spec, destination)
            self.assertEqual(receipt["records"], 1)
            descriptor = json.loads(destination.read_text())
            self.assertFalse(descriptor["retained"])
            self.assertEqual(descriptor["expected_attempts"], FROZEN_SEED_COUNT)

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

    def test_untrusted_single_record_is_sized_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.jsonl"
            with path.open("wb") as handle:
                handle.truncate(129)
            with self.assertRaisesRegex(SchemaError, "before read"):
                _read_single_json_record(path, 128, "test manifest")

    def test_preregistration_drift_fails_closed(self):
        for section, field, value in (("thresholds", "coverage_absolute_error_max", 0.03), ("determinism", "prng", "unspecified"), ("numeric_contract", "operator_relative_error", "trust Rust"), ("execution_protocol", "retained_attempt_records", True)):
            changed = copy.deepcopy(self.preregistration)
            changed[section][field] = value
            with self.subTest(section=section), self.assertRaises(SchemaError):
                validate_preregistration(changed)

    def test_cli_exposes_compact_lifecycle(self):
        completed = subprocess.run([sys.executable, str(VALIDATION / "spatial_covariance_simulation.py"), "--help"], check=True, capture_output=True, text=True)
        for command in ("prepare", "reduce-cell", "commit", "resume", "assemble"):
            self.assertIn(command, completed.stdout)

    def test_assembly_fails_closed_until_rust_replay_executable_is_available(self):
        with self.assertRaisesRegex(SchemaError, "Rust spatial_covariance_batch replay executable"):
            build_run_manifest(
                self.preregistration, Path.cwd(), (), CODE, BINARY, {}, self._resource_receipts()
            )


if __name__ == "__main__":
    unittest.main()
