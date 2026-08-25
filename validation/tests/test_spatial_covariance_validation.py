import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path

from validation.score_spatial_covariance import (
    FAIL,
    NOT_EVALUABLE,
    PAIR_SIGN,
    PASS,
    CellAccumulator,
    FROZEN_ATTEMPT_COUNT,
    FROZEN_CELL_COUNT,
    FROZEN_MAX_RECORD_BYTES,
    FROZEN_SEED_COUNT,
    FROZEN_SHARD_COUNT,
    SchemaError,
    ShardSpec,
    _expected_seed_hash,
    _validate_performance_probe,
    _validate_resources,
    expected_cell_ids,
    iter_shard_specs,
    load_preregistration,
    preregistration_digest,
    result_root_sha256,
    score_attempt_shard,
    score_receipt,
    sha256_json,
    validate_preregistration,
    validate_shard_manifest,
)
from validation.spatial_covariance_simulation import (
    committed_shard_matches,
    commit_output_shard,
    build_run_manifest,
    compact_json_line,
    derive_concurrency_receipt,
    inspect_one_input_one_output,
    iter_attempt_requests,
    write_jsonl_atomic,
)


VALIDATION = Path(__file__).parents[1]
PREREGISTRATION = VALIDATION / "spatial_covariance_preregistration.json"
PR61_FIXTURE = VALIDATION / "fixtures" / "spatial_covariance_validation" / "pr61_bookkeeping_receipt.json"


class SpatialCovarianceValidationV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preregistration = load_preregistration(PREREGISTRATION)

    def test_v3_freezes_full_matrix_attempt_count_and_outcome_free_state(self):
        validate_preregistration(self.preregistration)
        self.assertEqual(self.preregistration["schema_version"], 3)
        self.assertFalse(self.preregistration["outcomes_present"])
        self.assertEqual(self.preregistration["supersedes"]["schema_version"], 2)
        self.assertFalse(self.preregistration["supersedes"]["outcomes_present"])
        self.assertEqual(self.preregistration["matrix_contract"]["expected_cell_count"], FROZEN_CELL_COUNT)
        self.assertEqual(self.preregistration["matrix_contract"]["expected_attempt_count"], FROZEN_ATTEMPT_COUNT)
        self.assertEqual(len(expected_cell_ids(self.preregistration)), 89100)
        self.assertTrue(self.preregistration["matrix_contract"]["source_process_axis_required"])
        execution = self.preregistration["execution_protocol"]
        self.assertLessEqual(execution["max_cells_per_shard"], 120)
        self.assertEqual(execution["shard_count"], 891)
        self.assertLessEqual(execution["max_cells_per_shard"] * FROZEN_SEED_COUNT * execution["max_encoded_output_record_bytes"], execution["max_uncompressed_output_bytes"])

    def test_driver_exposes_prepare_commit_resume_and_assemble_entrypoints(self):
        completed = subprocess.run(
            [sys.executable, str(VALIDATION / "spatial_covariance_simulation.py"), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        for command in ("prepare", "commit", "resume", "assemble"):
            self.assertIn(command, completed.stdout)

    def test_44550_count_from_omitting_source_process_is_rejected(self):
        product_without_source_process = 3 * 3 * 3 * 5 * 11 * 5 * 2 * 3
        self.assertEqual(product_without_source_process, 44550)
        changed = copy.deepcopy(self.preregistration)
        changed["matrix_contract"]["expected_cell_count"] = product_without_source_process
        changed["matrix_contract"]["expected_attempt_count"] = product_without_source_process * FROZEN_SEED_COUNT
        changed["matrix_contract"]["source_process_axis_required"] = False
        with self.assertRaisesRegex(SchemaError, "89100"):
            validate_preregistration(changed)

    def test_scientific_axes_seeds_thresholds_and_generator_are_immutable(self):
        mutations = (
            ("thresholds", "coverage_absolute_error_max", 0.03),
            ("seed_schedule", "attempted_seeds_per_cell", 4999),
            ("generator", "source_centered_empirical", {"model": "drift"}),
            ("execution_protocol", "max_cells_per_shard", 121),
            ("cell_reducers", "coverage_95", "wrong denominator"),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(self.preregistration)
            changed[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(SchemaError):
                validate_preregistration(changed)

    def test_all_shards_are_contiguous_complete_and_at_most_120_cells(self):
        expected_start = 0
        count = 0
        last = None
        for spec in iter_shard_specs(self.preregistration):
            self.assertEqual(spec.index, count)
            self.assertEqual(spec.cell_ordinal_start, expected_start)
            self.assertEqual(spec.cell_ordinal_end_exclusive - spec.cell_ordinal_start, len(spec.cell_ids))
            self.assertLessEqual(len(spec.cell_ids), 120)
            expected_start = spec.cell_ordinal_end_exclusive
            count += 1
            last = spec
        self.assertEqual(count, FROZEN_SHARD_COUNT)
        self.assertEqual(expected_start, FROZEN_CELL_COUNT)
        self.assertEqual(len(last.cell_ids), 100)

    def test_request_stream_is_exact_ordered_seed_schedule_without_top_up(self):
        cell_id = next(iter(expected_cell_ids(self.preregistration)))
        spec = ShardSpec(0, 0, 1, (cell_id,))
        count = 0
        first = None
        last = None
        for request in iter_attempt_requests(self.preregistration, spec):
            first = request if first is None else first
            last = request
            self.assertEqual(request["seed_index"], count)
            count += 1
        self.assertEqual(count, 5000)
        self.assertEqual(first["seed_index"], 0)
        self.assertEqual(last["seed_index"], 4999)
        self.assertEqual(last["seed_sha256"], _expected_seed_hash(self.preregistration, cell_id, 4999))

    def test_byte_cap_fails_without_final_or_partial_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "attempts.jsonl"
            with self.assertRaisesRegex(SchemaError, "byte cap"):
                write_jsonl_atomic(({"value": "x" * 64},), destination, byte_limit=16)
            self.assertFalse(destination.exists())
            self.assertFalse(Path(str(destination) + ".partial").exists())

    def test_one_input_one_output_rejects_missing_out_of_order_and_malformed(self):
        cell_id = "hw_1x1|stride_1|rect|masked|coincident|one_block|emi|well_separated|independent_complex_looks"
        spec = ShardSpec(0, 0, 1, (cell_id,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            write_jsonl_atomic(iter_attempt_requests(self.preregistration, spec), input_path)
            missing = root / "missing.jsonl.partial"
            missing.write_bytes(b"".join(compact_json_line(self._attempt(cell_id, 0, seed, status="masked_target")) for seed in range(4999)))
            with self.assertRaisesRegex(SchemaError, "cardinality"):
                inspect_one_input_one_output(self.preregistration, spec, input_path, missing)
            order = root / "order.jsonl.partial"
            order.write_bytes(b"".join(compact_json_line(self._attempt(cell_id, 0, seed, status="masked_target")) for seed in (1, 0)))
            with self.assertRaisesRegex(SchemaError, "order/identity"):
                inspect_one_input_one_output(self.preregistration, spec, input_path, order)
            malformed = root / "malformed.jsonl.partial"
            malformed.write_bytes(b"{\n")
            with self.assertRaisesRegex(SchemaError, "malformed"):
                inspect_one_input_one_output(self.preregistration, spec, input_path, malformed)

    def _attempt(self, cell_id, ordinal, seed_index, status="valid", covered=True):
        labels = dict(zip(self.preregistration["matrix_contract"]["dimension_order"], cell_id.split("|")))
        generator = self.preregistration["generator"]
        window = generator["coordinates"]["window_stride"][f"{labels['half_window']}|{labels['stride']}"]
        target = window["target_by_position"][labels["position"]]
        delta = window["reference_delta_by_pair_geometry"][labels["pair_geometry"]]
        reference = [target[0] + delta[0], target[1] + delta[1]]
        masked = status == "masked_target"
        return {
            "schema": "dolphinrust.spatial-covariance.attempt-receipt/3",
            "cell_id": cell_id,
            "cell_ordinal": ordinal,
            "seed_index": seed_index,
            "seed_sha256": _expected_seed_hash(self.preregistration, cell_id, seed_index),
            "status": status,
            "emitted": not masked,
            "factor_emitted": not masked,
            "raw_input_sha256": "1" * 64,
            "truth_sha256": "2" * 64,
            "operator_hash": "3" * 64,
            "variance_hash": "4" * 64,
            "emission_hash": "5" * 64,
            "date_axis_sha256": sha256_json(generator["acquisition"]["topologies"][labels["block_topology"]]["date_axis"]),
            "generator_hash": sha256_json(generator),
            "config_hash": sha256_json(generator),
            "source_model_hash": sha256_json(generator["source_centered_empirical"]),
            "target_coordinate": target,
            "reference_coordinate": reference,
            "target_support_sha256": "6" * 64,
            "reference_support_sha256": "6" * 64,
            "target_source_count": 4,
            "reference_source_count": 4,
            "intersection_source_count": 4,
            "union_source_count": 4,
            "realized_overlap_jaccard": 1.0,
            "signed_cross_influence": None if masked else 0.0,
            "signed_influence_sign": PAIR_SIGN[labels["pair_geometry"]],
            "effective_looks_fraction": 1.0,
            "effective_looks_application": "source_factor_divided_by_sqrt_fraction",
            "operator_relative_error": None if masked else 0.0,
            "contrast_variance_reference": None if masked else 1.0,
            "contrast_variance_relative_error": None if masked else 0.0,
            "psd_min_eigenvalue": None if masked else 0.0,
            "covered_95": None if masked else covered,
            "interval_score": None if masked else 1.0,
            "interval_width": None if masked else 1.0,
        }

    def test_cell_reducers_use_frozen_max_min_and_denominators(self):
        cell_id = "hw_1x1|stride_1|rect|interior|coincident|one_block|emi|well_separated|independent_complex_looks"
        accumulator = CellAccumulator(self.preregistration, cell_id, 0, expected_seed_count=20)
        for seed_index in range(20):
            attempt = self._attempt(cell_id, 0, seed_index, covered=seed_index < 19)
            attempt["operator_relative_error"] = seed_index * 1e-13
            attempt["contrast_variance_relative_error"] = seed_index * 1e-3
            attempt["psd_min_eigenvalue"] = seed_index * -1e-12
            accumulator.add(attempt)
        summary = accumulator.finalize()
        self.assertEqual(summary["status"], PASS)
        self.assertEqual(summary["coverage_95"], 0.95)
        self.assertEqual(summary["emission_rate"], 1.0)
        self.assertEqual(summary["operator_relative_error"], 19e-13)
        self.assertEqual(summary["contrast_variance_relative_error"], 0.019)
        self.assertEqual(summary["psd_min_eigenvalue"], -19e-12)

    def test_masked_and_tied_cells_are_complete_not_pruned(self):
        masked_id = "hw_1x1|stride_1|rect|masked|coincident|one_block|emi|well_separated|independent_complex_looks"
        masked = CellAccumulator(self.preregistration, masked_id, 1, expected_seed_count=3)
        for seed_index in range(3):
            masked.add(self._attempt(masked_id, 1, seed_index, status="masked_target"))
        self.assertEqual(masked.finalize()["status"], PASS)
        tied_id = "hw_1x1|stride_1|rect|interior|coincident|one_block|emi|tied_eigenvalue|independent_complex_looks"
        tied = CellAccumulator(self.preregistration, tied_id, 2, expected_seed_count=3)
        for seed_index in range(3):
            tied.add(self._attempt(tied_id, 2, seed_index, status="tied_eigenvalue"))
        self.assertEqual(tied.finalize()["status"], NOT_EVALUABLE)

    def test_masked_shared_geometry_requires_null_influence_without_signed_gate(self):
        for geometry in ("shared_75_positive", "shared_50_negative", "shared_25_positive"):
            cell_id = f"hw_1x1|stride_1|rect|masked|{geometry}|one_block|emi|well_separated|independent_complex_looks"
            accumulator = CellAccumulator(self.preregistration, cell_id, 0, expected_seed_count=1)
            attempt = self._attempt(cell_id, 0, 0, status="masked_target")
            accumulator.add(attempt)
            self.assertEqual(accumulator.finalize()["status"], PASS)
            invalid = dict(attempt)
            invalid["signed_cross_influence"] = 1.0
            with self.subTest(geometry=geometry), self.assertRaisesRegex(SchemaError, "null numeric"):
                CellAccumulator(self.preregistration, cell_id, 0, expected_seed_count=1).add(invalid)

    def test_weak_zero_variance_is_the_only_null_relative_error_path(self):
        cell_id = "hw_1x1|stride_1|rect|interior|coincident|one_block|emi|well_separated|independent_complex_looks"
        weak = self._attempt(cell_id, 0, 0)
        weak["contrast_variance_reference"] = 0.0
        weak["contrast_variance_relative_error"] = None
        accumulator = CellAccumulator(self.preregistration, cell_id, 0, expected_seed_count=1)
        accumulator.add(weak)
        self.assertFalse(accumulator.finalize()["variance_evaluable"])
        invalid = dict(weak)
        invalid["contrast_variance_relative_error"] = 0.0
        with self.assertRaisesRegex(SchemaError, "weak-zero"):
            CellAccumulator(self.preregistration, cell_id, 0, expected_seed_count=1).add(invalid)

    def test_accumulator_rejects_duplicate_missing_out_of_order_top_up_and_tamper(self):
        cell_id = "hw_1x1|stride_1|rect|interior|coincident|one_block|emi|well_separated|independent_complex_looks"
        accumulator = CellAccumulator(self.preregistration, cell_id, 0, expected_seed_count=2)
        with self.assertRaisesRegex(SchemaError, "out-of-order seed"):
            accumulator.add(self._attempt(cell_id, 0, 1))
        accumulator.add(self._attempt(cell_id, 0, 0))
        with self.assertRaisesRegex(SchemaError, "out-of-order seed"):
            accumulator.add(self._attempt(cell_id, 0, 0))
        with self.assertRaisesRegex(SchemaError, "missing"):
            accumulator.finalize()
        tampered = self._attempt(cell_id, 0, 1)
        tampered["truth_sha256"] = "bad"
        with self.assertRaisesRegex(SchemaError, "identity hash"):
            accumulator.add(tampered)
        unknown = self._attempt(cell_id, 0, 1)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "unknown"):
            accumulator.add(unknown)

    def test_streaming_scorer_has_bounded_memory_and_rejects_hash_tamper(self):
        cell_id = "hw_1x1|stride_1|rect|masked|coincident|one_block|emi|well_separated|independent_complex_looks"
        spec = ShardSpec(0, 0, 1, (cell_id,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_receipt = write_jsonl_atomic(iter_attempt_requests(self.preregistration, spec), input_path)
            output_path = root / "output.jsonl"
            output = write_jsonl_atomic((self._attempt(cell_id, 0, seed, status="masked_target") for seed in range(5000)), output_path)
            self.assertLessEqual(max(len(line) for line in output_path.read_bytes().splitlines(keepends=True)), FROZEN_MAX_RECORD_BYTES)
            manifest = self._manifest(spec, output, input_receipt)
            tracemalloc.start()
            summaries = score_attempt_shard(self.preregistration, root, manifest, spec)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.assertEqual(summaries[0]["status"], PASS)
            self.assertLess(peak, 5 * 1024 * 1024)
            tampered = dict(manifest)
            tampered["output_sha256"] = "f" * 64
            with self.assertRaisesRegex(SchemaError, "hash/byte"):
                score_attempt_shard(self.preregistration, root, tampered, spec)
            with input_path.open("ab") as handle:
                handle.write(b"{}\n")
            with self.assertRaisesRegex(SchemaError, "top-up"):
                score_attempt_shard(self.preregistration, root, manifest, spec)

    def _manifest(self, spec, output, input_receipt=None):
        input_receipt = input_receipt or {"sha256": "a" * 64, "bytes": 1}
        return {
            "schema": "dolphinrust.spatial-covariance.shard-manifest",
            "schema_version": 3,
            "shard_index": spec.index,
            "cell_ordinal_start": spec.cell_ordinal_start,
            "cell_ordinal_end_exclusive": spec.cell_ordinal_end_exclusive,
            "expected_cells": len(spec.cell_ids),
            "expected_attempts": spec.expected_attempts,
            "input_path": "input.jsonl",
            "output_path": "output.jsonl",
            "input_sha256": input_receipt["sha256"],
            "output_sha256": output["sha256"],
            "input_bytes": input_receipt["bytes"],
            "output_bytes": output["bytes"],
            "input_records": spec.expected_attempts,
            "output_records": spec.expected_attempts,
            "preregistration_sha256": preregistration_digest(self.preregistration),
            "code_sha256": "b" * 64,
            "binary_sha256": "c" * 64,
            "generator_protocol_sha256": sha256_json(self.preregistration["execution_protocol"]),
            "elapsed_seconds": 1.0,
            "peak_rss_bytes": 1024,
            "committed": True,
        }

    def test_partial_resume_and_manifest_scope_fail_closed(self):
        cell_id = next(iter(expected_cell_ids(self.preregistration)))
        spec = ShardSpec(0, 0, 1, (cell_id,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = {"sha256": hashlib.sha256(b"").hexdigest(), "bytes": 0}
            manifest = self._manifest(spec, output)
            path = root / "manifest.jsonl"
            path.write_bytes(compact_json_line(manifest))
            self.assertFalse(committed_shard_matches(self.preregistration, spec, root, path, "b" * 64, "c" * 64))
            changed = dict(manifest)
            changed["cell_ordinal_end_exclusive"] = 2
            with self.assertRaisesRegex(SchemaError, "scope/order/count"):
                validate_shard_manifest(self.preregistration, changed, spec)

    def test_atomic_commit_and_resume_require_exact_immutable_shard(self):
        cell_id = "hw_1x1|stride_1|rect|masked|coincident|one_block|emi|well_separated|independent_complex_looks"
        spec = ShardSpec(0, 0, 1, (cell_id,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            write_jsonl_atomic(iter_attempt_requests(self.preregistration, spec), input_path)
            output_partial = root / "output.jsonl.partial"
            shutil.copyfile(input_path, output_partial)
            manifest_path = root / "manifest.jsonl"
            with self.assertRaisesRegex(SchemaError, "malformed or unknown"):
                commit_output_shard(
                    self.preregistration, spec, root, input_path, output_partial, manifest_path,
                    "a" * 64, "b" * 64, 1.0, 1024,
                )
            output_partial.unlink()
            write_jsonl_atomic(
                (self._attempt(cell_id, 0, seed, status="masked_target") for seed in range(FROZEN_SEED_COUNT)),
                root / "valid-output.jsonl",
            )
            (root / "valid-output.jsonl").rename(output_partial)
            commit_output_shard(
                self.preregistration,
                spec,
                root,
                input_path,
                output_partial,
                manifest_path,
                "a" * 64,
                "b" * 64,
                1.0,
                1024,
            )
            self.assertFalse(output_partial.exists())
            self.assertTrue(committed_shard_matches(self.preregistration, spec, root, manifest_path, "a" * 64, "b" * 64))
            self.assertFalse(committed_shard_matches(self.preregistration, spec, root, manifest_path, "d" * 64, "b" * 64))
            with (root / "output.jsonl").open("ab") as handle:
                handle.write(b"{}\n")
            self.assertFalse(committed_shard_matches(self.preregistration, spec, root, manifest_path, "a" * 64, "b" * 64))

    def test_partial_suffix_and_symlink_escape_fail_closed(self):
        cell_id = "hw_1x1|stride_1|rect|masked|coincident|one_block|emi|well_separated|independent_complex_looks"
        spec = ShardSpec(0, 0, 1, (cell_id,))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "run"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            input_path = root / "input.jsonl"
            write_jsonl_atomic(iter_attempt_requests(self.preregistration, spec), input_path)
            double_partial = root / "output.jsonl.partial.partial"
            double_partial.write_bytes(b"")
            with self.assertRaisesRegex(SchemaError, "exactly one"):
                inspect_one_input_one_output(self.preregistration, spec, input_path, double_partial)
            with self.assertRaisesRegex(SchemaError, "must not"):
                inspect_one_input_one_output(self.preregistration, spec, input_path, root / "final.partial", require_partial=False)
            partial_manifest = root / "manifest.jsonl.partial"
            partial_manifest.write_bytes(b"{}\n")
            self.assertFalse(committed_shard_matches(
                self.preregistration, spec, root, partial_manifest, "a" * 64, "b" * 64,
            ))
            (outside / "input.jsonl").write_bytes(b"")
            (outside / "output.jsonl").write_bytes(b"")
            (root / "escape").symlink_to(outside, target_is_directory=True)
            output = {"sha256": hashlib.sha256(b"").hexdigest(), "bytes": 0}
            manifest = self._manifest(spec, output)
            manifest["input_path"] = "escape/input.jsonl"
            manifest["output_path"] = "escape/output.jsonl"
            with self.assertRaisesRegex(SchemaError, "symlink"):
                score_attempt_shard(self.preregistration, root, manifest, spec)

    def test_result_root_binds_manifest_order_and_content(self):
        first = "1" * 64
        second = "2" * 64
        root = result_root_sha256((first, second))
        self.assertNotEqual(root, result_root_sha256((second, first)))
        self.assertNotEqual(root, result_root_sha256((first, "3" * 64)))
        with self.assertRaisesRegex(SchemaError, "exactly 891"):
            build_run_manifest(self.preregistration, Path("."), (), "a" * 64, "b" * 64, {}, [])

    def test_performance_concurrency_is_derived_without_frozen_total_time(self):
        probe = self.preregistration["execution_protocol"]["performance_probe"]
        self.assertIsNone(probe["total_wall_seconds_max"])
        self.assertTrue(probe["required_before_outcomes"])
        self.assertTrue(probe["derived_concurrency_receipt_required"])
        self.assertEqual(derive_concurrency_receipt(7200, 3600, 0.25), 3)
        rate = 100000.0
        projected = FROZEN_ATTEMPT_COUNT / rate
        measurements = [
            {
                "cell_class": cell_class,
                "seed_count": seed_count,
                "attempt_count": seed_count,
                "elapsed_seconds": seed_count / rate,
                "peak_rss_bytes": 1024,
                "outcomes_persisted": False,
            }
            for cell_class in probe["required_cell_classes"]
            for seed_count in probe["seed_counts"]
        ]
        receipt = {
            "schema": "dolphinrust.spatial-covariance.performance-probe",
            "schema_version": 1,
            "outcomes_persisted": False,
            "seed_counts": probe["seed_counts"],
            "cell_classes": probe["required_cell_classes"],
            "measurements": measurements,
            "attempts_per_second": rate,
            "peak_rss_bytes": 1024,
            "target_wall_seconds": 3600.0,
            "reserve_fraction": 0.25,
            "projected_serial_seconds": projected,
            "derived_concurrency": derive_concurrency_receipt(projected, 3600.0),
            "code_sha256": "a" * 64,
            "binary_sha256": "b" * 64,
            "config_sha256": sha256_json(self.preregistration["generator"]),
        }
        _validate_performance_probe(self.preregistration, receipt, "a" * 64, "b" * 64)
        receipt["derived_concurrency"] += 1
        with self.assertRaisesRegex(SchemaError, "derived concurrency"):
            _validate_performance_probe(self.preregistration, receipt, "a" * 64, "b" * 64)
        receipt["derived_concurrency"] -= 1
        receipt["measurements"][0]["peak_rss_bytes"] = True
        with self.assertRaisesRegex(SchemaError, "invalid RSS"):
            _validate_performance_probe(self.preregistration, receipt, "a" * 64, "b" * 64)

    def _resources(self, peaks=(1_000_000, 2_000_000, 4_000_000), growth_class="linear", status=PASS):
        sampling = self.preregistration["resource_sampling"]
        resources = []
        for matrix, peak in zip(self.preregistration["resource_matrix"], peaks):
            item = {
                "resource_id": matrix["id"],
                "status": status,
                "rss_bytes": peak,
                "growth_class": growth_class,
                "resource_hash": "",
                "config_hash": sha256_json(self.preregistration["generator"]),
                "binary_hash": "b" * 64,
                "growth_observation": [
                    {
                        "repetition": repetition,
                        "tile_pixels": matrix["tile_pixels"],
                        "date_count": matrix["dates"],
                        "peak_rss_bytes": peak,
                        "wall_seconds": float(repetition + 1),
                    }
                    for repetition in range(sampling["measured_repetitions"])
                ],
                **{key: sampling[key] for key in (
                    "os", "hardware_class", "ram_bytes", "rss_sampler", "rss_field", "sampling_interval_ms",
                    "warmup_runs", "measured_repetitions", "tool_versions", "growth_regression", "acceptance",
                )},
            }
            item["resource_hash"] = sha256_json({key: value for key, value in item.items() if key != "resource_hash"})
            resources.append(item)
        return resources

    def test_resource_receipts_derive_rss_and_growth_from_13_26_52_observations(self):
        resources = self._resources()
        self.assertEqual(_validate_resources(self.preregistration, resources, "b" * 64), [PASS, PASS, PASS])
        invalid_rss = copy.deepcopy(resources)
        invalid_rss[0]["rss_bytes"] = True
        with self.assertRaisesRegex(SchemaError, "invalid status/RSS"):
            _validate_resources(self.preregistration, invalid_rss, "b" * 64)
        wrong_scope = copy.deepcopy(resources)
        wrong_scope[1]["growth_observation"][0]["date_count"] = 13
        with self.assertRaisesRegex(SchemaError, "scope drifted"):
            _validate_resources(self.preregistration, wrong_scope, "b" * 64)
        superlinear_declared_linear = self._resources((1_000_000, 4_000_000, 16_000_000))
        with self.assertRaisesRegex(SchemaError, "contradicts measured evidence"):
            _validate_resources(self.preregistration, superlinear_declared_linear, "b" * 64)

    def test_legacy_aggregate_receipt_is_rejected(self):
        with PR61_FIXTURE.open(encoding="utf-8") as handle:
            receipt = json.load(handle)
        report = score_receipt(self.preregistration, receipt)
        self.assertEqual(report["status"], FAIL)
        self.assertIn("aggregate receipts are rejected", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
