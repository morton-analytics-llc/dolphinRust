from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from validation import temporal_covariance_simulation as simulation


ROOT = Path(__file__).parents[2]


def _identity(raw: bytes) -> dict[str, object]:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _scalar(
    method: str,
    dates: int,
    factor_sha256: str,
    direct_receipt_sha256: str,
    *,
    wall_trials: list[int],
    full_product_trials: list[int],
) -> dict[str, object]:
    evaluated = 256 * 256 - 1
    q_evaluations = 701
    adjusted = method == simulation.SELECTED_METHOD
    return {
        "method": method,
        "factor_sha256": factor_sha256,
        "direct_factor_receipt_sha256": direct_receipt_sha256,
        "factor_block_reads": 256,
        "nonreference_realized_rank": dates,
        "processed_pixels": 256 * 256,
        "evaluated_pixels": evaluated,
        "profile_fit_count": evaluated,
        "bootstrap_attempts": 0,
        "optimizer_rho_lane_evaluations": 1402,
        "optimizer_q_objective_evaluations": q_evaluations,
        "optimizer_primary_rho_pass_histogram": [evaluated] + [0] * 20,
        "covariance_parameter_derivative_lane_evaluations": (
            q_evaluations if adjusted else 0
        ),
        "covariance_parameter_adjustment_count": evaluated if adjusted else 0,
        "rayon_worker_count": 4,
        "maximum_worker_scratch_bytes": 4 * 1024 * 1024,
        "exact_optimizer_fallback_targets": 0,
        "condition_exact_fallbacks": 0,
        "wall_micros": max(wall_trials),
        "wall_micros_trials": wall_trials,
        "full_product_wall_micros": max(full_product_trials),
        "full_product_wall_micros_trials": full_product_trials,
        "peak_resident_set_bytes": 512 * 1024 * 1024,
        "checksum": 42.5,
    }


def _resource_receipt(
    batch_identity: dict[str, object],
    benchmark_identity: dict[str, object],
    selection_sha256: str | None,
) -> dict[str, object]:
    measurements = []
    for dates in (12, 48, 96):
        factor_sha256 = hashlib.sha256(f"factor-{dates}".encode()).hexdigest()
        direct_sha256 = hashlib.sha256(f"receipt-{dates}".encode()).hexdigest()
        plugin = _scalar(
            "plugin_gls_reml",
            dates,
            factor_sha256,
            direct_sha256,
            wall_trials=[100, 110],
            full_product_trials=[200, 220],
        )
        adjusted = _scalar(
            simulation.SELECTED_METHOD,
            dates,
            factor_sha256,
            direct_sha256,
            wall_trials=[150, 160],
            full_product_trials=[300, 330],
        )
        measurements.append({
            "post_gauge_date_count": dates,
            "acquisition_count": dates + 1,
            "target_count": 256 * 256,
            "varied_target_fingerprint_count": 258,
            "plugin_gls_reml": plugin,
            "reml_covariance_parameter_adjusted_scalar": adjusted,
            "adjusted_to_plugin_wall_ratio": 160 / 110,
            "adjusted_to_plugin_full_product_wall_ratio": 330 / 220,
        })
    return {
        "schema": simulation.TEMPORAL_RESOURCE_SCHEMA,
        "status": "pass" if selection_sha256 else "candidate_evidence_only",
        "benchmark_method": simulation.TEMPORAL_RESOURCE_BENCHMARK_METHOD,
        "baseline_method": "plugin_gls_reml",
        "candidate_method": simulation.SELECTED_METHOD,
            "candidate_method_version": simulation.SELECTED_METHOD_VERSION,
        "tile_rows": 256,
        "tile_columns": 256,
        "target_count": 256 * 256,
        "worker_scratch_limit_bytes": 8 * 1024 * 1024,
        "resident_set_limit_bytes": 24 * 1024 * 1024 * 1024,
        "maximum_targets_per_block": 65_536,
        "block_id_read_cap_bytes": 4 * 1024 * 1024,
        "factor_block_read_cap_bytes": 1024 * 1024 * 1024,
        "combined_working_set_cap_bytes": 2 * 1024 * 1024 * 1024,
        "product_source_sha256": simulation.sha256_file(
            ROOT / "crates/dolphin-workflows/src/temporal_covariance_product.rs",
            64 * 1024 * 1024,
        )[0],
        "benchmark_source_sha256": simulation.sha256_file(
            ROOT / "crates/dolphin-workflows/examples/temporal_inference_bench.rs",
            16 * 1024 * 1024,
        )[0],
        "batch_source_sha256": simulation.sha256_file(
            ROOT / "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs",
            16 * 1024 * 1024,
        )[0],
        "pre_outcome_selection_receipt_sha256": selection_sha256,
        "host": {
            "operating_system": "test-os",
            "architecture": "test-arch",
            "logical_processor_count": 12,
            "rayon_thread_count": 4,
            "omp_thread_count": 1,
            "openblas_thread_count": 1,
            "mkl_thread_count": 1,
            "veclib_thread_count": 1,
        },
        "temporal_covariance_batch_binary": batch_identity,
        "temporal_inference_bench_binary": benchmark_identity,
        "measurements": measurements,
    }


class TemporalResourceEvidenceTests(unittest.TestCase):
    def _write_chain(self, directory: Path) -> tuple[dict, dict, dict, dict]:
        batch_raw = b"final batch binary"
        benchmark_raw = b"final benchmark binary"
        (directory / simulation.TEMPORAL_BATCH_BINARY_FILENAME).write_bytes(batch_raw)
        (directory / simulation.TEMPORAL_INFERENCE_BENCH_BINARY_FILENAME).write_bytes(
            benchmark_raw
        )
        batch_identity = _identity(batch_raw)
        benchmark_identity = _identity(benchmark_raw)
        candidate = _resource_receipt(
            _identity(b"candidate batch"),
            _identity(b"candidate benchmark"),
            None,
        )
        candidate_raw = simulation.canonical_json_bytes(candidate) + b"\n"
        candidate_path = directory / simulation.TEMPORAL_CANDIDATE_RESOURCE_RECEIPT_FILENAME
        candidate_path.write_bytes(candidate_raw)
        selection = {
            "schema": simulation.TEMPORAL_METHOD_SELECTION_SCHEMA,
            "status": "pre_outcome_selected",
            "selected_method": simulation.SELECTED_METHOD,
            "selected_method_version": simulation.SELECTED_METHOD_VERSION,
            "candidate_resource_receipt_sha256": hashlib.sha256(
                candidate_raw
            ).hexdigest(),
            "canonical_v4_preregistration_sha256": simulation.canonical_v4_sha256(),
            "product_source_sha256": candidate["product_source_sha256"],
            "benchmark_source_sha256": candidate["benchmark_source_sha256"],
            "batch_source_sha256": candidate["batch_source_sha256"],
            "temporal_covariance_batch_binary_sha256": candidate[
                "temporal_covariance_batch_binary"
            ]["sha256"],
            "temporal_inference_bench_binary_sha256": candidate[
                "temporal_inference_bench_binary"
            ]["sha256"],
            "tile_rows": 256,
            "tile_columns": 256,
            "target_count": 256 * 256,
            "post_gauge_date_counts": [12, 48, 96],
            "adjusted_to_plugin_wall_ratio_limit": 2.0,
            "worker_scratch_limit_bytes": 8 * 1024 * 1024,
            "resident_set_limit_bytes": 24 * 1024 * 1024 * 1024,
            "outcomes_present": False,
        }
        selection_raw = simulation.canonical_json_bytes(selection) + b"\n"
        selection_path = directory / simulation.TEMPORAL_METHOD_SELECTION_FILENAME
        selection_path.write_bytes(selection_raw)
        selection_sha256 = hashlib.sha256(selection_raw).hexdigest()
        final = _resource_receipt(
            batch_identity,
            benchmark_identity,
            selection_sha256,
        )
        final_raw = simulation.canonical_json_bytes(final) + b"\n"
        (directory / simulation.TEMPORAL_RESOURCE_RECEIPT_FILENAME).write_bytes(
            final_raw
        )
        preregistration = {
            "selected_method": simulation.SELECTED_METHOD,
            "selected_method_version": simulation.SELECTED_METHOD_VERSION,
            "pre_outcome_selection_receipt_sha256": selection_sha256,
        }
        return preregistration, candidate, selection, final

    def test_complete_chain_is_structurally_valid_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preregistration, candidate, selection, final = self._write_chain(root)
            evidence = simulation.validate_release_resource_evidence(
                preregistration, root
            )
        self.assertEqual(
            evidence["method_selection_receipt_sha256"],
            preregistration["pre_outcome_selection_receipt_sha256"],
        )
        self.assertEqual(
            evidence["candidate_resource_receipt_sha256"],
            selection["candidate_resource_receipt_sha256"],
        )
        self.assertEqual(
            evidence["batch_binary"], final["temporal_covariance_batch_binary"]
        )
        self.assertEqual(candidate["status"], "candidate_evidence_only")

    def test_candidate_counter_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preregistration, candidate, _selection, _final = self._write_chain(root)
            candidate = deepcopy(candidate)
            candidate["measurements"][0]["plugin_gls_reml"][
                "condition_exact_fallbacks"
            ] = 1
            path = root / simulation.TEMPORAL_CANDIDATE_RESOURCE_RECEIPT_FILENAME
            path.write_bytes(simulation.canonical_json_bytes(candidate) + b"\n")
            with self.assertRaisesRegex(RuntimeError, "resource case"):
                simulation.validate_release_resource_evidence(preregistration, root)

    def test_selection_is_derived_only_from_valid_candidate_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _preregistration, candidate, _selection, _final = self._write_chain(root)
            candidate_path = (
                root / simulation.TEMPORAL_CANDIDATE_RESOURCE_RECEIPT_FILENAME
            )
            candidate_raw = candidate_path.read_bytes()
            selection = simulation.build_method_selection_receipt(candidate_path)
            self.assertEqual(
                selection["candidate_resource_receipt_sha256"],
                hashlib.sha256(candidate_raw).hexdigest(),
            )
            self.assertEqual(selection["selected_method"], simulation.SELECTED_METHOD)
            self.assertFalse(selection["outcomes_present"])
            candidate["status"] = "pass"
            candidate["pre_outcome_selection_receipt_sha256"] = "ab" * 32
            candidate_path.write_bytes(simulation.canonical_json_bytes(candidate) + b"\n")
            with self.assertRaisesRegex(RuntimeError, "candidate-only"):
                simulation.build_method_selection_receipt(candidate_path)

    def test_missing_candidate_and_observed_binary_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preregistration, _candidate, _selection, _final = self._write_chain(root)
            candidate_path = (
                root / simulation.TEMPORAL_CANDIDATE_RESOURCE_RECEIPT_FILENAME
            )
            candidate_raw = candidate_path.read_bytes()
            candidate_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "candidate resource"):
                simulation.validate_release_resource_evidence(preregistration, root)
            candidate_path.write_bytes(candidate_raw)
            (root / simulation.TEMPORAL_BATCH_BINARY_FILENAME).write_bytes(
                b"tampered binary"
            )
            with self.assertRaisesRegex(RuntimeError, "binary identity"):
                simulation.validate_release_resource_evidence(preregistration, root)


if __name__ == "__main__":
    unittest.main()
