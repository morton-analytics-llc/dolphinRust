"""Schema tests for the frozen #53 preregistration.

These tests validate the contract only; they intentionally do not run the
frozen scientific outcome experiment or acquire external GNSS data.
"""

import json
import importlib.util
import hashlib
import io
import math
import pathlib
import subprocess
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).parents[2]
FROZEN_PREREGISTRATION = ROOT / "validation/temporal_covariance_preregistration.json"
ENGINE_PREREGISTRATION = (
    ROOT / "validation/temporal_covariance_synthetic_engine_preregistration.json"
)
ENGINE_V4_PREREGISTRATION = (
    ROOT / "validation/temporal_covariance_synthetic_engine_preregistration_v4.json"
)


def load_generator():
    path = ROOT / "validation/temporal_covariance_simulation.py"
    spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_production_batch(request):
    requests = request if isinstance(request, list) else [request]
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "dolphin-timeseries", "--example",
         "temporal_covariance_batch"],
        cwd=ROOT,
        input="".join(json.dumps({
            "schema": "dolphinrust-temporal-covariance-batch/7",
            "requests": [value],
        }) + "\n" for value in requests),
        text=True,
        capture_output=True,
        check=True,
    )
    pairs = [json.loads(line)["records"] for line in result.stdout.splitlines()]
    return pairs if isinstance(request, list) else pairs[0]


def write_fake_batch(directory: pathlib.Path, batch_schema: str) -> pathlib.Path:
    path = directory / "fake_temporal_batch.py"
    counter = directory / "fake_temporal_batch.count"
    path.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import hashlib
        import json
        import os
        import pathlib
        import sys

        counter = pathlib.Path({str(counter)!r})
        count = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count + 1))
        methods = [
            "ols", "oracle_gls", "legacy_intercept_slope_wls_non_comparable",
            "lag_one_scalar_effective_n", "plugin_gls_reml",
            "reml_covariance_parameter_adjusted_scalar",
            "slope_profile_likelihood_ml", "complete_refit_bootstrap",
        ]
        def digest_record(record):
            payload = [
                record["schema"], record["execution_path"], record["cell_id"],
                record["cell_index"], record["outer_seed_index"],
                record["seed_sha256"], record["seed"], record["factor_sha256"],
                record["realized_factor_rank"], record["fixed_factor_status"],
                record["production_path_status"], record["comparator_methods"],
                record["fit"], record["provenance"], record["production_receipts"],
            ]
            encoded = json.dumps(payload, separators=(",", ":"))
            return hashlib.sha256(encoded.encode()).hexdigest()
        for line in sys.stdin:
            frame = json.loads(line)
            records = []
            for request in frame["requests"]:
                for execution_path in ("fixed_factor", "production_path"):
                    record = {{
                        "schema": {batch_schema!r},
                        "execution_path": execution_path,
                        "cell_id": request["cell_id"],
                        "cell_index": request["cell_index"],
                        "outer_seed_index": request["outer_seed_index"],
                        "seed_sha256": request["seed_sha256"],
                        "seed": request["seed"],
                        "factor_sha256": None,
                        "realized_factor_rank": None,
                        "fixed_factor_status": None,
                        "production_path_status": (
                            None if execution_path == "fixed_factor"
                            else "production_contract_mismatch"
                        ),
                        "comparator_methods": methods,
                        "attempted": True,
                        "emitted": False,
                        "failed": True,
                        "fit": None,
                        "provenance": None,
                        "production_receipts": None,
                        "record_sha256": "",
                    }}
                    record["record_sha256"] = digest_record(record)
                    records.append(record)
            count = len(frame["requests"])
            response = {{
                "schema": {batch_schema!r},
                "records": records,
                "resource": {{
                    "schema": "dolphinrust-temporal-covariance-batch-frame-resource/1",
                    "request_count": count,
                    "record_count": 2 * count,
                    "factor_generation_count": count,
                    "temporal_fit_count": count,
                    "profile_fit_count": 0,
                    "bootstrap_attempts": 0,
                    "attempt_record_count": 2 * count,
                    "rayon_worker_count": int(os.environ["RAYON_NUM_THREADS"]),
                    "wall_micros": count,
                    "resident_set_bytes_before": 1024,
                    "resident_set_bytes_after": 1024,
                }},
            }}
            print(json.dumps(response, separators=(",", ":")), flush=True)
        """))
    path.chmod(0o755)
    return path


class TemporalCovariancePreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with FROZEN_PREREGISTRATION.open() as handle:
            cls.frozen_prereg = json.load(handle)
        with ENGINE_PREREGISTRATION.open() as handle:
            cls.prereg = json.load(handle)

    def test_grid_dimensions_are_frozen(self):
        self.assertEqual(self.prereg["date_counts"], [12, 24, 48, 96])
        self.assertEqual(self.prereg["missingness"], [
            "none", "mcar_10_percent", "mcar_25_percent", "contiguous_20_percent"
        ])
        self.assertEqual(self.prereg["variance_ratios"], [1, 4, 16])
        self.assertEqual(self.prereg["reference_contribution_ratios"], [0, 0.5, 2])
        self.assertEqual(len(self.prereg["reference_contexts"]), 3)
        self.assertEqual(self.prereg["date_count_semantics"], "retained_post_gauge_dates_after_missingness")
        self.assertEqual(self.prereg["execution_paths"], ["fixed_factor", "production_path"])

    def test_boundary_is_fail_closed(self):
        self.assertEqual(self.prereg["status"], "pre_outcome_frozen")
        self.assertEqual(
            self.prereg["engine_validation_status"],
            "blocked_pending_complete_passing_synthetic_execution",
        )
        self.assertFalse(self.prereg["corrected_inferential_sigma_emission"])
        self.assertFalse(self.prereg["engine_validation"]["external_holdout_required"])
        self.assertFalse(self.prereg["engine_validation"]["independent_review_required"])
        self.assertEqual(self.prereg["eo_field_acceptance"]["owner"], "eo")

    def test_successor_moves_field_acceptance_out_of_engine_validation(self):
        frozen_bytes = FROZEN_PREREGISTRATION.read_bytes()
        frozen = json.loads(frozen_bytes)
        successor = json.loads(ENGINE_V4_PREREGISTRATION.read_bytes())
        frozen_canonical_sha256 = hashlib.sha256(
            json.dumps(
                frozen, allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()

        self.assertEqual(
            hashlib.sha256(frozen_bytes).hexdigest(),
            "8cebbdb4399fab031f1101be3c455fe7723b3a44f649c389777a7a35a0880ec6",
        )
        self.assertEqual(
            successor["schema"],
            "dolphinrust-temporal-covariance-preregistration/4",
        )
        self.assertEqual(
            successor["supersedes"],
            {
                "schema_version": 3,
                "canonical_preregistration_sha256": frozen_canonical_sha256,
                "outcomes_present": False,
                "reason": (
                    "separate engine-bounded synthetic validation from EO field "
                    "acceptance without changing the frozen synthetic experiment"
                ),
            },
        )
        self.assertEqual(successor["schemas"]["generator"],
                         "dolphinrust-temporal-covariance-simulation/8")
        self.assertEqual(
            successor["engine_validation"],
            {
                "attempt_count": 50_400,
                "required_gates": [
                    "exact_seed_denominator_complete",
                    "all_methods_pass",
                    "all_resource_gates_pass",
                    "producer_identity_match",
                ],
                "passing_status": "synthetic_validated_scope_match",
                "blocked_status": (
                    "blocked_pending_complete_passing_synthetic_execution"
                ),
                "external_holdout_required": False,
                "independent_review_required": False,
            },
        )
        self.assertEqual(
            successor["eo_field_acceptance"],
            {
                "owner": "eo",
                "evidence": ["held_out_gnss", "independent_review"],
                "required_for_engine_validation": False,
            },
        )
        self.assertNotIn("promotion_status", successor)
        self.assertNotIn("external_holdout_required", successor)
        self.assertEqual(
            successor["cell_count_without_outer_seeds"]
            * len(successor["execution_paths"])
            * successor["outer_seeds_per_supported_cell"],
            50_400,
        )
        changed_fields = {
            "schema",
            "supersedes",
            "producer_identity",
            "schemas",
            "file_hashes",
            "promotion_status",
            "external_holdout_required",
        }
        for field in set(frozen) - changed_fields:
            self.assertEqual(successor[field], frozen[field], field)
        for field in set(frozen["schemas"]) - {"generator"}:
            self.assertEqual(successor["schemas"][field], frozen["schemas"][field])
        for field in set(frozen["file_hashes"]) - {"generator_sha256"}:
            self.assertEqual(
                successor["file_hashes"][field], frozen["file_hashes"][field]
            )
        for field in set(frozen["producer_identity"]) - {"source_set_sha256"}:
            self.assertEqual(
                successor["producer_identity"][field],
                frozen["producer_identity"][field],
            )

    def test_v5_freezes_the_bounded_selected_method_execution(self):
        module = load_generator()
        v4_bytes = ENGINE_V4_PREREGISTRATION.read_bytes()
        v4 = json.loads(v4_bytes)
        successor = self.prereg
        expected_retained_bound = 24 * (
            module.MAX_SHARD_RECORD_BYTES
            + module.MAX_MANIFEST_BYTES
            + module.MAX_COMMIT_BYTES
        ) + (
            module.MAX_COMMIT_BYTES
            + module.MAX_MANIFEST_BYTES
            + module.MAX_COMMIT_BYTES
            + module.MAX_FINAL_RECEIPT_BYTES
        )

        self.assertEqual(
            successor["schema"],
            "dolphinrust-temporal-covariance-preregistration/5",
        )
        self.assertEqual(
            hashlib.sha256(v4_bytes).hexdigest(),
            "07c67eb9e6fb1b88143b6997cc289198a5ae998e2756b5916e905e82bc3e16ad",
        )
        self.assertEqual(
            module.canonical_v4_sha256(),
            "e98f01c9ad7223ae09ee3cb99df99c20b9752f04d291857c0a1359ac2e37ad99",
        )
        self.assertEqual(
            successor["supersedes"],
            {
                "schema_version": 4,
                "canonical_preregistration_sha256": module.canonical_v4_sha256(),
                "outcomes_present": False,
                "reason": (
                    "bind the pre-outcome method-v2 scalar, corrected actual-C54 "
                    "conditional DGP, active-set boundary inference, bounded "
                    "fallback, and cell-shard execution"
                ),
            },
        )
        self.assertEqual(
            successor["schemas"],
            {
                "generator": "dolphinrust-temporal-covariance-simulation/9",
                "batch": "dolphinrust-temporal-covariance-batch/7",
                "fixed_factor": "direct_difference_covariance/1",
                "production_path": (
                    "raw_complex_actual_evd_capture_replay_fixed_l2_"
                    "source_correlation/5"
                ),
                "scorer": "coverage_bias_interval_score/6",
                "provenance": "dolphinrust-temporal-covariance-provenance/2",
                "run_identity": module.RUN_IDENTITY_SCHEMA,
                "shard_manifest": module.SHARD_MANIFEST_SCHEMA,
                "shard_commit": module.SHARD_COMMIT_SCHEMA,
                "run_manifest": module.RUN_MANIFEST_SCHEMA,
                "run_commit": module.RUN_COMMIT_SCHEMA,
            },
        )
        self.assertEqual(successor["selected_method"], module.SELECTED_METHOD)
        self.assertEqual(
            successor["selected_method_version"], module.SELECTED_METHOD_VERSION
        )
        self.assertEqual(
            successor["promotion_methods"], list(module.FROZEN_PROMOTION_METHODS)
        )
        self.assertRegex(
            successor["pre_outcome_selection_receipt_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(successor["execution_protocol"]["shard_axis"], ["cell_index"])
        self.assertEqual(successor["execution_protocol"]["shard_count"], 24)
        self.assertEqual(successor["execution_protocol"]["frame_request_count"], 32)
        self.assertEqual(successor["execution_protocol"]["maximum_rayon_workers"], 12)
        self.assertEqual(successor["execution_protocol"]["seed_requests_per_shard"], 1050)
        self.assertEqual(successor["execution_protocol"]["attempt_records_per_shard"], 2100)
        self.assertEqual(successor["engine_validation"]["seed_request_count"], 25_200)
        self.assertEqual(successor["engine_validation"]["attempt_count"], 50_400)
        self.assertEqual(
            successor["generator"],
            {
                "latent_process": "stationary_irregular_continuous_time_ar1",
                "initial_state": "N(0,1)",
                "innovation_sd": "sqrt(1-rho^(2*gap/12))",
                "conditional_dgp": "y=Xbeta+F54*z+sqrt(q)*D(F54)*a_rho",
                "measurement_truth": (
                    "condition_on_actual_C54_factor_then_draw_F54_z_once"
                ),
                "temporal_process_truth": (
                    "sqrt(q)*sqrt(diag(C54)/geometric_mean_positive_diag(C54))*"
                    "continuous_time_ar1"
                ),
                "outer_coverage_dgp": module.OUTER_COVERAGE_DGP,
                "conditional_covariance_oracle": (
                    "fixed_capture_common_factor_monte_carlo_v1"
                ),
                "conditional_oracle_rule": (
                    "hold_capture_source_factors_and_phase_jvp_fixed_then_draw_"
                    "repeated_standard_normals_in_the_replay_joint_common_factor_"
                    "coordinates"
                ),
                "source_factor_shrinkage_alpha": 0.1,
                "normal_draw": "box_muller_from_splitmix64",
                "streams": {
                    "raw_complex": (
                        "proper_complex_stream_domain_v1:0xD1B54A32D192ED03"
                    ),
                    "missingness": (
                        "missingness_stream_domain_v1:0xA0761D6478BD642F"
                    ),
                    "temporal_process": (
                        "latent_ar_stream_domain_v1:0xE7037ED1A0B428DB"
                    ),
                    "measurement_factor": (
                        "measurement_normal_stream_domain_v1:0x0C54A53D9E3779B9"
                    ),
                    "bootstrap": "global_cell_outer_inner_splitmix64_v1",
                },
            },
        )
        self.assertEqual(
            successor["optimizer"]["boundary_policy"],
            "constrained_active_set_evaluated_and_recorded_v1",
        )
        self.assertEqual(
            successor["optimizer"]["nonconvergence_fallback"],
            (
                "coordinate_search_then_nested_rho_outer_log_process_variance_"
                "inner_adaptive_golden_section_v1"
            ),
        )
        self.assertEqual(successor["optimizer"]["max_iterations"], 12)
        self.assertEqual(
            successor["likelihoods"]["adjusted_scalar"],
            (
                "active_set_reml_nuisance_hessian_delta_method_with_student_t_"
                "residual_dof_v2"
            ),
        )
        self.assertIn("domain_v2", successor["cell_seed_rule"])
        active_boundaries = successor["unsupported_strata"][3:7]
        self.assertTrue(
            all(entry["expected_status"] == "Evaluated" for entry in active_boundaries)
        )
        self.assertEqual(
            [entry["expected_fitted_parameter_active_set"] for entry in active_boundaries],
            [
                "RhoLowerBoundary",
                "RhoUpperBoundary",
                "ProcessVarianceLowerBoundary",
                "ProcessVarianceUpperBoundary",
            ],
        )
        self.assertEqual(
            successor["resource_limits"]["retained_bound_bytes"],
            expected_retained_bound,
        )
        self.assertNotIn("projected_full_scene_minutes", successor["resource_limits"])
        changed_fields = {
            "schema",
            "supersedes",
            "promotion_methods",
            "cell_seed_rule",
            "unsupported_strata",
            "unsupported_cell_sha256",
            "optimizer",
            "generator",
            "likelihoods",
            "resource_limits",
            "execution_protocol",
            "producer_identity",
            "schemas",
            "file_hashes",
            "identities",
            "engine_validation",
        }
        for field in set(v4) - changed_fields:
            self.assertEqual(successor[field], v4[field], field)
        self.assertEqual(
            set(successor) - set(v4),
            {
                "selected_method",
                "selected_method_version",
                "pre_outcome_selection_receipt_sha256",
            },
        )

    def test_engine_receipt_uses_engine_bounded_status(self):
        module = load_generator()
        preregistration = json.loads(ENGINE_PREREGISTRATION.read_bytes())
        expected_attempts = (
            len(module.cells(preregistration))
            * preregistration["outer_seeds_per_supported_cell"]
            * len(preregistration["execution_paths"])
        )
        receipt = module._result_receipt(
            preregistration,
            preregistration["outer_seeds_per_supported_cell"],
            expected_attempts,
            expected_attempts,
            0,
            0,
            0,
            {"all_methods_pass": True},
            [],
            True,
            {},
            True,
        )
        self.assertTrue(receipt["engine_validation_eligible"])
        self.assertEqual(
            receipt["engine_validation_status"],
            "synthetic_validated_scope_match",
        )
        self.assertNotIn("promotion_eligible", receipt)
        self.assertNotIn("promotion_status", receipt)

    def test_holdout_wrappers_are_labeled_as_eo_field_acceptance(self):
        runner = (
            ROOT / "validation/run_temporal_covariance_holdout_cluster.py"
        ).read_text()
        scorer = (
            ROOT / "validation/score_temporal_covariance_holdout.py"
        ).read_text()
        self.assertIn("EO field acceptance", runner)
        self.assertIn("EO field acceptance", scorer)
        self.assertIn('"schema": "eo.temporal_covariance.field_acceptance_score"',
                      scorer)
        self.assertNotIn("held-out run identity differs from the promotion contract", runner)
        self.assertNotIn("held-out scorer output differs from the promotion schema", scorer)
        self.assertNotIn("held-out score path is not the frozen promotion path", scorer)

    def test_cell_count_and_seed_denominator_are_immutable(self):
        self.assertEqual(self.prereg["cell_count_without_outer_seeds"], 24)
        self.assertEqual(self.prereg["cell_count_by_execution_path"], {
            "fixed_factor": 24, "production_path": 24
        })
        self.assertEqual(
            self.prereg["supported_cell_sha256"],
            "e66e2a6f2b78f7f3307f3ae1e599f060e8abb23c179f21488155750a37e3a20e",
        )
        self.assertEqual(self.prereg["global_seed"], 5447718)
        self.assertEqual(
            self.prereg["attempt_denominator"],
            "all_attempted_outer_seeds_including_fit_failures",
        )
        self.assertEqual(self.prereg["bootstrap"]["interval_levels"], [0.68, 0.9, 0.95])
        self.assertEqual(self.prereg["bootstrap"]["count"], 200)
        self.assertEqual(self.prereg["bootstrap"]["minimum_successes"], 198)
        self.assertEqual(self.prereg["outer_seeds_per_supported_cell"], 1050)
        self.assertEqual(self.prereg["execution_protocol"]["shard_count"], 24)
        self.assertEqual(self.prereg["execution_protocol"]["shard_axis"], ["cell_index"])
        self.assertFalse(self.prereg["execution_protocol"]["top_up_allowed"])
        self.assertFalse(
            self.prereg["execution_protocol"]["dense_attempt_evidence_retained"]
        )

    def test_reduced_denominator_has_exact_wilson_margin_at_emission_floor(self):
        justification = self.prereg["seed_count_justification"]
        self.assertEqual(justification["minimum_scored_attempts"], 1040)
        for nominal, tolerance in self.prereg["coverage_tolerances"].items():
            lower, upper = justification["coverage_intervals"][nominal]
            target = float(nominal)
            self.assertGreaterEqual(lower, target - tolerance)
            self.assertLessEqual(upper, target + tolerance)
        self.assertTrue(justification["frozen_tolerances_unchanged"])

    def test_supported_cell_identities_match_frozen_hash(self):
        path = ROOT / "validation/temporal_covariance_simulation.py"
        spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frozen = module.cells(self.prereg)
        self.assertEqual(len(frozen), 24)
        self.assertEqual(module.cell_hash(frozen), self.prereg["supported_cell_sha256"])
        candidates = module.all_supported_cells(self.prereg)
        expected_pairs = set().union(*(module._pair_tokens(cell) for cell in candidates))
        actual_pairs = set().union(*(module._pair_tokens(cell) for cell in frozen))
        self.assertEqual(actual_pairs, expected_pairs)
        self.assertEqual(module.production_cells(self.prereg, frozen), frozen)
        unsupported = module.unsupported_cells(self.prereg)
        self.assertEqual(len(unsupported), 10)
        self.assertEqual(
            module.cell_hash(unsupported),
            "aa9d73e4fe7ebc28a445efb7db0e882ce73e42e208b89ea3305c07e5d7741cec",
        )
        self.assertEqual(
            module.cell_hash(unsupported), self.prereg["unsupported_cell_sha256"]
        )

    def test_compact_simulation_driver_is_deterministic_and_nonpromoting(self):
        module = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    "cargo", "build", "--release", "-p", "dolphin-timeseries",
                    "--example", "temporal_covariance_batch",
                ],
                check=True,
                cwd=ROOT,
            )
            binary = ROOT / "target/release/examples/temporal_covariance_batch"
            binary_identity = {
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "bytes": binary.stat().st_size,
            }
            resource_evidence = {
                "candidate_resource_receipt_sha256": "12" * 32,
                "method_selection_receipt_sha256": "34" * 32,
                "resource_receipt_sha256": "56" * 32,
                "batch_binary": binary_identity,
                "benchmark_binary": {"sha256": "78" * 32, "bytes": 1234},
            }
            producer = {
                "schema": module.RUN_IDENTITY_SCHEMA,
                "preregistration_sha256": hashlib.sha256(
                    module.canonical_json_bytes(self.prereg)
                ).hexdigest(),
                **self.prereg["file_hashes"],
                "source_set_schema": module.FROZEN_SOURCE_SET_SCHEMA,
                "source_set_sha256": self.prereg["producer_identity"][
                    "source_set_sha256"
                ],
                "binary_path": self.prereg["producer_identity"]["binary_path"],
                "binary_sha256": binary_identity["sha256"],
                "binary_bytes": binary_identity["bytes"],
                "batch_schema": self.prereg["schemas"]["batch"],
                "generator_schema": self.prereg["schemas"]["generator"],
                "source_correlation_model": module.SOURCE_CORRELATION_MODEL,
                "source_correlation_distance_scale_pixels": (
                    module.SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS
                ),
                "seed_count": self.prereg["outer_seeds_per_supported_cell"],
            }
            evidence_directory = pathlib.Path(directory)
            with (
                mock.patch.object(module, "producer_identity", return_value=producer),
                mock.patch.object(
                    module,
                    "validate_release_resource_evidence",
                    return_value=resource_evidence,
                ) as validate_resource,
            ):
                receipt = module.run(
                    self.prereg,
                    1,
                    1,
                    binary=binary,
                    resource_evidence_directory=evidence_directory,
                )
            validate_resource.assert_called_once_with(
                self.prereg, evidence_directory
            )
            self.assertEqual(receipt["expected_attempt_record_count"], 48)
            self.assertEqual(receipt["processed_attempt_record_count"], 2)
            self.assertEqual(receipt["emitted_attempt_record_count"], 2)
            self.assertEqual(receipt["failed_attempt_record_count"], 0)
            self.assertEqual(receipt["skipped_attempt_record_count"], 46)
            self.assertFalse(receipt["corrected_inferential_sigma_emission"])
            self.assertEqual(receipt["pre_outcome_status"], "pre_outcome_frozen")
            fixed, production = receipt["records"]
            self.assertEqual(fixed["execution_path"], "fixed_factor")
            self.assertEqual(fixed["fixed_factor_status"], "Evaluated")
            self.assertEqual(production["execution_path"], "production_path")
            self.assertEqual(production["production_path_status"], "evaluated")
            self.assertEqual(fixed["fit"]["valid_date_count"], 12)
            self.assertEqual(production["fit"]["status"], "Evaluated")
            self.assertEqual(
                production["fit"]["fitted_parameter_active_set"],
                "RhoLowerBoundary",
            )
            self.assertEqual(production["fit"]["valid_date_count"], 12)
            self.assertEqual(len(fixed["comparator_methods"]), 8)
            self.assertIsNotNone(production["provenance"])
            self.assertIsNotNone(production["production_receipts"])
            self.assertEqual(
                receipt["scores"]["schema"], self.prereg["schemas"]["scorer"]
            )
            self.assertEqual(receipt["scores"]["methods"]["ols"]["scored"], 2)
            self.assertEqual(len(receipt["scores"]["cell_summaries"]), 48)
            self.assertFalse(receipt["exact_seed_denominator_complete"])
            self.assertNotIn("resource", fixed)
            self.assertFalse(receipt["execution_complete"])
            self.assertFalse(receipt["engine_validation_eligible"])
            self.assertGreater(receipt["resource"]["peak_resident_set_bytes"], 0)

    def test_generator_uses_normal_draws_and_seed_varying_mcar(self):
        path = ROOT / "validation/temporal_covariance_simulation.py"
        spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = 1234
        draws = []
        for _ in range(20000):
            state, draw = module.normal_noise(state)
            draws.append(draw)
        mean = sum(draws) / len(draws)
        variance = sum((draw - mean) ** 2 for draw in draws) / (len(draws) - 1)
        self.assertLess(abs(mean), 0.03)
        self.assertLess(abs(variance - 1.0), 0.04)
        cell = next(cell for cell in module.cells(self.prereg)
                    if cell["date_count"] == 24 and cell["missingness"] == "mcar_25_percent")
        count = len(module.days_for(cell)) - 1
        self.assertNotEqual(
            module.missing_indices(cell, 1, count),
            module.missing_indices(cell, 2, count),
        )
        request = module.request_for(cell, 1, self.prereg)
        retained = sum(request["production_path"]["validity"]) - 1
        self.assertEqual(retained, cell["date_count"])
        production = module.request_for(cell, 1, self.prereg)["production_path"]
        self.assertNotIn("execution_path", request)
        self.assertNotIn("fixed_factor", request)
        self.assertNotIn("difference_covariance", json.dumps(request))
        self.assertNotIn("issue52_target_factor", production)
        self.assertNotIn("issue54_difference_factor", production)
        self.assertNotIn("issue52_seed", production)
        self.assertNotIn("issue54_seed", production)
        self.assertEqual(production["native_shape"], [1, 7])
        self.assertEqual(len(production["raw_complex_stack"]), len(module.days_for(cell)))
        self.assertTrue(all(len(row) == 7 for row in production["raw_complex_stack"]))
        self.assertEqual(len(production["capture_scope_sha256"]), 64)
        self.assertEqual(production["outer_coverage_dgp"], module.OUTER_COVERAGE_DGP)
        self.assertEqual(
            production["conditional_covariance_oracle"],
            module.CONDITIONAL_COVARIANCE_ORACLE,
        )

    def test_frame_contract_is_same_cell_consecutive_and_bounded(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        requests = [module.request_for(cell, seed, self.prereg) for seed in range(32)]
        frame = module.request_frame(requests, "dolphinrust-temporal-covariance-batch/7")
        self.assertEqual(len(frame["requests"]), 32)
        self.assertLessEqual(
            len(module.canonical_json_bytes(frame)) + 1,
            module.MAX_REQUEST_LINE_BYTES,
        )
        with self.assertRaisesRegex(RuntimeError, "request count"):
            module.request_frame([], frame["schema"])
        with self.assertRaisesRegex(RuntimeError, "request count"):
            module.request_frame(requests + [requests[-1]], frame["schema"])
        nonconsecutive = json.loads(json.dumps(requests[:2]))
        nonconsecutive[1]["outer_seed_index"] = 3
        with self.assertRaisesRegex(RuntimeError, "same-cell consecutive"):
            module.request_frame(nonconsecutive, frame["schema"])
        cross_cell = json.loads(json.dumps(requests[:2]))
        cross_cell[1]["cell_id"] = "different-cell"
        with self.assertRaisesRegex(RuntimeError, "same-cell consecutive"):
            module.request_frame(cross_cell, frame["schema"])

    def test_production_batch_uses_actual_capture_replay_and_fixed_l2(self):
        source = (ROOT / "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs").read_text()
        self.assertNotIn("InfluenceDag", source)
        self.assertNotIn("build_raw_dag", source)
        self.assertIn("run_sequential_with_covariance_capture_and_source_factors", source)
        self.assertIn("replay_global_reference_difference_covariance_from_provider_bundle", source)
        self.assertIn("propagate_fixed_l2_difference_covariance", source)
        self.assertIn("SourceCorrelationModel::ExponentialEuclidean", source)
        self.assertIn("source_correlation_support_union_count", source)
        self.assertIn("conditional_common_factor_covariance", source)
        self.assertNotIn('"source_factor_declared_v1"', source)

    def test_production_raw_ensemble_has_proper_complex_support_moments(self):
        module = load_generator()
        seed_count = module.PROPER_COMPLEX_MOMENT_SEEDS
        innovations = {
            column: [
                complex(*module.proper_complex_innovation(3, outer_seed, 5, column))
                for outer_seed in range(seed_count)
            ]
            for column in range(7)
        }
        for column_values in innovations.values():
            mean = sum(column_values) / seed_count
            variance = sum(abs(value) ** 2 for value in column_values) / seed_count
            pseudo_covariance = sum(value * value for value in column_values) / seed_count
            self.assertLessEqual(abs(mean.real), 4.0 / math.sqrt(seed_count))
            self.assertLessEqual(abs(mean.imag), 4.0 / math.sqrt(seed_count))
            self.assertLessEqual(abs(variance - 2.0), 8.0 / math.sqrt(seed_count))
            self.assertLessEqual(abs(pseudo_covariance.real), 8.0 / math.sqrt(seed_count))
            self.assertLessEqual(abs(pseudo_covariance.imag), 8.0 / math.sqrt(seed_count))

        for left in range(7):
            for right in range(7):
                covariance = sum(
                    innovations[left][seed] * innovations[right][seed].conjugate()
                    for seed in range(seed_count)
                ) / seed_count
                expected = 2.0 * module.spatial_correlation(left, right)
                self.assertLessEqual(abs(covariance.real - expected),
                                     8.0 / math.sqrt(seed_count))
                self.assertLessEqual(abs(covariance.imag), 8.0 / math.sqrt(seed_count))

        target_support = module.support_columns(1, 7)
        for reference_column, shared_columns in ((2, 2), (3, 1), (5, 0)):
            reference_support = module.support_columns(reference_column, 7)
            target_means = [
                sum(innovations[column][seed] for column in target_support) / 3.0
                for seed in range(seed_count)
            ]
            reference_means = [
                sum(innovations[column][seed] for column in reference_support) / 3.0
                for seed in range(seed_count)
            ]
            covariance = sum(
                (target * reference.conjugate()).real
                for target, reference in zip(target_means, reference_means)
            ) / seed_count
            target_variance = sum(abs(value) ** 2 for value in target_means) / seed_count
            reference_variance = sum(abs(value) ** 2 for value in reference_means) / seed_count
            correlation = covariance / math.sqrt(target_variance * reference_variance)
            expected = module.production_support_correlation(1, reference_column, 7)
            self.assertLessEqual(abs(correlation - expected), 4.0 / math.sqrt(seed_count))
            self.assertAlmostEqual(
                module.support_intersection_correlation(1, reference_column, 7),
                shared_columns / 3.0,
                places=12,
            )
            variance = 0.07
            fraction = module.production_temporal_noise_fraction(variance, expected)
            self.assertAlmostEqual(
                1.0 - fraction,
                math.exp(-variance / (2.0 * (1.0 - expected))),
                places=12,
            )

        noise_fraction = 0.2
        carrier = complex(math.cos(0.4), math.sin(0.4))
        temporal = []
        for outer_seed in range(seed_count):
            common = complex(*module.proper_complex_speckle(3, outer_seed, 2))
            innovation = complex(*module.proper_complex_innovation(3, outer_seed, 5, 2))
            temporal.append((
                common,
                carrier * (
                    math.sqrt(1.0 - noise_fraction) * common
                    + math.sqrt(noise_fraction) * innovation
                ),
            ))
        covariance = sum(
            right * left.conjugate() for left, right in temporal
        ) / seed_count
        expected = 2.0 * math.sqrt(1.0 - noise_fraction) * carrier
        self.assertLessEqual(abs(covariance - expected), 12.0 / math.sqrt(seed_count))

    def test_production_request_separates_raw_measurement_from_latent_temporal_signal(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        outer_seed = 17
        request = module.request_for(cell, outer_seed, self.prereg)
        production = request["production_path"]
        seed = request["seed"]
        days = request["days"]
        state, ar_path = module.stationary_ar_path(
            days,
            cell["rho_at_12_days"],
            module.splitmix64(seed ^ module.LATENT_AR_STREAM_DOMAIN),
        )
        del state
        self.assertEqual(production["latent_ar_path"], [0.0] + ar_path[1:])
        self.assertEqual(
            production["measurement_normal_path"],
            module.standard_normal_path(
                len(days), module.splitmix64(seed ^ 0xC54A53D9E3779B9)
            ),
        )
        self.assertEqual(production["truth_slope_per_day"], 0.01)
        self.assertEqual(
            production["outer_coverage_dgp"],
            "actual_c54_gaussian_measurement_post_link_ar_v1",
        )
        diagonal = []
        for index in range(len(days)):
            variance_scale = 1.0 if index < len(days) // 2 else cell["variance_ratio"]
            diagonal.append(0.01 * (variance_scale + cell["reference_contribution_ratio"]))
        reference_column = production["reference_pixel"][1]
        correlation = module.production_support_correlation(1, reference_column, 7)
        common_speckle = [
            complex(*module.proper_complex_speckle(cell["cell_index"], outer_seed, column))
            for column in range(7)
        ]
        for date_index, (day, row) in enumerate(zip(days, production["raw_complex_stack"])):
            carrier_value = 0.0 if date_index == 0 else 0.01 * day
            noise_fraction = module.production_temporal_noise_fraction(
                0.0 if date_index == 0 else diagonal[date_index], correlation
            )
            for column, raw in enumerate(row):
                phase = ((reference_column - column) / (reference_column - 1)) * carrier_value
                carrier = complex(math.cos(phase), math.sin(phase))
                innovation = complex(
                    *module.proper_complex_innovation(
                        cell["cell_index"], outer_seed, date_index, column
                    )
                )
                expected = carrier * (
                    math.sqrt(1.0 - noise_fraction) * common_speckle[column]
                    + math.sqrt(noise_fraction) * innovation
                )
                self.assertAlmostEqual(raw[0], expected.real, places=12)
                self.assertAlmostEqual(raw[1], expected.imag, places=12)
                truth = production["carrier_stack"][date_index][column]
                self.assertAlmostEqual(truth[0], carrier.real, places=12)
                self.assertAlmostEqual(truth[1], carrier.imag, places=12)

    def test_request_domain_separates_missingness_process_and_measurement_draws(self):
        module = load_generator()
        cell = next(
            cell
            for cell in module.cells(self.prereg)
            if cell["missingness"] == "mcar_25_percent"
        )
        request = module.request_for(cell, 17, self.prereg)
        seed = request["seed"]
        days = request["days"]
        missing_seed = module.splitmix64(seed ^ 0xA0761D6478BD642F)
        latent_seed = module.splitmix64(seed ^ 0xE7037ED1A0B428DB)
        expected_missing = module.missing_indices(cell, missing_seed, len(days) - 1)
        self.assertEqual(
            [
                index
                for index, valid in enumerate(request["production_path"]["validity"])
                if not valid
            ],
            sorted(expected_missing),
        )
        _, ar_path = module.stationary_ar_path(
            days, cell["rho_at_12_days"], latent_seed
        )
        self.assertEqual(
            request["production_path"]["latent_ar_path"], [0.0] + ar_path[1:]
        )
        self.assertEqual(
            request["production_path"]["measurement_normal_path"],
            module.standard_normal_path(
                len(days), module.splitmix64(seed ^ 0x0C54A53D9E3779B9)
            ),
        )

    def test_post_link_dgp_uses_actual_retained_c54_and_keeps_raw_receipts_independent(self):
        module = load_generator()
        request = module.request_for(module.cells(self.prereg)[0], 0, self.prereg, True)
        request["options"]["bootstrap_replicates"] = 0
        request["options"]["bootstrap_minimum_successes"] = 0
        mutated = json.loads(json.dumps(request))
        mutated["production_path"]["latent_ar_path"][1] += 0.125
        mutated["production_path"]["capture_scope_sha256"] = (
            module.capture_scope_sha256(mutated)
        )
        pairs = run_production_batch([request, mutated])

        def assert_identity(candidate, pair):
            fixed, production = pair
            self.assertEqual(fixed["fit"], production["fit"])
            receipts = production["production_receipts"]
            factor = receipts["fixed_l2_difference_factor"]
            diagonal = [sum(value * value for value in row) for row in factor]
            validity = candidate["production_path"]["validity"]
            retained = [
                index for index in range(1, len(diagonal)) if validity[index]
            ]
            scale = math.exp(
                sum(math.log(diagonal[index]) for index in retained) / len(retained)
            )
            carrier = receipts["carrier_difference_history"]
            linked = receipts["linked_difference_history"]
            for index in range(len(diagonal)):
                expected = 0.0 if index == 0 else (
                    candidate["production_path"]["truth_slope_per_day"]
                    * candidate["days"][index]
                    + math.sqrt(candidate["options"]["oracle_process_variance"])
                    * math.sqrt(diagonal[index] / scale)
                    * candidate["production_path"]["latent_ar_path"][index]
                )
                self.assertAlmostEqual(carrier[index], expected, places=12)
                measurement_error = 0.0 if index == 0 else sum(
                    factor[index][component]
                    * candidate["production_path"]["measurement_normal_path"][component]
                    for component in range(receipts["fixed_l2_realized_rank"])
                )
                self.assertAlmostEqual(
                    linked[index] - carrier[index], measurement_error,
                    places=12,
                )
            return fixed, production, receipts

        base_fixed, base_production, base_receipts = assert_identity(request, pairs[0])
        changed_fixed, changed_production, changed_receipts = assert_identity(mutated, pairs[1])
        self.assertEqual(base_fixed["factor_sha256"], changed_fixed["factor_sha256"])
        self.assertEqual(
            base_receipts["source_manifest_sha256"],
            changed_receipts["source_manifest_sha256"],
        )
        self.assertEqual(
            base_receipts["issue54_receipt_sha256"],
            changed_receipts["issue54_receipt_sha256"],
        )
        self.assertEqual(
            base_receipts["source_linked_difference_history"],
            changed_receipts["source_linked_difference_history"],
        )
        self.assertNotEqual(
            base_receipts["temporal_dgp_receipt_sha256"],
            changed_receipts["temporal_dgp_receipt_sha256"],
        )
        self.assertNotEqual(base_fixed["fit"], changed_fixed["fit"])
        self.assertNotEqual(
            base_production["production_receipts"]["linked_difference_history"],
            changed_production["production_receipts"]["linked_difference_history"],
        )

    def test_raw_outer_dgp_is_not_the_common_factor_conditional_oracle(self):
        module = load_generator()
        base_cell = module.cells(self.prereg)[0]
        cell = dict(
            base_cell,
            cell_id="raw-space-whitening-diagnostic",
            reference_context="near_exact",
            overlap_fraction=0.5,
            distance_pixels=1,
        )
        fixed = module.request_for(cell, 0, self.prereg)
        fixed_stack = fixed["production_path"]["raw_complex_stack"]
        dates = len(fixed_stack)

        def cholesky(covariance):
            lower = [[0j for _ in range(dates)] for _ in range(dates)]
            for row in range(dates):
                for column in range(row + 1):
                    residual = covariance[row][column] - sum(
                        lower[row][inner] * lower[column][inner].conjugate()
                        for inner in range(column)
                    )
                    lower[row][column] = (
                        complex(math.sqrt(residual.real), 0.0)
                        if row == column else residual / lower[column][column].real
                    )
            return lower

        def source_factor(column):
            support = module.support_columns(column, 7)
            covariance = [[0j for _ in range(dates)] for _ in range(dates)]
            for left in range(dates):
                for right in range(dates):
                    covariance[left][right] = sum(
                        complex(*fixed_stack[left][pixel])
                        * complex(*fixed_stack[right][pixel]).conjugate()
                        for pixel in support
                    ) / len(support)
                    if left != right:
                        covariance[left][right] *= 0.9
            return cholesky(covariance)

        def invert_lower(lower):
            inverse = [[0j for _ in range(dates)] for _ in range(dates)]
            for column in range(dates):
                for row in range(column, dates):
                    residual = complex(row == column, 0.0) - sum(
                        lower[row][inner] * inverse[inner][column]
                        for inner in range(column, row)
                    )
                    inverse[row][column] = residual / lower[row][row].real
            return inverse

        def multiply(left, right):
            return [[sum(left[row][inner] * right[inner][column]
                         for inner in range(dates))
                     for column in range(dates)] for row in range(dates)]

        def conjugate_transpose(matrix):
            return [[matrix[column][row].conjugate() for column in range(dates)]
                    for row in range(dates)]

        ensemble = [
            module.request_for(cell, seed, self.prereg)
            ["production_path"]["raw_complex_stack"]
            for seed in range(512)
        ]
        cross = [[0j for _ in range(dates)] for _ in range(dates)]
        for left in range(dates):
            for right in range(dates):
                left_values = [complex(*stack[left][1]) for stack in ensemble]
                right_values = [complex(*stack[right][2]) for stack in ensemble]
                left_mean = sum(left_values) / len(left_values)
                right_mean = sum(right_values) / len(right_values)
                cross[left][right] = sum(
                    (left_value - left_mean) * (right_value - right_mean).conjugate()
                    for left_value, right_value in zip(left_values, right_values)
                ) / (len(ensemble) - 1)
        whitened = multiply(
            multiply(invert_lower(source_factor(1)), cross),
            conjugate_transpose(invert_lower(source_factor(2))),
        )
        expected = module.spatial_correlation(1, 2)
        maximum_gap = max(
            abs(whitened[row][column] - (expected if row == column else 0.0))
            for row in range(dates) for column in range(dates)
        )
        self.assertGreater(maximum_gap, 0.25)

    def test_conditional_common_factor_oracle_matches_fixed_capture_prediction(self):
        module = load_generator()
        base_cell = module.cells(self.prereg)[0]
        replicates = 8192
        requests = []
        for context, (jaccard, distance) in {
                "near_exact": (0.5, 1),
                "mid_exact": (0.2, 2),
                "far_exact": (0.0, 4),
        }.items():
            cell = dict(
                base_cell,
                cell_id=f"conditional-{context}",
                reference_context=context,
                overlap_fraction=jaccard,
                distance_pixels=distance,
            )
            request = module.request_for(cell, 0, self.prereg)
            request["retain_dense_evidence"] = True
            request["conditional_oracle_replicates"] = replicates
            request["options"]["bootstrap_replicates"] = 0
            request["options"]["bootstrap_minimum_successes"] = 0
            requests.append(request)
        for request, pair in zip(requests, run_production_batch(requests)):
            fixed, record = pair
            self.assertEqual(fixed["fit"], record["fit"])
            self.assertEqual(fixed["factor_sha256"], record["factor_sha256"])
            receipts = record["production_receipts"]
            self.assertEqual(
                receipts["outer_coverage_dgp"],
                "actual_c54_gaussian_measurement_post_link_ar_v1",
            )
            self.assertEqual(
                receipts["conditional_covariance_oracle"],
                "fixed_capture_common_factor_monte_carlo_v1",
            )
            self.assertEqual(receipts["conditional_oracle_replicates"], replicates)
            predicted = receipts["fixed_l2_difference_covariance"]
            empirical = receipts["conditional_oracle_covariance"]
            for left in range(len(predicted)):
                for right in range(len(predicted)):
                    if left == 0 or right == 0:
                        self.assertEqual(empirical[left][right], 0.0)
                        continue
                    standard_error = math.sqrt(
                        (predicted[left][left] * predicted[right][right]
                         + predicted[left][right] ** 2) / (replicates - 1)
                    )
                    self.assertLessEqual(
                        abs(empirical[left][right] - predicted[left][right]),
                        4.0 * standard_error + 1e-12,
                        (request["cell_id"], left, right),
                    )

    def test_stationary_irregular_ar_generator_matches_oracle_covariance(self):
        path = ROOT / "validation/temporal_covariance_simulation.py"
        spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pairs = []
        for seed in range(5000):
            _, values = module.stationary_ar_path([0.0, 18.0], 0.6, seed)
            pairs.append(values)
        left_mean = sum(pair[0] for pair in pairs) / len(pairs)
        right_mean = sum(pair[1] for pair in pairs) / len(pairs)
        left_variance = sum((pair[0] - left_mean) ** 2 for pair in pairs) / (len(pairs) - 1)
        right_variance = sum((pair[1] - right_mean) ** 2 for pair in pairs) / (len(pairs) - 1)
        covariance = sum((pair[0] - left_mean) * (pair[1] - right_mean)
                         for pair in pairs) / (len(pairs) - 1)
        self.assertLess(abs(left_variance - 1.0), 0.05)
        self.assertLess(abs(right_variance - 1.0), 0.05)
        self.assertLess(abs(covariance - 0.6 ** 1.5), 0.05)

    def test_production_path_seed_mismatch_fails_closed(self):
        path = ROOT / "validation/temporal_covariance_simulation.py"
        spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cell = module.cells(self.prereg)[0]
        request = module.request_for(cell, 99, self.prereg)
        request["production_path"]["source_seed"] = 100
        fixed, record = run_production_batch(request)
        self.assertIsNone(fixed["fit"])
        self.assertEqual(record["production_path_status"], "source_seed_mismatch")
        self.assertFalse(record["emitted"])
        self.assertIsNone(record["fit"])

    def test_production_path_scope_mismatch_fails_closed(self):
        path = ROOT / "validation/temporal_covariance_simulation.py"
        spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cell = module.cells(self.prereg)[0]
        request = module.request_for(cell, 99, self.prereg)
        request["production_path"]["reference"]["distance_pixels"] += 1.0
        fixed, record = run_production_batch(request)
        self.assertIsNone(fixed["fit"])
        self.assertEqual(record["production_path_status"], "capture_scope_mismatch")
        self.assertFalse(record["emitted"])
        self.assertIsNone(record["fit"])

    def test_production_path_requires_frozen_method_and_synthetic_scope(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        mutations = [
            ("selected_method", "plugin_gls"),
            ("scope", "field_validation"),
            ("source_correlation_model", "caller_claimed_model"),
            ("source_correlation_distance_scale_pixels", 3.0),
            ("outer_coverage_dgp", "caller_claimed_dgp"),
            ("conditional_covariance_oracle", "caller_claimed_oracle"),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                request = module.request_for(cell, 99, self.prereg)
                request["production_path"][field] = value
                record = run_production_batch(request)[1]
                self.assertEqual(record["production_path_status"],
                                 "production_contract_mismatch")
                self.assertFalse(record["emitted"])
                self.assertIsNone(record["fit"])
        request = module.request_for(cell, 99, self.prereg)
        request["conditional_oracle_replicates"] = 16_385
        record = run_production_batch(request)[1]
        self.assertEqual(record["production_path_status"], "production_contract_mismatch")
        self.assertFalse(record["emitted"])
        self.assertIsNone(record["fit"])

    def test_production_path_rejects_self_consistent_claimed_geometry(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        request = module.request_for(cell, 99, self.prereg)
        request["production_path"]["reference"]["overlap_fraction"] = 0.25
        request["production_path"]["capture_scope_sha256"] = module.capture_scope_sha256(request)
        record = run_production_batch(request)[1]
        self.assertEqual(record["production_path_status"], "reference_context_mismatch")
        self.assertFalse(record["emitted"])
        self.assertIsNone(record["fit"])

    def test_production_path_captures_replays_and_binds_actual_receipts(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        request = module.request_for(cell, 99, self.prereg)
        request["options"]["bootstrap_replicates"] = 0
        request["options"]["bootstrap_minimum_successes"] = 0
        fixed, record = run_production_batch(request)
        self.assertEqual(fixed["fit"], record["fit"])
        self.assertEqual(fixed["factor_sha256"], record["factor_sha256"])
        self.assertEqual(record["production_path_status"], "estimator_failed")
        self.assertEqual(record["fit"]["status"], "DiagnosticNotComputed")
        self.assertEqual(
            record["fit"]["fitted_parameter_active_set"], "RhoUpperBoundary"
        )
        self.assertFalse(record["emitted"])
        self.assertIsNone(record["provenance"])
        receipts = record["production_receipts"]
        self.assertEqual(receipts["capture_scope_sha256"],
                         request["production_path"]["capture_scope_sha256"])
        digest_fields = [
            field for field in receipts
            if field.endswith("_sha256")
        ]
        for field in digest_fields:
            value = receipts[field]
            self.assertRegex(value, r"^[0-9a-f]{64}$")
            self.assertNotEqual(value, "0" * 64)
        self.assertEqual(receipts["source_correlation_model"],
                         module.SOURCE_CORRELATION_MODEL)
        self.assertEqual(receipts["source_correlation_distance_scale_pixels"],
                         module.SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS)
        self.assertGreater(receipts["source_correlation_support_union_count"], 0)
        self.assertGreater(receipts["effective_looks_fraction"], 0.0)
        self.assertLessEqual(receipts["effective_looks_fraction"], 1.0)

        drifted = json.loads(json.dumps(request))
        drifted["production_path"]["raw_complex_stack"][0][0][0] *= 1.0001
        drifted_record = run_production_batch(drifted)[1]
        drifted_receipts = drifted_record["production_receipts"]
        self.assertIsNotNone(drifted_receipts)
        for identity in (
            "source_manifest_sha256",
            "evd_operator_sha256",
            "evd_source_factor_sha256",
            "issue52_receipt_sha256",
            "issue54_receipt_sha256",
        ):
            self.assertNotEqual(receipts[identity], drifted_receipts[identity])

    def test_streaming_scorer_rejects_missing_or_reordered_seed(self):
        path = ROOT / "validation/temporal_covariance_simulation.py"
        spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        request_iter = module.iter_requests(self.prereg, 2)
        self.assertNotIsInstance(request_iter, list)
        request = next(request_iter)
        scorer = module.StreamingScores(self.prereg)
        stale = {
            "cell_id": request["cell_id"],
            "execution_path": "fixed_factor",
            "outer_seed_index": 1,
            "seed": request["seed"],
            "seed_sha256": request["seed_sha256"],
            "fit": None,
        }
        with self.assertRaisesRegex(RuntimeError, "duplicate, missing, or reordered"):
            scorer.update(stale)

    def test_resumable_shard_cleans_partial_and_reuses_exact_commit(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = write_fake_batch(root, self.prereg["schemas"]["batch"])
            identity = {
                "schema": module.RUN_IDENTITY_SCHEMA,
                "batch_schema": self.prereg["schemas"]["batch"],
                "source_set_sha256": "11" * 32,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "source_correlation_model": module.SOURCE_CORRELATION_MODEL,
                "source_correlation_distance_scale_pixels": (
                    module.SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS
                ),
                "seed_count": 2,
            }
            shards = module.initialize_run_root(root / "run", identity)
            paths = module._shard_paths(shards, cell)
            partial = paths["records"].with_name(paths["records"].name + ".partial")
            partial.write_bytes(b"owned interrupted bytes")
            manifest, reused = module.execute_or_resume_shard(
                self.prereg, cell, 2, shards, binary, identity
            )
            self.assertFalse(reused)
            self.assertEqual(manifest["seed_request_count"], 2)
            self.assertEqual(manifest["attempted"], 4)
            self.assertFalse(partial.exists())
            committed = paths["records"].read_bytes().splitlines()
            self.assertEqual(len(committed), 4)
            self.assertTrue(all(b"fixed_l2_difference_covariance" not in line
                                for line in committed))
            resumed, reused = module.execute_or_resume_shard(
                self.prereg, cell, 2, shards, binary, identity
            )
            self.assertTrue(reused)
            self.assertEqual(resumed, manifest)
            self.assertEqual((root / "fake_temporal_batch.count").read_text(), "1")

    def test_resumable_shard_rejects_tamper_missing_duplicate_and_reorder(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]

        def committed(directory: pathlib.Path):
            binary = write_fake_batch(directory, self.prereg["schemas"]["batch"])
            identity = {
                "schema": module.RUN_IDENTITY_SCHEMA,
                "batch_schema": self.prereg["schemas"]["batch"],
                "source_set_sha256": "11" * 32,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "seed_count": 2,
            }
            shards = module.initialize_run_root(directory / "run", identity)
            module.execute_or_resume_shard(
                self.prereg, cell, 2, shards, binary, identity
            )
            return binary, identity, shards, module._shard_paths(shards, cell)

        def rebind(paths, lines):
            payload = b"".join(line.rstrip(b"\n") + b"\n" for line in lines)
            semantic = hashlib.sha256()
            for line in lines:
                semantic.update(module._response_semantic_bytes(json.loads(line)))
            paths["records"].write_bytes(payload)
            manifest = json.loads(paths["manifest"].read_bytes())
            manifest["records_sha256"] = hashlib.sha256(payload).hexdigest()
            manifest["records_bytes"] = len(payload)
            manifest["response_semantic_sha256"] = semantic.hexdigest()
            manifest_bytes = module.canonical_json_bytes(manifest) + b"\n"
            paths["manifest"].write_bytes(manifest_bytes)
            commit = {
                "schema": module.SHARD_COMMIT_SCHEMA,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "records_sha256": hashlib.sha256(payload).hexdigest(),
                "response_semantic_sha256": semantic.hexdigest(),
                "producer_source_set_sha256": manifest[
                    "producer_source_set_sha256"
                ],
                "producer_binary_sha256": manifest["producer_binary_sha256"],
            }
            paths["commit"].write_bytes(module.canonical_json_bytes(commit) + b"\n")

        with self.subTest("same-byte tamper"):
            with tempfile.TemporaryDirectory() as directory:
                binary, identity, shards, paths = committed(pathlib.Path(directory))
                paths["records"].write_bytes(paths["records"].read_bytes() + b" ")
                with self.assertRaisesRegex(RuntimeError, "hash is stale or tampered"):
                    module.execute_or_resume_shard(
                        self.prereg, cell, 2,
                        shards, binary, identity,
                    )
        for label, mutate, pattern in (
            ("missing", lambda lines: lines[:-1], "missing, duplicated, or reordered"),
            ("duplicate", lambda lines: [lines[0], lines[0], *lines[2:]], "record pair"),
            ("reorder", lambda lines: list(reversed(lines)), "attempt identity"),
        ):
            with self.subTest(label):
                with tempfile.TemporaryDirectory() as directory:
                    binary, identity, shards, paths = committed(pathlib.Path(directory))
                    lines = paths["records"].read_bytes().splitlines()
                    rebind(paths, mutate(lines))
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        module.execute_or_resume_shard(
                            self.prereg, cell, 2,
                            shards, binary, identity,
                        )

    def test_compact_record_digest_rejects_stale_allowed_semantic_tamper(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        request = module.request_for(cell, 0, self.prereg)
        record = {
            "schema": self.prereg["schemas"]["batch"],
            "execution_path": "production_path",
            "cell_id": request["cell_id"],
            "cell_index": request["cell_index"],
            "outer_seed_index": request["outer_seed_index"],
            "seed_sha256": request["seed_sha256"],
            "seed": request["seed"],
            "factor_sha256": None,
            "realized_factor_rank": None,
            "fixed_factor_status": None,
            "production_path_status": "production_inputs_missing",
            "comparator_methods": module.COMPARATOR_METHOD_IDENTITIES,
            "attempted": True,
            "emitted": False,
            "failed": True,
            "fit": None,
            "provenance": None,
            "production_receipts": None,
            "record_sha256": None,
        }
        record["record_sha256"] = module.compact_record_sha256(record)
        module._validate_compact_record(
            record, request, self.prereg["schemas"]["batch"]
        )
        record["production_path_status"] = "source_seed_mismatch"
        with self.assertRaisesRegex(RuntimeError, "record digest"):
            module._validate_compact_record(
                record, request, self.prereg["schemas"]["batch"]
            )

    def test_rehashed_semantic_response_tamper_fails_closed(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]

        def committed(directory, binary):
            identity = {
                "schema": module.RUN_IDENTITY_SCHEMA,
                "batch_schema": self.prereg["schemas"]["batch"],
                "source_set_sha256": "11" * 32,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "seed_count": 1,
            }
            shards = module.initialize_run_root(directory / "run", identity)
            module.execute_or_resume_shard(
                self.prereg, cell, 1, shards, binary, identity
            )
            return identity, shards, module._shard_paths(shards, cell)

        def rebind(paths, records):
            payload = b"".join(module.record_wire_bytes(record) + b"\n"
                               for record in records)
            semantic_digest = hashlib.sha256()
            for record in records:
                semantic_digest.update(module._response_semantic_bytes(record))
            semantic = semantic_digest.hexdigest()
            paths["records"].write_bytes(payload)
            manifest = json.loads(paths["manifest"].read_bytes())
            manifest["records_sha256"] = hashlib.sha256(payload).hexdigest()
            manifest["records_bytes"] = len(payload)
            manifest["response_semantic_sha256"] = semantic
            manifest_bytes = module.canonical_json_bytes(manifest) + b"\n"
            paths["manifest"].write_bytes(manifest_bytes)
            commit = json.loads(paths["commit"].read_bytes())
            commit["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            commit["records_sha256"] = hashlib.sha256(payload).hexdigest()
            commit["response_semantic_sha256"] = semantic
            paths["commit"].write_bytes(module.canonical_json_bytes(commit) + b"\n")

        cases = {
            "fabricated failed disposition": ("fixed_factor", lambda record:
                record.update({"emitted": False, "failed": True}), True, "emission"),
            "nonfinite comparator": ("fixed_factor", lambda record:
                record["fit"]["ols"].update({"point_estimate": "nan"}), True,
                "comparator"),
            "omitted production receipts": ("production_path", lambda record:
                record.update({"production_receipts": None}), True, "production state"),
            "invalid production provenance": ("production_path", lambda record:
                record.update({"provenance": {"schema": "forged"}}), True,
                "provenance"),
            "unknown production status": ("production_path", lambda record:
                record.update({"production_path_status": "fabricated_fail_closed"}),
                True, "unknown production status"),
            "capture receipt mismatch": ("production_path", lambda record:
                record["production_receipts"].update({"capture_scope_sha256": "ff" * 32}),
                True, "source-correlation provenance"),
            "finite coupled slope forgery": ("both", lambda record: (
                record["fit"].update({
                    "ols_slope": record["fit"]["ols_slope"] + 1000.0,
                }),
                record["fit"]["ols"].update({
                    "point_estimate": record["fit"]["ols"]["point_estimate"] + 1000.0,
                }),
            ), False, "record digest"),
            "finite coupled interval forgery": ("both", lambda record: (
                record["fit"]["ols"]["interval_95"].update({
                    "upper": record["fit"]["ols"]["interval_95"]["upper"] + 1000.0,
                }),
                record["fit"]["ols"].update({
                    "width_95": record["fit"]["ols"]["width_95"] + 1000.0,
                }),
            ), False, "record digest"),
            "paired selected interval forgery": ("both", lambda record: (
                record["fit"]["adjusted_scalar"]["interval_95"].update({
                    "lower": record["fit"]["adjusted_scalar"]["interval_95"]["lower"]
                    - 1.0,
                }),
                record["fit"]["adjusted_scalar"].update({
                    "width_95": record["fit"]["adjusted_scalar"]["width_95"] + 1.0,
                }),
            ), False, "record digest"),
        }
        for label, (execution_path, mutate, rehash_record, pattern) in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                binary = ROOT / "target/release/examples/temporal_covariance_batch"
                identity, shards, paths = committed(root, binary)
                records = [json.loads(line)
                           for line in paths["records"].read_bytes().splitlines()]
                if execution_path == "both":
                    for record in records:
                        mutate(record)
                else:
                    record = records[0 if execution_path == "fixed_factor" else 1]
                    mutate(record)
                if rehash_record:
                    for record in records:
                        record["record_sha256"] = module.compact_record_sha256(record)
                rebind(paths, records)
                with self.assertRaisesRegex(RuntimeError, pattern):
                    module.execute_or_resume_shard(
                        self.prereg, cell, 1,
                        shards, binary, identity,
                    )

    def test_resumable_run_rejects_stale_binary_and_partial_commit(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = write_fake_batch(root, self.prereg["schemas"]["batch"])
            identity = {
                "schema": module.RUN_IDENTITY_SCHEMA,
                "batch_schema": self.prereg["schemas"]["batch"],
                "source_set_sha256": "11" * 32,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "seed_count": 2,
            }
            run_root = root / "run"
            shards = module.initialize_run_root(run_root, identity)
            module.execute_or_resume_shard(
                self.prereg, cell, 2, shards, binary, identity
            )
            stale = dict(identity, binary_sha256="ff" * 32)
            with self.assertRaisesRegex(RuntimeError, "identity is stale"):
                module.initialize_run_root(run_root, stale)
            original_binary = binary.read_bytes()
            binary.write_bytes(original_binary + b"\n# changed after commit\n")
            with self.assertRaisesRegex(RuntimeError, "batch binary identity is stale"):
                module.execute_or_resume_shard(
                    self.prereg, cell, 2,
                    shards, binary, identity,
                )
            binary.write_bytes(original_binary)
            paths = module._shard_paths(shards, cell)
            paths["manifest"].unlink()
            with self.assertRaisesRegex(RuntimeError, "partial or missing"):
                module.execute_or_resume_shard(
                    self.prereg, cell, 2,
                    shards, binary, identity,
                )

    def test_full_24_shard_run_is_bounded_atomic_and_exactly_resumable(self):
        module = load_generator()
        preregistration = json.loads(json.dumps(self.prereg))
        preregistration["outer_seeds_per_supported_cell"] = 1
        preregistration["execution_protocol"].update({
            "seed_requests_per_shard": 1,
            "attempt_records_per_shard": 2,
        })
        preregistration["engine_validation"]["attempt_count"] = 48
        preregistration["engine_validation"]["seed_request_count"] = 24
        preregistration["resource_limits"]["retained_bound_bytes"] = (
            24 * (
                module.MAX_SHARD_RECORD_BYTES
                + module.MAX_MANIFEST_BYTES
                + module.MAX_COMMIT_BYTES
            )
            + module.MAX_COMMIT_BYTES
            + module.MAX_MANIFEST_BYTES
            + module.MAX_COMMIT_BYTES
            + module.MAX_FINAL_RECEIPT_BYTES
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = write_fake_batch(root, preregistration["schemas"]["batch"])
            run_root = root / "run"
            identity = {
                "schema": module.RUN_IDENTITY_SCHEMA,
                "batch_schema": preregistration["schemas"]["batch"],
                "source_set_sha256": "11" * 32,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "binary_bytes": len(binary.read_bytes()),
                "preregistration_sha256": hashlib.sha256(
                    module.canonical_json_bytes(preregistration)
                ).hexdigest(),
                "seed_count": 1,
            }
            resource_evidence = {
                "candidate_resource_receipt_sha256": "12" * 32,
                "method_selection_receipt_sha256": "34" * 32,
                "resource_receipt_sha256": "56" * 32,
                "batch_binary": {
                    "sha256": identity["binary_sha256"],
                    "bytes": identity["binary_bytes"],
                },
                "benchmark_binary": {
                    "sha256": "78" * 32,
                    "bytes": 1234,
                },
            }
            with (
                mock.patch.object(module, "producer_identity", return_value=identity),
                mock.patch.object(
                    module,
                    "validate_release_resource_evidence",
                    return_value=resource_evidence,
                ) as validate_resource,
            ):
                first = module.run(
                    preregistration,
                    1,
                    None,
                    run_root=run_root,
                    binary=binary,
                    resource_evidence_directory=root,
                )
            validate_resource.assert_called_once_with(preregistration, root)
            run_identity = json.loads((run_root / "run_identity.json").read_bytes())
            self.assertEqual(
                run_identity["candidate_resource_receipt_sha256"], "12" * 32
            )
            self.assertEqual(
                run_identity["method_selection_receipt_sha256"], "34" * 32
            )
            self.assertEqual(run_identity["resource_receipt_sha256"], "56" * 32)
            self.assertEqual(
                run_identity["resource_benchmark_binary_sha256"], "78" * 32
            )
            self.assertEqual(first["processed_attempt_record_count"], 48)
            self.assertTrue(first["exact_seed_denominator_complete"])
            self.assertFalse(first["engine_validation_eligible"])
            self.assertEqual(first["records"], [])
            self.assertEqual(len(list((run_root / "shards").iterdir())), 72)
            self.assertTrue((run_root / "run_manifest.json").is_file())
            self.assertTrue((run_root / "run_commit.json").is_file())
            self.assertLessEqual(
                first["resource"]["result_artifact_bytes"],
                preregistration["resource_limits"]["retained_bound_bytes"],
            )
            self.assertTrue(first["resource_gates"]["retained_bound"])
            self.assertEqual(
                set(first["resource_gates"]),
                {
                    "rss",
                    "artifact_size",
                    "bound_resource_receipt",
                    "retained_bound",
                },
            )
            with (
                mock.patch.object(module, "producer_identity", return_value=identity),
                mock.patch.object(
                    module,
                    "validate_release_resource_evidence",
                    return_value=resource_evidence,
                ),
            ):
                second = module.run(
                    preregistration,
                    1,
                    None,
                    run_root=run_root,
                    binary=binary,
                    resource_evidence_directory=root,
                )
            self.assertEqual(second, first)
            self.assertEqual((root / "fake_temporal_batch.count").read_text(), "24")
            swapped = dict(resource_evidence, resource_receipt_sha256="9a" * 32)
            with (
                mock.patch.object(module, "producer_identity", return_value=identity),
                mock.patch.object(
                    module,
                    "validate_release_resource_evidence",
                    return_value=swapped,
                ),
                self.assertRaisesRegex(RuntimeError, "identity is stale"),
            ):
                module.run(
                    preregistration,
                    1,
                    None,
                    run_root=run_root,
                    binary=binary,
                    resource_evidence_directory=root,
                )

    def test_retained_bound_matches_exact_24_shard_composition(self):
        module = load_generator()
        expected = 24 * (
            module.MAX_SHARD_RECORD_BYTES
            + module.MAX_MANIFEST_BYTES
            + module.MAX_COMMIT_BYTES
        ) + (
            module.MAX_COMMIT_BYTES
            + module.MAX_MANIFEST_BYTES
            + module.MAX_COMMIT_BYTES
            + module.MAX_FINAL_RECEIPT_BYTES
        )
        self.assertEqual(
            self.prereg["resource_limits"]["retained_bound_bytes"], expected
        )
        self.assertEqual(
            self.prereg["resource_limits"]["retained_bound_formula"],
            "24*(16777216+1048576+65536)+65536+1048576+65536+1048576",
        )
        self.assertLessEqual(
            expected, self.prereg["resource_limits"]["artifact_size_limit_bytes"]
        )

    def test_line_and_atomic_output_caps_fail_closed(self):
        module = load_generator()
        with self.assertRaisesRegex(RuntimeError, "line exceeds"):
            module._read_bounded_line(
                io.BytesIO(b"x" * (module.MAX_RESPONSE_LINE_BYTES + 1)),
                module.MAX_RESPONSE_LINE_BYTES,
            )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            bounded = root / "bounded.json"
            bounded.write_bytes(b"x" * module.MAX_COMMIT_BYTES)
            self.assertEqual(
                len(module._read_bounded_regular(bounded, module.MAX_COMMIT_BYTES)),
                module.MAX_COMMIT_BYTES,
            )
            bounded.write_bytes(b"x" * (module.MAX_COMMIT_BYTES + 1))
            with self.assertRaisesRegex(RuntimeError, "exceeds its retained byte cap"):
                module._read_bounded_regular(bounded, module.MAX_COMMIT_BYTES)
            link = root / "bounded-link.json"
            link.symlink_to(bounded)
            with self.assertRaisesRegex(RuntimeError, "not an openable regular file"):
                module._read_bounded_regular(link, module.MAX_COMMIT_BYTES)
            output = root / "receipt.json"
            module.atomic_write_no_replace(output, b"first\n")
            with self.assertRaises(FileExistsError):
                module.atomic_write_no_replace(output, b"second\n")
            self.assertEqual(output.read_bytes(), b"first\n")

    def test_coverage_tolerances_are_explicit(self):
        self.assertEqual(self.prereg["coverage_tolerances"], {
            "0.68": 0.03, "0.90": 0.02, "0.95": 0.015
        })
        self.assertEqual(self.prereg["standardized_bias_tolerance"], 0.05)
        self.assertEqual(self.prereg["minimum_successful_emission_fraction"], 0.99)

    def test_no_outcome_placeholders_remain(self):
        serialized = json.dumps(self.prereg)
        for forbidden in ("freeze_after_performance_dry_run", "computed_before_outcomes", "record_at_run"):
            self.assertNotIn(forbidden, serialized)

    def test_frozen_source_hashes_remain_immutable_after_estimator_change(self):
        paths = {
            "generator_sha256": ROOT / "validation/temporal_covariance_simulation.py",
            "batch_source_sha256": ROOT / "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs",
            "estimator_source_sha256": ROOT / "crates/dolphin-timeseries/src/temporal_covariance.rs",
        }
        self.assertEqual(
            self.prereg["file_hashes"],
            {
                "generator_sha256": (
                    "6684130b2b8f596bef67de70ed39f00b8cb65cb1023beb169307f660834f7d56"
                ),
                "batch_source_sha256": (
                    "448afa0813edf06b1ee435c724ab11de16f64a6fee56fec41b08aaea742ee937"
                ),
                "estimator_source_sha256": (
                    "f6c3713b72f4a5f2067e63153d4c7ebdd790cbd53f15d086442a66fe7adff206"
                ),
            },
        )
        for identity in ("generator_sha256", "batch_source_sha256"):
            self.assertEqual(
                hashlib.sha256(paths[identity].read_bytes()).hexdigest(),
                self.prereg["file_hashes"][identity],
            )
        self.assertNotEqual(
            hashlib.sha256(paths["estimator_source_sha256"].read_bytes()).hexdigest(),
            self.prereg["file_hashes"]["estimator_source_sha256"],
        )

    def test_canonical_source_closure_detects_pr84_and_temporal_mutations(self):
        module = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            files = set(module.FROZEN_SOURCE_SET_FILES)
            files.add("crates/dolphin-workflows/src/sequential_covariance.rs")
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                content = f"fixture:{relative}\n"
                if relative == "validation/temporal_covariance_simulation.py":
                    content += 'FROZEN_SOURCE_SET_SHA256 = "fixture"\n'
                path.write_text(content)
            baseline = module.canonical_source_set_sha256(root)
            for relative in (
                "crates/dolphin-workflows/src/sequential_covariance.rs",
                "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs",
                "validation/temporal_covariance_simulation.py",
                "crates/dolphin-timeseries/src/temporal_covariance.rs",
            ):
                with self.subTest(relative=relative):
                    path = root / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b"mutation\n")
                    self.assertNotEqual(module.canonical_source_set_sha256(root), baseline)
                    path.write_bytes(original)

    def test_producer_identity_rejects_source_drift_and_noncanonical_binary(self):
        module = load_generator()
        release = ROOT / "target/release/examples/temporal_covariance_batch"
        with self.assertRaisesRegex(RuntimeError, "source hashes do not match"):
            module.producer_identity(self.prereg, release)
        source_hashes = self.prereg["file_hashes"]
        with (
            mock.patch.object(
                module,
                "sha256_file",
                side_effect=[
                    (source_hashes["generator_sha256"], 0),
                    (source_hashes["batch_source_sha256"], 0),
                    (source_hashes["estimator_source_sha256"], 0),
                    ("ab" * 32, 1234567),
                ],
            ),
            mock.patch.object(
                module,
                "canonical_source_set_sha256",
                return_value=self.prereg["producer_identity"]["source_set_sha256"],
            ),
        ):
            identity = module.producer_identity(self.prereg, release)
        self.assertEqual(
            identity["source_set_sha256"],
            self.prereg["producer_identity"]["source_set_sha256"],
        )
        self.assertNotIn("binary_sha256", self.prereg["producer_identity"])
        self.assertNotIn("binary_bytes", self.prereg["producer_identity"])
        self.assertEqual(identity["binary_sha256"], "ab" * 32)
        self.assertEqual(identity["binary_bytes"], 1234567)
        with tempfile.TemporaryDirectory() as directory:
            copied = pathlib.Path(directory) / "temporal_covariance_batch"
            copied.write_bytes(release.read_bytes())
            copied.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError, "exact prebuilt release executable"):
                module.producer_identity(self.prereg, copied)


if __name__ == "__main__":
    unittest.main()
