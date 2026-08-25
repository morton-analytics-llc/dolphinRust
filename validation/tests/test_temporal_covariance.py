"""Schema tests for the frozen #53 preregistration.

These tests validate the contract only; they intentionally do not run the
5,000-seed experiment or acquire external GNSS data.
"""

import json
import importlib.util
import hashlib
import math
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).parents[2]


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
        input="".join(json.dumps(value) + "\n" for value in requests),
        text=True,
        capture_output=True,
        check=True,
    )
    records = [json.loads(line) for line in result.stdout.splitlines()]
    return records if isinstance(request, list) else records[0]


class TemporalCovariancePreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "validation/temporal_covariance_preregistration.json").open() as handle:
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
            self.prereg["promotion_status"],
            "blocked_pending_synthetic_field_review_and_manifest",
        )
        self.assertFalse(self.prereg["corrected_inferential_sigma_emission"])
        self.assertTrue(self.prereg["external_holdout_required"])

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
        self.assertEqual(module.cell_hash(unsupported), self.prereg["unsupported_cell_sha256"])

    def test_compact_simulation_driver_is_deterministic_and_nonpromoting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "receipt.json"
            command = [
                "python3",
                str(ROOT / "validation/temporal_covariance_simulation.py"),
                "--prereg",
                str(ROOT / "validation/temporal_covariance_preregistration.json"),
                "--output",
                str(output),
                "--seeds",
                "1",
                "--limit",
                "2",
            ]
            subprocess.run(command, check=True)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["attempted_cells"], 48)
            self.assertEqual(receipt["batch_attempted_cells"], 2)
            self.assertEqual(receipt["emitted_cells"], 0)
            self.assertEqual(receipt["failed_cells"], 2)
            self.assertEqual(receipt["skipped_contract_cells"], 46)
            self.assertFalse(receipt["corrected_inferential_sigma_emission"])
            self.assertEqual(receipt["pre_outcome_status"], "pre_outcome_frozen")
            fixed, production = receipt["records"]
            self.assertEqual(fixed["execution_path"], "fixed_factor")
            self.assertNotEqual(fixed["fixed_factor_status"], "Evaluated")
            self.assertEqual(production["execution_path"], "production_path")
            self.assertEqual(production["production_path_status"], "estimator_failed")
            self.assertEqual(fixed["fit"]["valid_date_count"], 12)
            self.assertEqual(production["fit"]["status"], "RhoUpperBoundary")
            self.assertEqual(production["fit"]["valid_date_count"], 0)
            self.assertEqual(len(fixed["comparator_methods"]), 8)
            self.assertIsNone(production["provenance"])
            self.assertIsNotNone(production["production_receipts"])
            self.assertEqual(receipt["scores"]["schema"], "coverage_bias_interval_score/3")
            self.assertEqual(receipt["scores"]["methods"]["ols"]["scored"], 1)
            self.assertEqual(len(receipt["scores"]["cell_summaries"]), 48)
            self.assertFalse(receipt["exact_seed_denominator_complete"])
            self.assertIn("resource", fixed)
            self.assertFalse(receipt["execution_complete"])
            self.assertFalse(receipt["promotion_eligible"])
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
        request = module.request_for(cell, 1, self.prereg, "fixed_factor")
        retained = sum(value is not None for value in request["fixed_factor"]["observations"]) - 1
        self.assertEqual(retained, cell["date_count"])
        production = module.request_for(cell, 1, self.prereg, "production_path")["production_path"]
        self.assertNotIn("issue52_target_factor", production)
        self.assertNotIn("issue54_difference_factor", production)
        self.assertNotIn("issue52_seed", production)
        self.assertNotIn("issue54_seed", production)
        self.assertEqual(production["native_shape"], [1, 7])
        self.assertEqual(len(production["raw_complex_stack"]), len(module.days_for(cell)))
        self.assertTrue(all(len(row) == 7 for row in production["raw_complex_stack"]))
        self.assertEqual(len(production["capture_scope_sha256"]), 64)

    def test_production_batch_uses_actual_capture_replay_and_fixed_l2(self):
        source = (ROOT / "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs").read_text()
        self.assertNotIn("InfluenceDag", source)
        self.assertNotIn("build_raw_dag", source)
        self.assertIn("run_sequential_with_covariance_capture_and_source_factors", source)
        self.assertIn("replay_global_reference_difference_covariance_from_provider_bundle", source)
        self.assertIn("propagate_fixed_l2_difference_covariance", source)

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

    def test_production_request_uses_raw_proper_complex_temporal_covariance(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        outer_seed = 17
        request = module.request_for(cell, outer_seed, self.prereg, "production_path")
        production = request["production_path"]
        seed = request["seed"]
        days = request["days"]
        state, ar_path = module.stationary_ar_path(days, cell["rho_at_12_days"], seed)
        del state
        diagonal = []
        for index in range(len(days)):
            variance_scale = 1.0 if index < len(days) // 2 else cell["variance_ratio"]
            diagonal.append(0.01 * (variance_scale + cell["reference_contribution_ratio"]))
        geometric_mean = math.exp(
            sum(math.log(value) for value in diagonal[1:]) / (len(diagonal) - 1)
        )
        reference_column = production["reference_pixel"][1]
        correlation = module.production_support_correlation(1, reference_column, 7)
        common_speckle = [
            complex(*module.proper_complex_speckle(cell["cell_index"], outer_seed, column))
            for column in range(7)
        ]
        for date_index, (day, row) in enumerate(zip(days, production["raw_complex_stack"])):
            shape = math.sqrt(diagonal[date_index] / geometric_mean)
            carrier_value = 0.0 if date_index == 0 else (
                0.01 * day + math.sqrt(0.04) * shape * ar_path[date_index]
            )
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

    def test_actual_evd_fixed_l2_covariance_matches_raw_ensemble_moment(self):
        module = load_generator()
        base_cell = module.cells(self.prereg)[0]
        seed_count = 64
        contexts = {
            "near_exact": (0.5, 1),
            "mid_exact": (0.2, 2),
            "far_exact": (0.0, 4),
        }
        requests = []
        for context, (jaccard, distance) in contexts.items():
            cell = dict(
                base_cell,
                reference_context=context,
                overlap_fraction=jaccard,
                distance_pixels=distance,
            )
            for outer_seed in range(seed_count):
                request = module.request_for(cell, outer_seed, self.prereg, "production_path")
                request["options"]["bootstrap_replicates"] = 0
                request["options"]["bootstrap_minimum_successes"] = 0
                requests.append(request)
        records = run_production_batch(requests)
        by_context = {context: [] for context in contexts}
        for context_index, context in enumerate(contexts):
            start = context_index * seed_count
            for request, record in zip(
                    requests[start:start + seed_count], records[start:start + seed_count]):
                self.assertIsNotNone(record["production_receipts"])
                receipts = record["production_receipts"]
                error = [
                    linked - truth
                    for linked, truth in zip(
                        receipts["linked_difference_history"],
                        receipts["carrier_difference_history"],
                    )
                ]
                by_context[context].append((
                    receipts["fixed_l2_difference_covariance"], error
                ))
        for context, realizations in by_context.items():
            self.assertEqual(len(realizations), seed_count, context)
            covariance_size = len(realizations[0][0])
            error_means = [
                sum(value[1][date] for value in realizations) / seed_count
                for date in range(covariance_size)
            ]
            for left in range(covariance_size):
                for right in range(covariance_size):
                    predicted = [value[0][left][right] for value in realizations]
                    predicted_mean = sum(predicted) / seed_count
                    if left == 0 or right == 0:
                        self.assertEqual(predicted_mean, 0.0)
                        continue
                    empirical_covariance = sum(
                        (value[1][left] - error_means[left])
                        * (value[1][right] - error_means[right])
                        for value in realizations
                    ) / (seed_count - 1)
                    left_variance = sum(
                        (value[1][left] - error_means[left]) ** 2
                        for value in realizations
                    ) / (seed_count - 1)
                    right_variance = sum(
                        (value[1][right] - error_means[right]) ** 2
                        for value in realizations
                    ) / (seed_count - 1)
                    predicted_variance = sum(
                        (value - predicted_mean) ** 2 for value in predicted
                    ) / (seed_count - 1)
                    standard_error = math.sqrt(
                        (left_variance * right_variance + empirical_covariance ** 2)
                        / (seed_count - 1)
                        + predicted_variance / seed_count
                    )
                    self.assertLessEqual(
                        abs(empirical_covariance - predicted_mean),
                        4.0 * standard_error + 1e-12,
                        (
                            context,
                            left,
                            right,
                            empirical_covariance,
                            predicted_mean,
                            4.0 * standard_error,
                        ),
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
        request = module.request_for(cell, 99, self.prereg, "production_path")
        request["production_path"]["source_seed"] = 100
        result = subprocess.run(
            ["cargo", "run", "--release", "-p", "dolphin-timeseries", "--example",
             "temporal_covariance_batch"],
            cwd=ROOT,
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        record = json.loads(result.stdout)
        self.assertEqual(record["production_path_status"], "source_seed_mismatch")
        self.assertFalse(record["emitted"])
        self.assertIsNone(record["fit"])

    def test_production_path_scope_mismatch_fails_closed(self):
        path = ROOT / "validation/temporal_covariance_simulation.py"
        spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cell = module.cells(self.prereg)[0]
        request = module.request_for(cell, 99, self.prereg, "production_path")
        request["production_path"]["reference"]["distance_pixels"] += 1.0
        result = subprocess.run(
            ["cargo", "run", "--release", "-p", "dolphin-timeseries", "--example",
             "temporal_covariance_batch"],
            cwd=ROOT,
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        record = json.loads(result.stdout)
        self.assertEqual(record["production_path_status"], "capture_scope_mismatch")
        self.assertFalse(record["emitted"])
        self.assertIsNone(record["fit"])

    def test_production_path_requires_frozen_method_and_synthetic_scope(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        mutations = [
            ("selected_method", "plugin_gls"),
            ("scope", "field_validation"),
            ("effective_looks_model", "caller_claimed_model"),
            ("effective_looks_distance_scale_pixels", 3.0),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                request = module.request_for(cell, 99, self.prereg, "production_path")
                request["production_path"][field] = value
                record = run_production_batch(request)
                self.assertEqual(record["production_path_status"],
                                 "production_contract_mismatch")
                self.assertFalse(record["emitted"])
                self.assertIsNone(record["fit"])

    def test_production_path_rejects_self_consistent_claimed_geometry(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        request = module.request_for(cell, 99, self.prereg, "production_path")
        request["production_path"]["reference"]["overlap_fraction"] = 0.25
        request["production_path"]["capture_scope_sha256"] = module.capture_scope_sha256(request)
        record = run_production_batch(request)
        self.assertEqual(record["production_path_status"], "reference_context_mismatch")
        self.assertFalse(record["emitted"])
        self.assertIsNone(record["fit"])

    def test_production_path_captures_replays_and_binds_actual_receipts(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]
        request = module.request_for(cell, 99, self.prereg, "production_path")
        request["options"]["bootstrap_replicates"] = 0
        request["options"]["bootstrap_minimum_successes"] = 0
        record = run_production_batch(request)
        self.assertEqual(record["production_path_status"], "estimator_failed")
        self.assertEqual(record["fit"]["status"], "RhoUpperBoundary")
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
        self.assertEqual(receipts["effective_looks_model"],
                         module.EFFECTIVE_LOOKS_MODEL)
        self.assertEqual(receipts["effective_looks_distance_scale_pixels"],
                         module.EFFECTIVE_LOOKS_DISTANCE_SCALE_PIXELS)
        self.assertGreater(receipts["effective_looks_fraction"], 0.0)
        self.assertLessEqual(receipts["effective_looks_fraction"], 1.0)

        drifted = json.loads(json.dumps(request))
        drifted["production_path"]["raw_complex_stack"][0][0][0] *= 1.0001
        drifted_record = run_production_batch(drifted)
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
            "execution_path": request["execution_path"],
            "outer_seed_index": 1,
            "seed": request["seed"],
            "seed_sha256": request["seed_sha256"],
            "fit": None,
        }
        with self.assertRaisesRegex(RuntimeError, "duplicate, missing, or reordered"):
            scorer.update(stale)

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

    def test_frozen_source_hashes_match_files(self):
        paths = {
            "generator_sha256": ROOT / "validation/temporal_covariance_simulation.py",
            "batch_source_sha256": ROOT / "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs",
            "estimator_source_sha256": ROOT / "crates/dolphin-timeseries/src/temporal_covariance.rs",
        }
        for identity, path in paths.items():
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                self.prereg["file_hashes"][identity],
            )


if __name__ == "__main__":
    unittest.main()
