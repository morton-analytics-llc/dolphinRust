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


def write_fake_batch(directory: pathlib.Path, batch_schema: str) -> pathlib.Path:
    path = directory / "fake_temporal_batch.py"
    counter = directory / "fake_temporal_batch.count"
    path.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import json
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
        def comparator(status):
            return {{
                "point_estimate": None, "standard_error_diagnostic": None,
                "interval_68": None, "interval_90": None, "interval_95": None,
                "width_68": None, "width_90": None, "width_95": None,
                "status": status, "attempted_replicates": 0,
                "successful_replicates": 0,
            }}
        def failed_fit():
            status = "InsufficientDates"
            diagnostic = comparator(status)
            return {{
                "status": status, "ols_slope": None, "oracle_gls_slope": None,
                "plugin_gls_slope": None, "adjusted_profile_slope": None,
                "bootstrap_slope": None, "bootstrap_interval": None,
                "fitted_rho": None, "fitted_process_variance": None,
                "raw_correlation": {{"rho": None, "pair_count": 0,
                    "minimum_gap_days": None, "median_gap_days": None,
                    "maximum_gap_days": None}},
                "valid_date_count": 0, "rank": 0, "degrees_of_freedom": 0,
                "covariance_condition_number": None,
                "ols": diagnostic, "oracle_gls": diagnostic,
                "conditional_wls": diagnostic, "scalar_effective_n": diagnostic,
                "plugin_gls": diagnostic, "adjusted_scalar": diagnostic,
                "adjusted_profile": diagnostic,
                "complete_refit_bootstrap": diagnostic,
                "bootstrap_attempts": 0, "bootstrap_successes": 0,
            }}
        for line in sys.stdin:
            request = json.loads(line)
            fixed = request["execution_path"] == "fixed_factor"
            response = {{
                "schema": {batch_schema!r},
                "execution_path": request["execution_path"],
                "cell_id": request["cell_id"],
                "cell_index": request["cell_index"],
                "outer_seed_index": request["outer_seed_index"],
                "seed_sha256": request["seed_sha256"],
                "seed": request["seed"],
                "fixed_factor_status": "InsufficientDates" if fixed else None,
                "production_path_status": None if fixed else "raw_complex_invalid",
                "comparator_methods": methods,
                "attempted": True,
                "emitted": False,
                "failed": True,
                "fit": failed_fit() if fixed else None,
                "provenance": None,
                "production_receipts": None,
                "resource": {{
                    "wall_micros": 1,
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
        self.assertEqual(self.prereg["outer_seeds_per_supported_cell"], 1050)
        self.assertEqual(self.prereg["execution_protocol"]["shard_count"], 48)
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
            self.assertEqual(production["fit"]["valid_date_count"], 12)
            self.assertEqual(len(fixed["comparator_methods"]), 8)
            self.assertIsNone(production["provenance"])
            self.assertIsNotNone(production["production_receipts"])
            self.assertEqual(
                receipt["scores"]["schema"], self.prereg["schemas"]["scorer"]
            )
            self.assertEqual(receipt["scores"]["methods"]["ols"]["scored"], 2)
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
        self.assertEqual(production["outer_coverage_dgp"], module.OUTER_COVERAGE_DGP)
        self.assertEqual(
            production["conditional_covariance_oracle"],
            module.CONDITIONAL_COVARIANCE_ORACLE,
        )

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
        fixed = module.request_for(cell, 0, self.prereg, "production_path")
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
            module.request_for(cell, seed, self.prereg, "production_path")
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
            request = module.request_for(cell, 0, self.prereg, "production_path")
            request["retain_dense_evidence"] = True
            request["conditional_oracle_replicates"] = replicates
            request["options"]["bootstrap_replicates"] = 0
            request["options"]["bootstrap_minimum_successes"] = 0
            requests.append(request)
        for request, record in zip(requests, run_production_batch(requests)):
            receipts = record["production_receipts"]
            self.assertEqual(
                receipts["outer_coverage_dgp"], "physical_raw_space_v1"
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
            ("source_correlation_model", "caller_claimed_model"),
            ("source_correlation_distance_scale_pixels", 3.0),
            ("outer_coverage_dgp", "caller_claimed_dgp"),
            ("conditional_covariance_oracle", "caller_claimed_oracle"),
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
        request = module.request_for(cell, 99, self.prereg, "production_path")
        request["conditional_oracle_replicates"] = 16_385
        record = run_production_batch(request)
        self.assertEqual(record["production_path_status"], "production_contract_mismatch")
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
        self.assertEqual(receipts["source_correlation_model"],
                         module.SOURCE_CORRELATION_MODEL)
        self.assertEqual(receipts["source_correlation_distance_scale_pixels"],
                         module.SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS)
        self.assertGreater(receipts["source_correlation_support_union_count"], 0)
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
            paths = module._shard_paths(shards, cell, "fixed_factor")
            partial = paths["records"].with_name(paths["records"].name + ".partial")
            partial.write_bytes(b"owned interrupted bytes")
            manifest, reused = module.execute_or_resume_shard(
                self.prereg, cell, "fixed_factor", 2, shards, binary, identity
            )
            self.assertFalse(reused)
            self.assertEqual(manifest["attempted"], 2)
            self.assertFalse(partial.exists())
            committed = paths["records"].read_bytes().splitlines()
            self.assertEqual(len(committed), 2)
            self.assertTrue(all(b"fixed_l2_difference_covariance" not in line
                                for line in committed))
            resumed, reused = module.execute_or_resume_shard(
                self.prereg, cell, "fixed_factor", 2, shards, binary, identity
            )
            self.assertTrue(reused)
            self.assertEqual(resumed, manifest)
            self.assertEqual((root / "fake_temporal_batch.count").read_text(), "2")

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
                self.prereg, cell, "fixed_factor", 2, shards, binary, identity
            )
            return binary, identity, shards, module._shard_paths(
                shards, cell, "fixed_factor"
            )

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
                        self.prereg, cell, "fixed_factor", 2,
                        shards, binary, identity,
                    )
        for label, mutate, pattern in (
            ("missing", lambda lines: lines[:1], "missing, duplicated, or reordered"),
            ("duplicate", lambda lines: [lines[0], lines[0]], "attempt identity"),
            ("reorder", lambda lines: list(reversed(lines)), "attempt identity"),
        ):
            with self.subTest(label):
                with tempfile.TemporaryDirectory() as directory:
                    binary, identity, shards, paths = committed(pathlib.Path(directory))
                    lines = paths["records"].read_bytes().splitlines()
                    rebind(paths, mutate(lines))
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        module.execute_or_resume_shard(
                            self.prereg, cell, "fixed_factor", 2,
                            shards, binary, identity,
                        )

    def test_rehashed_semantic_response_tamper_fails_closed(self):
        module = load_generator()
        cell = module.cells(self.prereg)[0]

        def committed(directory, execution_path, binary):
            identity = {
                "schema": module.RUN_IDENTITY_SCHEMA,
                "batch_schema": self.prereg["schemas"]["batch"],
                "source_set_sha256": "11" * 32,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "seed_count": 1,
            }
            shards = module.initialize_run_root(directory / "run", identity)
            module.execute_or_resume_shard(
                self.prereg, cell, execution_path, 1, shards, binary, identity
            )
            return identity, shards, module._shard_paths(shards, cell, execution_path)

        def rebind(paths, record):
            payload = module.canonical_json_bytes(record) + b"\n"
            semantic = hashlib.sha256(module._response_semantic_bytes(record)).hexdigest()
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
            "fabricated evaluated fit": ("fixed_factor", lambda record: (
                record.update({"fixed_factor_status": "Evaluated", "emitted": True,
                               "failed": False}),
                record["fit"].update({"status": "Evaluated"}),
            ), "batch returned"),
            "nonfinite comparator": ("fixed_factor", lambda record:
                record["fit"]["ols"].update({"point_estimate": "nan"}), "batch returned"),
            "omitted production receipts": ("production_path", lambda record:
                record.update({"production_receipts": None}), "batch returned"),
            "invalid production provenance": ("production_path", lambda record:
                record.update({"provenance": {"schema": "forged"}}), "batch returned"),
            "unknown production status": ("production_path", lambda record:
                record.update({"production_path_status": "fabricated_fail_closed"}),
                "batch returned"),
            "capture receipt mismatch": ("production_path", lambda record:
                record["production_receipts"].update({"capture_scope_sha256": "ff" * 32}),
                "batch returned"),
            "finite coupled slope forgery": ("fixed_factor", lambda record: (
                record["fit"].update({
                    "ols_slope": record["fit"]["ols_slope"] + 1000.0,
                }),
                record["fit"]["ols"].update({
                    "point_estimate": record["fit"]["ols"]["point_estimate"] + 1000.0,
                }),
            ), "response semantics inconsistent"),
            "finite coupled interval forgery": ("fixed_factor", lambda record: (
                record["fit"]["ols"]["interval_95"].update({
                    "upper": record["fit"]["ols"]["interval_95"]["upper"] + 1000.0,
                }),
                record["fit"]["ols"].update({
                    "width_95": record["fit"]["ols"]["width_95"] + 1000.0,
                }),
            ), "response semantics inconsistent"),
        }
        for label, (execution_path, mutate, pattern) in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                binary = ROOT / "target/release/examples/temporal_covariance_batch"
                identity, shards, paths = committed(root, execution_path, binary)
                record = json.loads(paths["records"].read_bytes())
                mutate(record)
                rebind(paths, record)
                with self.assertRaisesRegex(RuntimeError, pattern):
                    module.execute_or_resume_shard(
                        self.prereg, cell, execution_path, 1,
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
                self.prereg, cell, "fixed_factor", 2, shards, binary, identity
            )
            stale = dict(identity, binary_sha256="ff" * 32)
            with self.assertRaisesRegex(RuntimeError, "identity is stale"):
                module.initialize_run_root(run_root, stale)
            original_binary = binary.read_bytes()
            binary.write_bytes(original_binary + b"\n# changed after commit\n")
            with self.assertRaisesRegex(RuntimeError, "batch binary identity is stale"):
                module.execute_or_resume_shard(
                    self.prereg, cell, "fixed_factor", 2,
                    shards, binary, identity,
                )
            binary.write_bytes(original_binary)
            paths = module._shard_paths(shards, cell, "fixed_factor")
            paths["manifest"].unlink()
            with self.assertRaisesRegex(RuntimeError, "partial or missing"):
                module.execute_or_resume_shard(
                    self.prereg, cell, "fixed_factor", 2,
                    shards, binary, identity,
                )

    def test_full_48_shard_run_is_bounded_atomic_and_exactly_resumable(self):
        module = load_generator()
        preregistration = json.loads(json.dumps(self.prereg))
        preregistration["outer_seeds_per_supported_cell"] = 1
        preregistration["file_hashes"] = {
            "generator_sha256": hashlib.sha256(
                (ROOT / "validation/temporal_covariance_simulation.py").read_bytes()
            ).hexdigest(),
            "batch_source_sha256": hashlib.sha256(
                (ROOT / "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs")
                .read_bytes()
            ).hexdigest(),
            "estimator_source_sha256": hashlib.sha256(
                (ROOT / "crates/dolphin-timeseries/src/temporal_covariance.rs").read_bytes()
            ).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = write_fake_batch(root, self.prereg["schemas"]["batch"])
            run_root = root / "run"
            identity = {
                "schema": module.RUN_IDENTITY_SCHEMA,
                "batch_schema": self.prereg["schemas"]["batch"],
                "source_set_sha256": "11" * 32,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "seed_count": 1,
            }
            with mock.patch.object(module, "producer_identity", return_value=identity):
                first = module.run(
                    preregistration, 1, None, run_root=run_root, binary=binary
                )
            self.assertEqual(first["batch_attempted_cells"], 48)
            self.assertTrue(first["exact_seed_denominator_complete"])
            self.assertFalse(first["promotion_eligible"])
            self.assertEqual(first["records"], [])
            self.assertEqual(len(list((run_root / "shards").iterdir())), 144)
            self.assertLessEqual(
                first["resource"]["result_artifact_bytes"],
                preregistration["resource_limits"]["retained_bound_bytes"],
            )
            self.assertTrue(first["resource_gates"]["retained_bound"])
            with mock.patch.object(module, "producer_identity", return_value=identity):
                second = module.run(
                    preregistration, 1, None, run_root=run_root, binary=binary
                )
            self.assertEqual(second, first)
            self.assertEqual((root / "fake_temporal_batch.count").read_text(), "96")

    def test_retained_bound_matches_exact_48_shard_composition(self):
        module = load_generator()
        expected = 48 * (
            module.MAX_SHARD_RECORD_BYTES
            + module.MAX_MANIFEST_BYTES
            + module.MAX_COMMIT_BYTES
        ) + module.MAX_COMMIT_BYTES + module.MAX_FINAL_RECEIPT_BYTES
        self.assertEqual(
            self.prereg["resource_limits"]["retained_bound_bytes"], expected
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

    def test_producer_identity_requires_frozen_release_binary_and_source_set(self):
        module = load_generator()
        release = ROOT / "target/release/examples/temporal_covariance_batch"
        identity = module.producer_identity(self.prereg, release)
        self.assertEqual(
            identity["source_set_sha256"],
            self.prereg["producer_identity"]["source_set_sha256"],
        )
        self.assertNotIn("binary_sha256", self.prereg["producer_identity"])
        self.assertNotIn("binary_bytes", self.prereg["producer_identity"])
        with mock.patch.object(
                module, "_runtime_binary_identity", return_value=("ab" * 32, 1234567)):
            alternate = module.producer_identity(self.prereg, release)
        self.assertEqual(alternate["binary_sha256"], "ab" * 32)
        self.assertEqual(alternate["binary_bytes"], 1234567)
        self.assertEqual(alternate["source_set_sha256"], identity["source_set_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            copied = pathlib.Path(directory) / "temporal_covariance_batch"
            copied.write_bytes(release.read_bytes())
            copied.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError, "exact prebuilt release executable"):
                module.producer_identity(self.prereg, copied)


if __name__ == "__main__":
    unittest.main()
