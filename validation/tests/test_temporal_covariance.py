"""Schema tests for the frozen #53 preregistration.

These tests validate the contract only; they intentionally do not run the
5,000-seed experiment or acquire external GNSS data.
"""

import json
import importlib.util
import hashlib
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).parents[2]


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
        self.assertEqual(self.prereg["cell_count_without_outer_seeds"], 4032)
        self.assertEqual(self.prereg["cell_count_by_execution_path"], {
            "fixed_factor": 4032, "production_path": 4032
        })
        self.assertEqual(
            self.prereg["supported_cell_sha256"],
            "99d37d61fe76033f1caeaf0ae858a7ac6f3de1ea918c00e50267fbc3ab4db3fe",
        )
        self.assertEqual(self.prereg["global_seed"], 5447718)
        self.assertEqual(
            self.prereg["attempt_denominator"],
            "all_attempted_outer_seeds_including_fit_failures",
        )
        self.assertEqual(self.prereg["bootstrap"]["interval_levels"], [0.68, 0.9, 0.95])
        self.assertEqual(self.prereg["bootstrap"]["count"], 200)
        self.assertEqual(self.prereg["bootstrap"]["minimum_successes"], 180)

    def test_supported_cell_identities_match_frozen_hash(self):
        path = ROOT / "validation/temporal_covariance_simulation.py"
        spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frozen = module.cells(self.prereg)
        self.assertEqual(len(frozen), 4032)
        self.assertEqual(module.cell_hash(frozen), self.prereg["supported_cell_sha256"])

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
            self.assertEqual(receipt["attempted_cells"], 8064)
            self.assertEqual(receipt["batch_attempted_cells"], 2)
            self.assertEqual(receipt["emitted_cells"], 2)
            self.assertEqual(receipt["failed_cells"], 0)
            self.assertEqual(receipt["skipped_contract_cells"], 8062)
            self.assertFalse(receipt["corrected_inferential_sigma_emission"])
            self.assertEqual(receipt["pre_outcome_status"], "pre_outcome_frozen")
            fixed, production = receipt["records"]
            self.assertEqual(fixed["execution_path"], "fixed_factor")
            self.assertEqual(fixed["fixed_factor_status"], "Evaluated")
            self.assertEqual(production["execution_path"], "production_path")
            self.assertEqual(production["production_path_status"], "evaluated")
            self.assertEqual(fixed["fit"]["valid_date_count"], 12)
            self.assertEqual(production["fit"]["valid_date_count"], 12)
            self.assertAlmostEqual(
                fixed["fit"]["plugin_gls_slope"],
                production["fit"]["plugin_gls_slope"],
                places=12,
            )
            self.assertEqual(len(fixed["comparator_methods"]), 7)
            self.assertEqual(
                production["provenance"]["schema"],
                "dolphinrust-temporal-covariance-provenance/1",
            )
            self.assertEqual(receipt["scores"]["schema"], "coverage_bias_interval_score/1")
            self.assertEqual(receipt["scores"]["methods"]["ols"]["scored"], 2)
            self.assertIn("resource", fixed)

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

    def test_production_path_seed_mismatch_fails_closed(self):
        path = ROOT / "validation/temporal_covariance_simulation.py"
        spec = importlib.util.spec_from_file_location("temporal_covariance_simulation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cell = module.cells(self.prereg)[0]
        request = module.request_for(cell, 99, self.prereg, "production_path")
        request["production_path"]["issue54_seed"] = 100
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
