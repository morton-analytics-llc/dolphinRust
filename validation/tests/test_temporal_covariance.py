"""Schema tests for the frozen #53 preregistration.

These tests validate the contract only; they intentionally do not run the
5,000-seed experiment or acquire external GNSS data.
"""

import json
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

    def test_boundary_is_fail_closed(self):
        self.assertEqual(self.prereg["status"], "pre_outcome_draft")
        self.assertEqual(
            self.prereg["promotion_status"],
            "blocked_pending_synthetic_field_review_and_manifest",
        )
        self.assertFalse(self.prereg["corrected_inferential_sigma_emission"])
        self.assertTrue(self.prereg["external_holdout_required"])

    def test_cell_count_and_seed_denominator_are_immutable(self):
        self.assertEqual(self.prereg["cell_count_without_outer_seeds"], 4032)
        self.assertEqual(
            self.prereg["attempt_denominator"],
            "all_attempted_outer_seeds_including_fit_failures",
        )
        self.assertEqual(self.prereg["bootstrap"]["interval_levels"], [0.68, 0.9, 0.95])

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
            ]
            subprocess.run(command, check=True)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["attempted_cells"], 4032)
            self.assertEqual(receipt["emitted_cells"], 4032)
            self.assertFalse(receipt["corrected_inferential_sigma_emission"])
            self.assertEqual(receipt["pre_outcome_status"], "pre_outcome_draft")

    def test_coverage_tolerances_are_explicit(self):
        self.assertEqual(self.prereg["coverage_tolerances"], {
            "0.68": 0.03, "0.90": 0.02, "0.95": 0.015
        })
        self.assertEqual(self.prereg["standardized_bias_tolerance"], 0.05)
        self.assertEqual(self.prereg["minimum_successful_emission_fraction"], 0.99)


if __name__ == "__main__":
    unittest.main()
