"""Schema tests for the frozen #53 preregistration.

These tests validate the contract only; they intentionally do not run the
5,000-seed experiment or acquire external GNSS data.
"""

import json
import pathlib
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
        self.assertEqual(
            self.prereg["promotion_status"],
            "blocked_pending_synthetic_field_review_and_manifest",
        )
        self.assertFalse(self.prereg["corrected_inferential_sigma_emission"])
        self.assertTrue(self.prereg["external_holdout_required"])

    def test_coverage_tolerances_are_explicit(self):
        self.assertEqual(self.prereg["coverage_tolerances"], {
            "0.68": 0.03, "0.90": 0.02, "0.95": 0.015
        })
        self.assertEqual(self.prereg["standardized_bias_tolerance"], 0.05)
        self.assertEqual(self.prereg["minimum_successful_emission_fraction"], 0.99)


if __name__ == "__main__":
    unittest.main()
