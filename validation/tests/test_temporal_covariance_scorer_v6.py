from __future__ import annotations

import unittest
from unittest.mock import patch

from validation.temporal_covariance_simulation import (
    MethodReducer,
    StreamingScores,
    seed_identity,
)


PREREGISTRATION = {
    "thresholds": {
        "coverage": {"0.68": 0.03, "0.90": 0.02, "0.95": 0.015},
        "standardized_bias": 0.05,
        "proper_score": 0.10,
        "maximum_interval_width_ratio": 10.0,
        "minimum_successful_emission_fraction": 0.99,
    }
}


def comparator(status: str, truth: float) -> dict:
    return {
        "status": status,
        "point_estimate": truth,
        "interval_68": {"lower": truth - 1.0, "upper": truth + 1.0},
        "interval_90": {"lower": truth - 2.0, "upper": truth + 2.0},
        "interval_95": {"lower": truth - 3.0, "upper": truth + 3.0},
    }


class TemporalCovarianceScorerV6Tests(unittest.TestCase):
    def test_non_evaluated_numeric_comparator_is_not_scored(self):
        truth = 3.6525
        reducer = MethodReducer()
        reducer.update(comparator("OptimizerNonconverged", truth), truth)
        result = reducer.finalize(PREREGISTRATION)
        self.assertEqual(result["aggregate"]["attempted"], 1)
        self.assertEqual(result["aggregate"]["scored"], 0)
        self.assertEqual(result["aggregate"]["failed"], 1)

    def test_conditional_and_unconditional_coverage_count_abstention_as_miss(self):
        truth = 3.6525
        reducer = MethodReducer()
        reducer.update(comparator("Evaluated", truth), truth)
        reducer.update(None, truth)
        result = reducer.finalize(PREREGISTRATION)
        aggregate = result["aggregate"]
        for label in ("68", "90", "95"):
            self.assertEqual(aggregate[f"interval_emitted_{label}"], 1)
            self.assertEqual(aggregate[f"conditional_coverage_{label}"], 1.0)
            self.assertEqual(aggregate[f"unconditional_coverage_{label}"], 0.5)

    def test_narrowed_promotion_method_list_is_rejected(self):
        preregistration = self._streaming_preregistration()
        preregistration["promotion_methods"] = [preregistration["selected_method"]]
        with patch(
            "validation.temporal_covariance_simulation.cells",
            return_value=[{"cell_id": "c0", "cell_index": 0}],
        ):
            with self.assertRaisesRegex(RuntimeError, "promotion methods"):
                StreamingScores(preregistration)

    def _streaming_preregistration(self) -> dict:
        return {
            "global_seed": 17,
            "outer_seeds_per_supported_cell": 3,
            "execution_paths": ["fixed_factor"],
            "selected_method": "reml_covariance_parameter_adjusted_scalar",
            "promotion_methods": [
                "oracle_gls",
                "plugin_gls_reml",
                "reml_covariance_parameter_adjusted_scalar",
                "slope_profile_likelihood_ml",
                "complete_refit_bootstrap",
            ],
            "schemas": {"scorer": "coverage_bias_interval_score/6"},
            "thresholds": {
                "coverage": {"0.68": 1.0, "0.90": 1.0, "0.95": 1.0},
                "standardized_bias": 0.05,
                "proper_score": 0.10,
                "maximum_interval_width_ratio": 10.0,
                "minimum_successful_emission_fraction": 0.99,
            },
        }

    def _scored_comparator(
        self, status: str, point: float, half_width: float,
    ) -> dict:
        value = comparator(status, point)
        for label in ("68", "90", "95"):
            value[f"interval_{label}"] = {
                "lower": point - half_width,
                "upper": point + half_width,
            }
        return value

    def _record(self, preregistration: dict, seed_index: int, fit: dict) -> dict:
        seed, digest = seed_identity(preregistration, 0, seed_index)
        return {
            "cell_id": "c0",
            "execution_path": "fixed_factor",
            "outer_seed_index": seed_index,
            "seed": seed,
            "seed_sha256": digest,
            "fit": fit,
        }

    def test_profile_and_bootstrap_failures_block_full_promotion(self):
        preregistration = self._streaming_preregistration()
        truth = 0.01 * 365.25
        with patch(
            "validation.temporal_covariance_simulation.cells",
            return_value=[{"cell_id": "c0", "cell_index": 0}],
        ):
            scorer = StreamingScores(preregistration)
        for index, offset in enumerate((-0.01, 0.0, 0.01)):
            scorer.update(self._record(preregistration, index, {
                "ols": self._scored_comparator("Evaluated", truth + offset, 10.0),
                "oracle_gls": self._scored_comparator(
                    "Evaluated", truth + offset, 1.0
                ),
                "plugin_gls": self._scored_comparator(
                    "Evaluated", truth + offset, 1.05
                ),
                "adjusted_scalar": self._scored_comparator(
                    "Evaluated", truth + offset, 1.0
                ),
                "adjusted_profile": self._scored_comparator(
                    "OptimizerNonconverged", truth, 1.0
                ),
                "complete_refit_bootstrap": self._scored_comparator(
                    "BootstrapInsufficientSuccess", truth, 1.0
                ),
            }))
        result = scorer.finalize(require_complete=True)
        self.assertTrue(result["selected_method_pass"])
        self.assertTrue(result["oracle_reference_pass"])
        self.assertTrue(result["comparison_complete"])
        self.assertFalse(result["all_methods_pass"])

    def test_all_preregistered_promotion_methods_are_required_and_can_pass(self):
        preregistration = self._streaming_preregistration()
        truth = 0.01 * 365.25
        with patch(
            "validation.temporal_covariance_simulation.cells",
            return_value=[{"cell_id": "c0", "cell_index": 0}],
        ):
            scorer = StreamingScores(preregistration)
        for index, offset in enumerate((-0.01, 0.0, 0.01)):
            scorer.update(self._record(preregistration, index, {
                "ols": self._scored_comparator("Evaluated", truth + offset, 10.0),
                "oracle_gls": self._scored_comparator(
                    "Evaluated", truth + offset, 1.0
                ),
                "plugin_gls": self._scored_comparator(
                    "Evaluated", truth + offset, 1.05
                ),
                "adjusted_scalar": self._scored_comparator(
                    "Evaluated", truth + offset, 1.0
                ),
                "adjusted_profile": self._scored_comparator(
                    "Evaluated", truth + offset, 1.0
                ),
                "complete_refit_bootstrap": self._scored_comparator(
                    "Evaluated", truth + offset, 1.0
                ),
            }))
        result = scorer.finalize(require_complete=True)
        self.assertEqual(
            result["promotion_methods"], preregistration["promotion_methods"]
        )
        self.assertTrue(result["all_methods_pass"])

    def test_adjusted_scalar_failure_blocks_selection_when_bootstrap_passes(self):
        preregistration = self._streaming_preregistration()
        truth = 0.01 * 365.25
        with patch(
            "validation.temporal_covariance_simulation.cells",
            return_value=[{"cell_id": "c0", "cell_index": 0}],
        ):
            scorer = StreamingScores(preregistration)
        for index, offset in enumerate((-0.01, 0.0, 0.01)):
            scorer.update(self._record(preregistration, index, {
                "ols": self._scored_comparator("Evaluated", truth + offset, 10.0),
                "oracle_gls": self._scored_comparator(
                    "Evaluated", truth + offset, 1.0
                ),
                "plugin_gls": self._scored_comparator(
                    "Evaluated", truth + offset, 1.05
                ),
                "adjusted_scalar": self._scored_comparator(
                    "OptimizerNonconverged", truth, 1.0
                ),
                "complete_refit_bootstrap": self._scored_comparator(
                    "Evaluated", truth + offset, 1.0
                ),
            }))
        result = scorer.finalize(require_complete=True)
        self.assertFalse(result["selected_method_pass"])

    def test_asymmetric_selected_abstention_invalidates_paired_comparison(self):
        preregistration = self._streaming_preregistration()
        preregistration["thresholds"]["minimum_successful_emission_fraction"] = 0.0
        truth = 0.01 * 365.25
        with patch(
            "validation.temporal_covariance_simulation.cells",
            return_value=[{"cell_id": "c0", "cell_index": 0}],
        ):
            scorer = StreamingScores(preregistration)
        for index, offset in enumerate((-0.01, 0.0, 0.01)):
            selected_status = "OptimizerNonconverged" if index == 0 else "Evaluated"
            scorer.update(self._record(preregistration, index, {
                "ols": self._scored_comparator("Evaluated", truth + offset, 10.0),
                "oracle_gls": self._scored_comparator(
                    "Evaluated", truth + offset, 1.0
                ),
                "plugin_gls": self._scored_comparator(
                    "Evaluated", truth + offset, 8.0
                ),
                "adjusted_scalar": self._scored_comparator(
                    selected_status, truth + offset, 1.0
                ),
            }))
        result = scorer.finalize(require_complete=True)
        summary = result["cell_summaries"][0]
        self.assertFalse(result["comparison_complete"])
        self.assertFalse(result["selected_method_pass"])
        for label in ("68", "90", "95"):
            comparison = summary["paired_comparisons"][label]
            self.assertEqual(comparison["paired_count"], 2)
            self.assertEqual(comparison["emitted_counts"]["selected"], 2)
            self.assertEqual(comparison["emitted_counts"]["oracle_gls"], 3)
            self.assertFalse(comparison["same_emission_set"])


if __name__ == "__main__":
    unittest.main()
