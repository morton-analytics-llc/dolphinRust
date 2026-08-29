"""Contracts for the successor temporal-covariance scorer."""

import hashlib
import importlib.util
import json
import pathlib
import unittest
from fractions import Fraction


ROOT = pathlib.Path(__file__).parents[2]
SCORER = ROOT / "validation/score_temporal_covariance_synthetic_v7.py"
POLICY = ROOT / "validation/temporal_covariance_scorer_policy_v7.json"
FROZEN_V5 = {
    ROOT / "validation/temporal_covariance_synthetic_engine_preregistration.json": (
        "bf8a0cc92d6f0f4e03bb3c0fea88ea411b897d20373376d021540c55dce77166"
    ),
    ROOT / "validation/temporal_covariance_simulation.py": (
        "6684130b2b8f596bef67de70ed39f00b8cb65cb1023beb169307f660834f7d56"
    ),
    ROOT / "validation/results/temporal_covariance/no_go_summary.json": (
        "0c885ac25f6680a18b1739e7c126c5821bc153c808c00e7b51c0b4e001ef483e"
    ),
}


def load_scorer():
    spec = importlib.util.spec_from_file_location(
        "score_temporal_covariance_synthetic_v7", SCORER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy(
    *,
    cell_count=24,
    calibration_count=5000,
    bias_tolerance="0.05",
    methods=None,
    baselines=None,
):
    methods = methods or ["candidate", "oracle", "baseline", "diagnostic"]
    baselines = baselines or ["oracle", "baseline"]
    return {
        "schema": "dolphinrust-temporal-covariance-scorer-policy/1",
        "scientific_cell_count": cell_count,
        "truth": 0.0,
        "selected_method": "candidate",
        "oracle_method": "oracle",
        "methods": methods,
        "baseline_methods": baselines,
        "familywise_alpha": "0.05",
        "standardized_bias_tolerance": bias_tolerance,
        "calibration_count_per_cell": calibration_count,
        "method_emission_minimum": "0.99",
        "pairwise_overlap_minimum": "0.98",
        "coverage": {
            "68": {"nominal": "0.68", "tolerance": "0.03"},
            "90": {"nominal": "0.90", "tolerance": "0.02"},
            "95": {"nominal": "0.95", "tolerance": "0.015"},
        },
        "pairwise_rules": {
            "oracle": {
                "maximum_mean_score_ratio": "1.10",
                "maximum_mean_width_ratio": "10.0",
            },
            "baseline": {"maximum_mean_score_ratio": "1.0"},
        },
    }


def source_identity(role, start, count, calibration_receipt=None):
    identity = {
        "source_sha256": "11" * 32,
        "source_preregistration_sha256": "22" * 32,
        "run_manifest_sha256": "33" * 32,
        "run_commit_sha256": "44" * 32,
        "seed_domain_role": role,
        "seed_domain": {"start": start, "count": count},
    }
    if calibration_receipt is not None:
        identity["calibration_receipt"] = calibration_receipt
    return identity


def forensic_source_identity():
    return {
        "source_sha256": FROZEN_V5[
            ROOT / "validation/temporal_covariance_simulation.py"
        ],
        "source_preregistration_sha256": FROZEN_V5[
            ROOT / "validation/temporal_covariance_synthetic_engine_preregistration.json"
        ],
        "no_go_summary_sha256": FROZEN_V5[
            ROOT / "validation/results/temporal_covariance/no_go_summary.json"
        ],
        "run_manifest_sha256": (
            "bdab395890265496f1fbba8118f741b33be222647e30e3d27b4d84ad33aef05c"
        ),
        "run_commit_sha256": (
            "db53c284bda9be95010622b77c91f783fe668ac65552605f3460f1484ac8f0d6"
        ),
        "seed_domain_role": "frozen_v5",
        "seed_domain": {"start": 0, "count": 1050},
    }


def comparator(
    seed, *, truth=0.0, covered_68=None, covered_90=None, covered_95=None
):
    point = truth + (-1.0 if seed % 2 == 0 else 1.0)
    covered = {
        "68": seed % 100 < 68 if covered_68 is None else covered_68,
        "90": seed % 100 < 90 if covered_90 is None else covered_90,
        "95": seed % 100 < 95 if covered_95 is None else covered_95,
    }
    return {
        "status": "Evaluated",
        "point_estimate": point,
        "intervals": {
            label: ({"lower": truth - 0.5, "upper": truth + 0.5} if is_covered else {
                "lower": truth + 1.0,
                "upper": truth + 2.0,
            })
            for label, is_covered in covered.items()
        },
    }


def records(
    cell_count, seed_count, methods, *, seed_start=0, truth=0.0, cell_ids=None
):
    cell_ids = cell_ids or [f"cell-{cell_index:02d}" for cell_index in range(cell_count)]
    for cell_index, cell_id in enumerate(cell_ids):
        for seed in range(seed_start, seed_start + seed_count):
            yield {
                "cell_id": cell_id,
                "seed": seed,
                "methods": {
                    method: comparator(seed, truth=truth) for method in methods
                },
            }


def canonical_policy():
    return json.loads(POLICY.read_bytes())


def forensic_records(methods):
    for cell_index in range(24):
        for execution_path in ("fixed_factor", "production_path"):
            for seed in range(1050):
                yield {
                    "cell_id": f"u{cell_index:02d}",
                    "cell_index": cell_index,
                    "execution_path": execution_path,
                    "outer_seed_index": seed,
                    "methods": {
                        method: comparator(seed) for method in methods
                    },
                }


class TemporalCovarianceScorerV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = load_scorer()

    def test_1050_count_family_is_not_bias_calibrated(self):
        receipt = self.scorer.score_records(
            records(24, 1050, ["oracle"]),
            policy(),
            source_identity("throwaway_oracle_calibration", 0, 1050),
            "oracle_calibration",
        )

        self.assertGreater(receipt["bias_family"]["minimum_scored_per_cell"], 1050)
        self.assertFalse(receipt["calibration_pass"])
        self.assertIn("oracle_calibration_count", receipt["failing_gates"])

    def test_5000_count_unbiased_oracle_family_calibrates(self):
        receipt = self.scorer.score_records(
            records(24, 5000, ["oracle"]),
            policy(),
            source_identity("throwaway_oracle_calibration", 0, 5000),
            "oracle_calibration",
        )

        self.assertEqual(receipt["schema"], "coverage_bias_interval_score/7")
        self.assertLessEqual(receipt["bias_family"]["minimum_scored_per_cell"], 5000)
        self.assertTrue(receipt["calibration_pass"])
        self.assertFalse(receipt["certification_eligible"])
        self.assertFalse(receipt["certification_policy_match"])
        self.assertRegex(receipt["receipt_sha256"], r"^[0-9a-f]{64}$")

    def test_frozen_policy_can_certify_oracle_calibration(self):
        frozen_policy = canonical_policy()
        self.assertEqual(
            self.scorer._sha256(frozen_policy),
            self.scorer.CERTIFICATION_POLICY_SHA256,
        )
        truth = float(frozen_policy["truth"])
        receipt = self.scorer.score_records(
            records(
                24,
                5000,
                ["oracle_gls"],
                truth=truth,
                cell_ids=frozen_policy["scientific_cells"],
            ),
            frozen_policy,
            source_identity("throwaway_oracle_calibration", 0, 5000),
            "oracle_calibration",
        )

        self.assertTrue(receipt["calibration_pass"])
        self.assertTrue(receipt["certification_policy_match"])
        self.assertTrue(receipt["certification_eligible"])

        substituted = self.scorer.score_records(
            records(24, 1, ["oracle_gls"], truth=truth),
            frozen_policy,
            source_identity("throwaway_oracle_calibration", 0, 1),
            "oracle_calibration",
        )
        self.assertFalse(substituted["cell_family_complete"])
        self.assertFalse(substituted["certification_eligible"])

    def test_exact_coverage_lower_boundaries_pass_and_one_count_below_fails(self):
        boundary_policy = policy(cell_count=1, calibration_count=5000)
        for level, covered, attempted in (
            ("68", 65, 100),
            ("90", 88, 100),
            ("95", 187, 200),
        ):
            receipts = []
            for covered_count in (covered, covered - 1):
                fixture = []
                for seed in range(attempted):
                    value = comparator(seed)
                    value["intervals"][level] = (
                        {"lower": -0.5, "upper": 0.5}
                        if seed < covered_count
                        else {"lower": 1.0, "upper": 2.0}
                    )
                    fixture.append({
                        "cell_id": "cell-00",
                        "seed": seed,
                        "methods": {"candidate": value},
                    })
                receipts.append(self.scorer.score_records(
                    fixture,
                    boundary_policy,
                    source_identity(
                        "throwaway_oracle_calibration", 0, attempted
                    ),
                    "oracle_calibration",
                ))
            passing_row, failing_row = (
                next(row for row in receipt["method_rows"]
                     if row["method"] == "candidate")
                for receipt in receipts
            )
            expected = Fraction(covered, attempted)
            with self.subTest(level=level):
                self.assertEqual(
                    passing_row["intervals"][level]["coverage"],
                    {
                        "numerator": expected.numerator,
                        "denominator": expected.denominator,
                    },
                )
                self.assertEqual(
                    passing_row["intervals"][level]["covered"], covered
                )
                self.assertEqual(
                    passing_row["intervals"][level]["emitted"], attempted
                )
                self.assertTrue(passing_row["gates"][f"coverage_{level}"])
                self.assertFalse(failing_row["gates"][f"coverage_{level}"])

    def test_exact_99_percent_emission_boundary_passes(self):
        boundary_policy = policy(cell_count=1, calibration_count=5000)
        receipts = []
        for missing in (1, 2):
            fixture = [
                {
                    "cell_id": "cell-00",
                    "seed": seed,
                    "methods": (
                        {} if seed < missing else {"candidate": comparator(seed)}
                    ),
                }
                for seed in range(100)
            ]
            receipts.append(self.scorer.score_records(
                fixture,
                boundary_policy,
                source_identity("throwaway_oracle_calibration", 0, 100),
                "oracle_calibration",
            ))
        passing_row, failing_row = (
            next(row for row in receipt["method_rows"]
                 if row["method"] == "candidate")
            for receipt in receipts
        )

        self.assertTrue(passing_row["gates"]["emission"])
        self.assertTrue(passing_row["gates"]["interval_emission_95"])
        self.assertFalse(failing_row["gates"]["emission"])
        self.assertFalse(failing_row["gates"]["interval_emission_95"])

    def test_pairwise_reducers_use_each_baseline_intersection(self):
        pair_policy = policy(
            cell_count=1,
            calibration_count=5000,
            methods=["candidate", "oracle", "baseline"],
        )
        fixture = []
        for seed in range(100):
            available = {}
            if seed != 0:
                available["candidate"] = comparator(seed)
            if seed != 1:
                available["oracle"] = comparator(seed)
            if seed != 2:
                available["baseline"] = comparator(seed)
            fixture.append({"cell_id": "cell-00", "seed": seed, "methods": available})

        receipt = self.scorer.score_records(
            fixture,
            pair_policy,
            source_identity("throwaway_oracle_calibration", 0, 100),
            "oracle_calibration",
        )
        rows = {
            (row["baseline_method"], row["level"]): row
            for row in receipt["pairwise_rows"]
        }

        self.assertEqual(rows[("oracle", "95")]["paired"], 98)
        self.assertEqual(rows[("oracle", "95")]["selected_only"], 1)
        self.assertEqual(rows[("oracle", "95")]["baseline_only"], 1)
        self.assertTrue(rows[("oracle", "95")]["gates"]["overlap"])
        self.assertEqual(rows[("baseline", "95")]["paired"], 98)
        self.assertTrue(rows[("baseline", "95")]["gates"]["overlap"])

    def test_pairwise_requires_each_method_emission_floor(self):
        pair_policy = policy(
            cell_count=1,
            calibration_count=5000,
            methods=["candidate", "oracle"],
            baselines=["oracle"],
        )
        fixture = [
            {
                "cell_id": "cell-00",
                "seed": seed,
                "methods": {
                    "candidate": comparator(seed),
                    **({} if seed < 2 else {"oracle": comparator(seed)}),
                },
            }
            for seed in range(100)
        ]

        receipt = self.scorer.score_records(
            fixture,
            pair_policy,
            source_identity("throwaway_oracle_calibration", 0, 100),
            "oracle_calibration",
        )
        row = next(
            row
            for row in receipt["pairwise_rows"]
            if row["baseline_method"] == "oracle" and row["level"] == "95"
        )

        self.assertTrue(row["gates"]["overlap"])
        self.assertTrue(row["gates"]["selected_emission"])
        self.assertFalse(row["gates"]["baseline_emission"])
        self.assertFalse(row["passes_all_gates"])

    def test_candidate_gate_ignores_unselected_diagnostic_failure(self):
        compact_policy = policy(
            cell_count=1,
            calibration_count=100,
            bias_tolerance="1.0",
            methods=["candidate", "oracle", "diagnostic"],
            baselines=["oracle"],
        )
        calibration = self.scorer.score_records(
            records(1, 100, ["oracle"]),
            compact_policy,
            source_identity("throwaway_oracle_calibration", 0, 100),
            "oracle_calibration",
        )
        self.assertTrue(calibration["calibration_pass"])

        receipt = self.scorer.score_records(
            records(1, 100, ["candidate", "oracle"], seed_start=100),
            compact_policy,
            source_identity(
                "candidate_evaluation", 100, 100, calibration
            ),
            "candidate_evaluation",
        )
        diagnostic = next(
            row for row in receipt["method_rows"] if row["method"] == "diagnostic"
        )

        self.assertFalse(diagnostic["passes_all_gates"])
        self.assertTrue(receipt["selected_method_pass"])
        self.assertTrue(receipt["candidate_pass"])
        self.assertFalse(receipt["certification_eligible"])
        self.assertFalse(receipt["certification_policy_match"])

    def test_candidate_rejects_a_stale_or_same_domain_calibration_receipt(self):
        compact_policy = policy(
            cell_count=1,
            calibration_count=100,
            bias_tolerance="1.0",
            methods=["candidate", "oracle"],
            baselines=["oracle"],
        )
        calibration = self.scorer.score_records(
            records(1, 100, ["oracle"]),
            compact_policy,
            source_identity("throwaway_oracle_calibration", 0, 100),
            "oracle_calibration",
        )

        with self.assertRaisesRegex(ValueError, "seed domain"):
            self.scorer.score_records(
                records(1, 100, ["candidate", "oracle"]),
                compact_policy,
                source_identity("candidate_evaluation", 0, 100, calibration),
                "candidate_evaluation",
            )

        stale = dict(calibration)
        stale["policy_sha256"] = "ff" * 32
        with self.assertRaisesRegex(ValueError, "receipt hash"):
            self.scorer.score_records(
                records(1, 100, ["candidate", "oracle"], seed_start=100),
                compact_policy,
                source_identity("candidate_evaluation", 100, 100, stale),
                "candidate_evaluation",
            )

    def test_receipt_retains_every_cell_method_and_named_pair(self):
        compact_policy = policy(
            cell_count=2,
            calibration_count=100,
            bias_tolerance="1.0",
            methods=["candidate", "oracle", "baseline", "diagnostic"],
        )
        receipt = self.scorer.score_records(
            records(2, 4, ["candidate", "oracle", "baseline"]),
            compact_policy,
            source_identity("throwaway_oracle_calibration", 0, 4),
            "oracle_calibration",
        )

        self.assertEqual(len(receipt["method_rows"]), 2 * 4)
        self.assertEqual(len(receipt["pairwise_rows"]), 2 * 2 * 3)
        method_row = receipt["method_rows"][0]
        self.assertTrue({
            "cell_id", "execution_path", "method", "attempted", "scored",
            "bias_moments", "intervals", "gates", "failing_gates",
        }.issubset(method_row))

    def test_forensic_mode_cannot_retroactively_certify_v5(self):
        receipt = self.scorer.score_records(
            forensic_records(["candidate", "oracle", "baseline"]),
            policy(),
            forensic_source_identity(),
            "forensic_v5",
        )

        self.assertFalse(receipt["certification_eligible"])
        self.assertFalse(receipt["retroactive_v5_certification"])
        self.assertFalse(receipt["validation_pass"])

    def test_forensic_mode_rejects_non_v5_schedule(self):
        with self.assertRaisesRegex(ValueError, "frozen v5"):
            self.scorer.score_records(
                records(1, 4, ["candidate", "oracle", "baseline"]),
                policy(cell_count=1, calibration_count=100, bias_tolerance="1.0"),
                forensic_source_identity(),
                "forensic_v5",
            )

    def test_frozen_v5_artifacts_are_byte_identical(self):
        for path, expected in FROZEN_V5.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
