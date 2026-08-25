from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import discover_gnss_cohort as discovery
import score_gnss_cohort as scorer


HOLDINGS = """Sta  Lat(deg)   Long(deg) Hgt(m)  X(m) Y(m) Z(m) Dtbeg Dtend Dtmod NumSol StaOrigName
AAAA 35.0000 -120.0000 10 0 0 0 2020-01-01 2025-01-01 2025-01-02 1000
BBBB 35.1000 -120.0000 11 0 0 0 2020-01-01 2025-01-01 2025-01-02 900
CCCC 50.0000 -100.0000 12 0 0 0 2020-01-01 2025-01-01 2025-01-02 800
DDDD 19.4000 260.9000 12 0 0 0 2020-01-01 2025-01-01 2025-01-02 700
"""


class CohortContract(unittest.TestCase):
    def test_holdings_and_pair_selection_are_outcome_blind(self) -> None:
        stations = discovery.parse_holdings(HOLDINGS)
        self.assertAlmostEqual(stations[-1].longitude, -99.1)
        pairs = discovery.candidate_pairs(
            stations,
            discovery.dt.date(2023, 1, 1),
            discovery.dt.date(2024, 1, 1),
            1.0,
            30.0,
            (-170.0, 5.0, -50.0, 75.0),
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0][0].station_id, pairs[0][1].station_id}, {"AAAA", "BBBB"})

    def test_preregistered_exclusions_are_applied_before_cohort_freeze(self) -> None:
        stations = discovery.parse_holdings(HOLDINGS)
        exclusions = discovery.Exclusions(
            station_ids=frozenset({"AAAA"}),
            burst_ids=frozenset({"T001_000001_IW1"}),
            site_ids=frozenset({"t002_000002_iw1_bbbb_cccc"}),
        )
        eligible = discovery.exclude_stations(stations, exclusions)
        self.assertEqual(
            [station.station_id for station in eligible],
            ["BBBB", "CCCC", "DDDD"],
        )
        shared = [
            ("T001_000001_IW1", ["2023-01-01"]),
            ("T002_000002_IW1", ["2023-01-01"]),
            ("T003_000003_IW1", ["2023-01-01"]),
        ]
        selected = discovery.select_shared_burst(
            shared,
            used_bursts=set(),
            first_station_id="BBBB",
            second_station_id="CCCC",
            exclusions=exclusions,
        )
        self.assertEqual(selected, ("T003_000003_IW1", ["2023-01-01"]))

    def test_interval_score_abstains_on_invalid_sigma(self) -> None:
        metrics = scorer.interval_metrics(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, np.nan])
        )
        self.assertEqual(metrics["evaluated"], 1)
        self.assertEqual(metrics["abstained"], 2)

    def test_five_distinct_receipts_run_the_direction_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(5):
                residual = np.array([0.0, 0.8, -0.9, 1.1, -1.0]) * (1 + index * 0.02)
                truth = np.zeros_like(residual)
                insar = residual
                payload = {
                    "status": "pass",
                    "comparison": f"pair_{index}",
                    "gnss_diff_mm": truth.tolist(),
                    "context": {"burst_id": f"T{index:03d}_000001_IW1"},
                    "engines": {
                        "native": {
                            "insar_diff_mm": insar.tolist(),
                            "uncertainty_reliability": {
                                "crlb_only": {"sigma_mm": [0.0, 0.4, 0.4, 0.4, 0.4]},
                                "posterior_only": {"sigma_mm": [0.0, 0.5, 0.5, 0.5, 0.5]},
                            },
                        }
                    },
                }
                path = Path(tmp) / f"site_{index}.json"
                path.write_text(json.dumps(payload))
                paths.append(path)
            result = scorer.score(paths, "native")
            self.assertEqual(result["site_count"], 5)
            self.assertIn(result["status"], {"pass", "fail"})
            self.assertEqual(len(result["folds"]), 5)

    def test_too_few_receipts_are_not_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "status": "pass",
                "comparison": "pair",
                "gnss_diff_mm": [0.0, 0.0, 0.0],
                "context": {"burst_id": "T001_000001_IW1"},
                "engines": {
                    "native": {
                        "insar_diff_mm": [0.0, 1.0, -1.0],
                        "uncertainty_reliability": {
                            "crlb_only": {"sigma_mm": [0.0, 1.0, 1.0]},
                            "posterior_only": {"sigma_mm": [0.0, 1.5, 1.5]},
                        },
                    }
                },
            }
            path = Path(tmp) / "site.json"
            path.write_text(json.dumps(payload))
            result = scorer.score([path], "native")
            self.assertEqual(result["status"], "not_evaluable")
            self.assertFalse(result["direction_gate"]["minimum_distinct_bursts"]["pass"])


if __name__ == "__main__":
    unittest.main()
