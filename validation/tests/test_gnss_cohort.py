from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import discover_gnss_cohort as discovery
import score_gnss_cohort as scorer
from heldout_temporal_covariance.cohort import (
    build_manifest,
    canonical_digest,
    validate_candidate,
    validate_manifest,
)


HOLDINGS = """Sta  Lat(deg)   Long(deg) Hgt(m)  X(m) Y(m) Z(m) Dtbeg Dtend Dtmod NumSol StaOrigName
AAAA 35.0000 -120.0000 10 0 0 0 2020-01-01 2025-01-01 2025-01-02 1000
BBBB 35.1000 -120.0000 11 0 0 0 2020-01-01 2025-01-01 2025-01-02 900
CCCC 50.0000 -100.0000 12 0 0 0 2020-01-01 2025-01-01 2025-01-02 800
DDDD 19.4000 260.9000 12 0 0 0 2020-01-01 2025-01-01 2025-01-02 700
"""


class CatalogResult:
    def __init__(self, burst: str, date: str, orbit: int = 100) -> None:
        self.properties = {
            "operaBurstID": burst,
            "startTime": f"{date}T00:00:00Z",
            "fileName": f"OPERA_{burst}_{date}_v1.1.h5",
            "fileID": f"OPERA_{burst}_{date}_v1.1",
            "flightDirection": "ASCENDING",
            "pathNumber": 1,
            "orbit": orbit,
            "groupID": f"S1A_IWDV_0001_0008_{orbit:06d}_001",
            "processingDate": "2024-01-02T00:00:00Z",
            "productVersion": "1.1",
        }
        self._geometry = {
            "type": "Polygon",
            "coordinates": [[[-120.2, 34.9], [-119.8, 34.9], [-119.8, 35.2], [-120.2, 35.2], [-120.2, 34.9]]],
        }

    def geojson(self):
        return {"type": "Feature", "geometry": self._geometry, "properties": self.properties}


class CohortContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preregistration_path = Path(__file__).parents[1] / "temporal_covariance_heldout_preregistration.json"
        cls.preregistration = json.loads(cls.preregistration_path.read_text())

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

    def test_live_catalog_metadata_maps_to_exact_frozen_candidate_schema(self) -> None:
        first, second = discovery.parse_holdings(HOLDINGS)[:2]
        dates = [f"2023-{month:02d}-01" for month in range(1, 13)]
        results = {
            date: CatalogResult("T001_000001_IW1", date, 100 + index)
            for index, date in enumerate(dates)
        }
        candidate = discovery.catalog_candidate(
            first,
            second,
            "T001_000001_IW1",
            dates,
            results,
            "a" * 64,
            self.preregistration["candidate_query"]["query_digest"],
        )
        validate_candidate(candidate, self.preregistration)
        self.assertEqual(set(candidate), discovery.CANDIDATE_FIELDS)
        self.assertEqual(candidate["station_ids"], ["AAAA", "BBBB"])
        self.assertEqual(candidate["orbit_id"], "ascending-r001")
        self.assertEqual(candidate["frame_id"], "ascending-r001-burst-t001_000001_iw1")
        self.assertEqual(candidate["epoch_count"], 12)
        self.assertEqual(candidate["metadata_hashes"]["catalog_sha256"], "a" * 64)
        self.assertEqual(len(candidate["metadata_hashes"]["burst_metadata_sha256"]), 64)
        self.assertEqual(len(candidate["metadata_hashes"]["gnss_station_metadata_sha256"]), 64)

    def test_asf_scene_group_drift_does_not_change_burst_frame_identity(self) -> None:
        first = CatalogResult("T001_000001_IW1", "2023-01-01").properties
        second = dict(first, groupID="S1A_IWDV_0002_0007_000100_001")
        self.assertEqual(discovery.frame_identity(first), discovery.frame_identity(second))

    def test_catalog_footprint_identity_binds_all_epoch_geometry_metadata(self) -> None:
        first, second = discovery.parse_holdings(HOLDINGS)[:2]
        dates = [f"2023-{month:02d}-01" for month in range(1, 13)]
        results = {
            date: CatalogResult("T001_000001_IW1", date, 100 + index)
            for index, date in enumerate(dates)
        }
        results[dates[-1]]._geometry["coordinates"][0][0][0] = -120.3
        candidate = discovery.catalog_candidate(
            first,
            second,
            "T001_000001_IW1",
            dates,
            results,
            "a" * 64,
            self.preregistration["candidate_query"]["query_digest"],
        )
        validate_candidate(candidate, self.preregistration)
        self.assertTrue(candidate["footprint_id"].startswith("sha256-"))

    def test_discovery_reads_holdings_and_asf_metadata_only(self) -> None:
        dates = [f"2023-{month:02d}-01" for month in range(1, 13)]
        results = [CatalogResult("T001_000001_IW1", date, 100 + index) for index, date in enumerate(dates)]
        args = SimpleNamespace(
            preregistration=self.preregistration_path,
            start="2023-01-01",
            end="2024-01-01",
            min_distance_km=1.0,
            max_distance_km=30.0,
            west=-170.0,
            south=5.0,
            east=-50.0,
            north=75.0,
            max_pairs=1,
            min_epochs=12,
            min_span_days=300,
            target_sites=1,
        )
        response = SimpleNamespace(
            text=HOLDINGS,
            content=HOLDINGS.encode(),
            raise_for_status=lambda: None,
        )
        with patch.object(discovery.requests, "get", return_value=response) as get, patch.object(
            discovery, "search_station", side_effect=[results, results]
        ):
            payload = discovery.discover(args)
        get.assert_called_once_with(discovery.HOLDINGS_URL, timeout=60)
        self.assertTrue(payload["metadata_only"])
        self.assertFalse(payload["bulk_fetch_performed"])
        self.assertEqual(len(payload["candidates"]), 1)
        validate_candidate(payload["candidates"][0], self.preregistration)

    def test_freeze_is_exact_lexical_disjoint_reconstruction(self) -> None:
        query_digest = self.preregistration["candidate_query"]["query_digest"]
        records = []
        for index in range(116):
            records.append(
                {
                    "candidate_id": f"candidate-{index:03d}",
                    "source_kind": "catalog_metadata",
                    "burst_id": f"T999_{index:06d}_IW1",
                    "orbit_id": f"orbit-{index:03d}",
                    "footprint_id": f"footprint-{index:03d}",
                    "site_id": f"site-{index:03d}",
                    "frame_id": f"frame-{index:03d}",
                    "station_ids": [f"A{index:03d}", f"B{index:03d}"],
                    "date_start": "2023-01-01",
                    "date_end": "2024-01-01",
                    "epoch_count": 24,
                    "metadata_hashes": {
                        "catalog_sha256": "0" * 64,
                        "burst_metadata_sha256": "1" * 64,
                        "gnss_station_metadata_sha256": "2" * 64,
                    },
                    "query_digest": query_digest,
                }
            )
        discovery_payload = {
            "query_digest": query_digest,
            "metadata_only": True,
            "bulk_fetch_performed": False,
            "candidates": list(reversed(records)),
            "rejected": [],
        }
        manifest = build_manifest(discovery_payload, self.preregistration)
        validate_manifest(manifest, self.preregistration)
        self.assertEqual(manifest["candidate_pool"][0]["candidate_id"], "candidate-000")
        self.assertEqual(manifest["frozen_clusters"][0]["candidate_id"], "candidate-000")
        tampered = json.loads(json.dumps(manifest))
        tampered["frozen_clusters"][0], tampered["frozen_clusters"][1] = (
            tampered["frozen_clusters"][1],
            tampered["frozen_clusters"][0],
        )
        with self.assertRaisesRegex(ValueError, "lexical"):
            validate_manifest(tampered, self.preregistration)
        self.assertEqual(manifest["preregistration_sha256"], canonical_digest(self.preregistration))

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
