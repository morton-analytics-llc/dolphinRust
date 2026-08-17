from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_gps_ground_truth as runner


class RunnerContract(unittest.TestCase):
    def test_backend_configs_have_only_allowed_differences(self) -> None:
        base = {
            "work_directory": "/tmp/base",
            "cslc_file_list": ["a.h5", "b.h5"],
            "unwrap_options": {"unwrap_method": "snaphu", "run_unwrap": True},
            "correction_options": {"geometry_files": []},
            "timeseries_options": {"reference_point": [2, 3]},
        }
        native, snaphu = runner.build_backend_configs(
            base, [Path("static.h5"), Path("neighbour.h5")], Path("/tmp/run")
        )
        runner.assert_backend_config_identity(native, snaphu)
        self.assertEqual(native["unwrap_options"]["unwrap_method"], "native")
        self.assertEqual(snaphu["unwrap_options"]["unwrap_method"], "snaphu")
        # A frame wider than one burst carries every covering STATIC, in order.
        self.assertEqual(
            native["correction_options"]["geometry_files"],
            ["static.h5", "neighbour.h5"],
        )
        native["timeseries_options"]["reference_point"] = [9, 9]
        with self.assertRaisesRegex(ValueError, "scientific settings"):
            runner.assert_backend_config_identity(native, snaphu)

    def test_result_payload_distinguishes_not_evaluable(self) -> None:
        payload = runner.result_payload("not_evaluable", "SNAPHU unavailable", {}, {})
        self.assertEqual(payload["schema"], "dolphinrust-gps-ground-truth/1")
        self.assertEqual(payload["status"], "not_evaluable")
        self.assertNotIn("token", str(payload).lower())

    def test_matrix_status_preserves_operational_failures(self) -> None:
        self.assertEqual(
            runner.matrix_status(
                {
                    "native": {"status": "complete"},
                    "snaphu": {"status": "not_evaluable"},
                }
            ),
            "not_evaluable",
        )
        self.assertEqual(
            runner.matrix_status(
                {
                    "native": {"status": "complete"},
                    "snaphu": {"status": "error"},
                }
            ),
            "error",
        )

    def test_scored_pass_overwrites_stale_failure_receipt(self) -> None:
        # issue #43: a scored run that passes must leave a receipt describing
        # that pass, not a failure receipt an earlier attempt left behind.
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            runner.write_run_receipt(
                run_root,
                "not_evaluable",
                "common GNSS availability 0.923 is below 1.000",
                {"commit": "aaaaaaa"},
                {"native": {"status": "not_evaluable"}},
            )
            score_payload = {
                "status": "pass",
                "comparison": "MMX1_minus_ICMX_common_frame",
            }
            result = runner.finalize_score_run(
                run_root,
                {"commit": "bbbbbbb"},
                {"native": {"status": "complete"}},
                score_payload,
            )
            receipt = json.loads((run_root / "run_receipt.json").read_text())
            scores = json.loads((run_root / "gps_ground_truth.json").read_text())
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["context"]["commit"], "bbbbbbb")
            self.assertEqual(scores["context"]["commit"], "bbbbbbb")
            self.assertEqual(receipt["status"], scores["status"])
            self.assertEqual(result, scores)

    def test_score_run_without_prior_receipt_still_writes_one(self) -> None:
        # issue #43: a fresh run root with no prior failure must still gain a
        # receipt on a scored pass, not be left with none at all.
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            self.assertFalse((run_root / "run_receipt.json").exists())
            runner.finalize_score_run(
                run_root, {"commit": "ccccccc"}, {}, {"status": "pass"}
            )
            self.assertTrue((run_root / "run_receipt.json").exists())
            receipt = json.loads((run_root / "run_receipt.json").read_text())
            self.assertEqual(receipt["status"], "pass")

    def test_scientific_failure_has_nonzero_exit(self) -> None:
        self.assertEqual(runner.exit_code_for_status("pass"), 0)
        self.assertEqual(runner.exit_code_for_status("complete"), 0)
        self.assertEqual(runner.exit_code_for_status("fail"), 1)
        self.assertEqual(runner.exit_code_for_status("error"), 1)
        self.assertEqual(runner.exit_code_for_status("not_evaluable"), 2)

    def test_yaml_round_trip_preserves_native_and_geometry(self) -> None:
        config = {
            "work_directory": "/tmp/native",
            "unwrap_options": {"unwrap_method": "native"},
            "correction_options": {"geometry_files": ["static.h5"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            runner.write_yaml(path, config)
            self.assertEqual(runner.load_yaml(path), config)

    def test_fixture_manifest_is_pinned_to_recipe(self) -> None:
        recipe = {
            "burst_id": "T005_008704_IW1",
            "burst_filename_id": "T005-008704-IW1",
            "expected_dates": ["2023-01-04"],
        }
        manifest = {
            "schema": "dolphinrust-gps-fixture/1",
            "fixture": "mmx1_core",
            "burst_id": recipe["burst_id"],
            "expected_dates": recipe["expected_dates"],
        }
        cslcs = [
            Path("OPERA_L2_CSLC-S1_T005-008704-IW1_20230104T004053Z_a.h5")
        ]
        static = Path(
            "OPERA_L2_CSLC-S1-STATIC_T005-008704-IW1_20140403_S1A_v1.0.h5"
        )
        runner.validate_fixture_contract(
            manifest, recipe, "mmx1_core", cslcs, static
        )
        manifest["expected_dates"] = ["2023-01-28"]
        with self.assertRaisesRegex(runner.gps.NotEvaluable, "dates"):
            runner.validate_fixture_contract(
                manifest, recipe, "mmx1_core", cslcs, static
            )

    def test_csv_and_svg_artifacts_are_written(self) -> None:
        import numpy as np
        import gps_ground_truth as gps

        dates = ["2023-01-04", "2023-01-28", "2023-02-09"]
        truth = np.array([0.0, -10.0, -20.0])
        engines = {
            "native": {"insar_diff_mm": [0.0, -9.0, -19.0]},
            "snaphu": {"insar_diff_mm": [0.0, -11.0, -21.0]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gps.write_csv(root / "result.csv", dates, truth, engines)
            gps.write_svg(root / "result.svg", dates, truth, engines)
            self.assertIn("gnss_diff_mm", (root / "result.csv").read_text())
            svg = (root / "result.svg").read_text()
            self.assertIn("<svg", svg)
            self.assertIn("native", svg)

    def test_uncertainty_reliability_artifacts_are_regenerable(self) -> None:
        import gps_ground_truth as gps

        reliability = gps.uncertainty_reliability(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 2.0, 2.0]),
        )
        payload = {
            "comparison": "MMX1_minus_ICMX_common_frame",
            "dates": ["2023-01-04", "2023-01-28", "2023-02-09"],
            "gnss_date_quality": {"MMX1": ["exact"] * 3, "ICMX": ["exact"] * 3},
            "limitations": ["synthetic contract"],
            "engines": {
                "native": {
                    "gnss_projected_sigma_mm": [0.0, 0.5, 0.5],
                    "velocity_comparison": {"insar_velocity_mm_yr": 1.0},
                    "uncertainty_reliability": reliability,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gps.write_reliability_artifacts(root, payload)
            self.assertTrue((root / "uncertainty_reliability.json").exists())
            csv_text = (root / "uncertainty_reliability.csv").read_text()
            self.assertIn("crlb_only", csv_text)
            self.assertNotIn("combined_quadrature", csv_text)
            self.assertIn("<svg", (root / "uncertainty_reliability.svg").read_text())


if __name__ == "__main__":
    unittest.main()
