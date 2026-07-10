from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gps_ground_truth as gps


LINE = "MMX1 23JAN04 2023.0082 59948 2243 3 -99.1 3319 0.500000 2149449 0.250000 2236 -0.100000 0.0000 0.001 0.001 0.004 0 0 0 19.4316533 -99.0683894 2235.9"


class GroundTruthContract(unittest.TestCase):
    def test_tenv3_parser_maps_documented_columns(self) -> None:
        record = gps.parse_tenv3(LINE)[0]
        self.assertEqual(record.date, dt.date(2023, 1, 4))
        self.assertAlmostEqual(record.east_m, 3319.5)
        self.assertAlmostEqual(record.north_m, 2149449.25)
        self.assertAlmostEqual(record.up_m, 2235.9)
        self.assertAlmostEqual(record.latitude, 19.4316533)
        with self.assertRaisesRegex(ValueError, "23 columns"):
            gps.parse_tenv3("MMX1 23JAN04 bad")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            gps.parse_tenv3(f"{LINE}\n{LINE}")
        with self.assertRaisesRegex(ValueError, "mixes station IDs"):
            gps.parse_tenv3(
                f"{LINE}\n{LINE.replace('MMX1 23JAN04', 'ICMX 23JAN05')}"
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            gps.parse_tenv3(LINE.replace("0.001 0.001", "nan 0.001"))

    def test_date_alignment_exact_interpolated_and_rejects_extrapolation(self) -> None:
        records = gps.parse_tenv3("\n".join([LINE, LINE.replace("23JAN04", "23JAN06").replace("-0.100000", "-0.120000")]))
        aligned = gps.align_records(records, [dt.date(2023, 1, 4), dt.date(2023, 1, 5)], 2)
        self.assertEqual([a.quality for a in aligned], ["exact", "interpolated"])
        self.assertAlmostEqual(aligned[1].record.up_m, 2235.89)
        with self.assertRaisesRegex(gps.NotEvaluable, "extrapolate"):
            gps.align_records(records, [dt.date(2023, 1, 3)], 2)

    def test_enu_projection_sign_and_reference_cancellation(self) -> None:
        los = np.array([-0.4, 0.1, np.sqrt(0.83)])
        self.assertAlmostEqual(
            gps.project_enu(np.array([0.0, 0.0, -0.1]), los),
            -0.1 * np.sqrt(0.83),
        )
        mmx1 = np.array([0.0, -20.0, -100.0])
        icmx = np.array([0.0, -2.0, -5.0])
        offset = np.array([0.0, 42.0, -11.0])
        self.assertTrue(np.array_equal(gps.spatial_difference(mmx1, icmx), gps.spatial_difference(mmx1 + offset, icmx + offset)))

    def test_window_stats_and_invalid_fraction(self) -> None:
        data = np.arange(25.0).reshape(5, 5)
        stats = gps.window_stats(data, 2, 2, 5, 0.5)
        self.assertEqual(stats.valid_count, 25)
        self.assertAlmostEqual(stats.mean, 12.0)
        data[:4, :] = np.nan
        with self.assertRaisesRegex(gps.NotEvaluable, "finite"):
            gps.window_stats(data, 2, 2, 5, 0.5)
        with self.assertRaisesRegex(ValueError, "finite fraction"):
            gps.window_stats(np.ones((3, 3)), 1, 1, 3, 0.0)

    def test_epoch_reconstruction_and_metrics(self) -> None:
        bands = [np.full((2, 2), 1.0), np.full((2, 2), 2.0)]
        cube = gps.prepend_reference_epoch(bands)
        self.assertEqual(cube.shape, (3, 2, 2))
        self.assertTrue(np.array_equal(cube[0], np.zeros((2, 2))))
        truth = np.array([0.0, -50.0, -100.0])
        same = gps.compute_metrics(truth, truth, {"endpoint_error_mm": 20, "tls_slope_min": 0.85, "tls_slope_max": 1.15, "correlation_min": 0.9})
        self.assertEqual(same["status"], "pass")
        inverted = gps.compute_metrics(truth, -truth, {"endpoint_error_mm": 20, "tls_slope_min": 0.85, "tls_slope_max": 1.15, "correlation_min": 0.9})
        self.assertEqual(inverted["status"], "fail")
        self.assertAlmostEqual(inverted["tls_slope"], -1.0)

    def test_reference_pixel_inference(self) -> None:
        cube = np.ones((3, 4, 4))
        cube[:, 2, 1] = 0.0
        inferred = gps.infer_reference_pixel(cube, Path("/nonexistent/coherence.tif"))
        self.assertEqual(inferred["pixel"], [2, 1])


if __name__ == "__main__":
    unittest.main()
