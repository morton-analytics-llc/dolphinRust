from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import crop_real


class CropContract(unittest.TestCase):
    def test_center_window_and_bounds_window(self) -> None:
        x = np.arange(100.0, 110.0)
        y = np.arange(210.0, 200.0, -1.0)
        row, col = crop_real.projected_to_pixel(105.0, 205.0, x, y)
        self.assertEqual((row, col), (5, 5))
        win = crop_real.centered_window(row, col, 4, (10, 10))
        self.assertEqual(win, crop_real.Window(3, 3, 4, 4))
        bounds_win = crop_real.window_for_projected_bounds((102.0, 203.0, 106.0, 207.0), x, y, 1)
        self.assertEqual(bounds_win, crop_real.Window(2, 1, 7, 7))

    def test_out_of_bounds_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            crop_real.centered_window(0, 0, 4, (10, 10))

    def test_source_dates_and_burst_must_match_recipe(self) -> None:
        recipe = {
            "burst_filename_id": "T005-008704-IW1",
            "expected_dates": ["2023-01-04", "2023-01-28"],
        }
        good = [
            Path("OPERA_L2_CSLC-S1_T005-008704-IW1_20230104T004053Z_a.h5"),
            Path("OPERA_L2_CSLC-S1_T005-008704-IW1_20230128T004052Z_b.h5"),
        ]
        crop_real.validate_cslc_files(good, recipe)
        with self.assertRaisesRegex(ValueError, "dates do not match"):
            crop_real.validate_cslc_files(list(reversed(good)), recipe)
        with self.assertRaisesRegex(ValueError, "outside burst"):
            crop_real.validate_cslc_files(
                [Path(str(good[0]).replace("T005-008704-IW1", "T006-000001-IW1"))],
                recipe,
            )

    def test_product_crop_preserves_identity_and_uses_own_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cslc = root / "cslc.h5"
            static = root / "static.h5"
            self._write_product(cslc, "cslc", np.arange(10.0), np.arange(20.0, 10.0, -1.0))
            self._write_product(static, "static", np.arange(0.0, 20.0, 2.0), np.arange(30.0, 10.0, -2.0))
            bounds = (4.0, 14.0, 8.0, 18.0)
            cslc_info = crop_real.crop_product_to_bounds(cslc, root / "cslc_crop.h5", "cslc", bounds)
            static_info = crop_real.crop_product_to_bounds(static, root / "static_crop.h5", "static", bounds)
            self.assertNotEqual(cslc_info["window"], static_info["window"])
            for path, dataset in [(root / "cslc_crop.h5", "VV"), (root / "static_crop.h5", "los_east")]:
                with h5py.File(path, "r") as f:
                    self.assertIn("identification/burst_id", f)
                    self.assertIn(f"data/{dataset}", f)

    @staticmethod
    def _write_product(path: Path, kind: str, x: np.ndarray, y: np.ndarray) -> None:
        with h5py.File(path, "w") as f:
            data = f.create_group("data")
            shape = (len(y), len(x))
            if kind == "cslc":
                data.create_dataset("VV", data=np.ones(shape, dtype=np.complex64))
            else:
                data.create_dataset("los_east", data=np.full(shape, -0.4, dtype=np.float32))
                data.create_dataset("los_north", data=np.full(shape, 0.2, dtype=np.float32))
            data.create_dataset("x_coordinates", data=x)
            data.create_dataset("y_coordinates", data=y)
            data.create_dataset("projection", data=np.int32(4326))
            ident = f.create_group("identification")
            ident.create_dataset("burst_id", data=np.bytes_("t005_008704_iw1"))
            ident.create_dataset("orbit_pass_direction", data=np.bytes_("Descending"))


if __name__ == "__main__":
    unittest.main()
