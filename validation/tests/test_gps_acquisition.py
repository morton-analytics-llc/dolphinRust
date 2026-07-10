from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_real


@dataclass
class FakeResult:
    properties: dict[str, object]


def result(level: str, date: str, name: str | None = None) -> FakeResult:
    stamp = date.replace("-", "")
    return FakeResult(
        {
            "processingLevel": level,
            "fileName": name or f"OPERA_L2_{level}_T005-008704-IW1_{stamp}T004052Z_v1.h5",
            "startTime": f"{date}T00:40:52Z",
            "url": f"https://example.test/{stamp}",
        }
    )


class AcquisitionContract(unittest.TestCase):
    def test_selects_exact_declared_dates_in_order(self) -> None:
        expected = ["2023-01-04", "2023-01-28", "2023-02-09"]
        results = [
            result("RTC", "2023-01-04"),
            result("CSLC", "2023-02-09"),
            result("CSLC", "2023-01-04"),
            result("CSLC", "2023-01-28"),
        ]
        selected = fetch_real.select_expected_cslcs(results, expected, "T005-008704-IW1")
        self.assertEqual(
            [item.properties["startTime"][:10] for item in selected],
            expected,
        )

    def test_missing_duplicate_and_wrong_burst_fail(self) -> None:
        expected = ["2023-01-04", "2023-01-28"]
        with self.assertRaisesRegex(ValueError, "missing"):
            fetch_real.select_expected_cslcs([result("CSLC", "2023-01-04")], expected, "T005-008704-IW1")
        duplicate = [result("CSLC", "2023-01-04"), result("CSLC", "2023-01-04", "duplicate_T005-008704-IW1.h5")]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fetch_real.select_expected_cslcs(duplicate, ["2023-01-04"], "T005-008704-IW1")
        with self.assertRaisesRegex(ValueError, "burst"):
            fetch_real.select_expected_cslcs([result("CSLC", "2023-01-04", "OPERA_T006-000001-IW1_20230104.h5")], ["2023-01-04"], "T005-008704-IW1")

    def test_selects_one_matching_static(self) -> None:
        good = result("CSLC-STATIC", "2014-04-03", "OPERA_L2_CSLC-S1-STATIC_T005-008704-IW1_20140403_S1A_v1.0.h5")
        self.assertIs(fetch_real.select_static_result([good], "T005-008704-IW1"), good)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            fetch_real.select_static_result([], "T005-008704-IW1")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            fetch_real.select_static_result([good, good], "T005-008704-IW1")

    def test_token_required_without_leaking_value(self) -> None:
        secret = "do-not-print-me"
        self.assertEqual(fetch_real.require_token({"GP_EARTHDATA_TOKEN": secret}), secret)
        with self.assertRaisesRegex(RuntimeError, "GP_EARTHDATA_TOKEN") as ctx:
            fetch_real.require_token({})
        self.assertNotIn(secret, str(ctx.exception))

    def test_downloaded_hdf_identities_must_match_burst_and_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cslc = root / "cslc.h5"
            static = root / "static.h5"
            self._write_identity(cslc, "t005_008704_iw1", "Ascending")
            self._write_identity(static, "t005_008704_iw1", "Ascending")
            identity = fetch_real.validate_product_identities(
                [cslc], [static], "T005_008704_IW1"
            )
            self.assertEqual(identity["orbit_pass_direction"], "ascending")
            self._write_identity(static, "t005_008704_iw1", "Descending")
            with self.assertRaisesRegex(RuntimeError, "pass identities disagree"):
                fetch_real.validate_product_identities(
                    [cslc], [static], "T005_008704_IW1"
                )
            self._write_identity(static, "t006_000001_iw1", "Ascending")
            with self.assertRaisesRegex(RuntimeError, "burst identity"):
                fetch_real.validate_product_identities(
                    [cslc], [static], "T005_008704_IW1"
                )
            self._write_identity(static, "t005_008704_iw1", "Sideways")
            with self.assertRaisesRegex(RuntimeError, "unrecognized orbit pass"):
                fetch_real.validate_product_identities(
                    [], [static], "T005_008704_IW1"
                )

    @staticmethod
    def _write_identity(path: Path, burst: str, orbit_pass: str) -> None:
        if path.exists():
            path.unlink()
        with h5py.File(path, "w") as product:
            identification = product.create_group("identification")
            identification.create_dataset("burst_id", data=np.bytes_(burst))
            identification.create_dataset(
                "orbit_pass_direction", data=np.bytes_(orbit_pass)
            )


if __name__ == "__main__":
    unittest.main()
