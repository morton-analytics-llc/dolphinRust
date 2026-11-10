#!/usr/bin/env python
"""Generate the full-stack GLRT SHP oracle fixtures from dolphin v0.42.0 (issue #165).

dolphin's sequential workflow selects GLRT neighbours for every ministack from
one amplitude mean/dispersion pair computed over the whole input stack
(`ps.create_ps` over the stack VRT), with `nslc` set to the whole stack's
length (`run_wrapped_phase_sequential`: `shp_nslc = slc_vrt_stack.shape[0]`).
This script runs exactly those two workflow functions on a synthetic stack and
saves the stack and the per-ministack `shp_counts` rasters dolphin wrote.

The scene is built so the answer depends on which statistics are used, and so
no decision sits near the test's threshold (chi2(1) at 1 − alpha = 10.83).
Amplitudes are deterministic; only the phase is random. Three classes:

- A (left half): amplitude 1.
- B (right half): amplitude 3.
- C (a block of A that borders B): amplitude 1, doubled during ministack 1 only.

With whole-stack statistics (nslc = 30) the GLRT statistic is 30.7 for A–B and
15.6 for C–B (rejected) and 3.5 for A–C (accepted). With per-ministack
statistics (nslc = 10) A–B is 10.2, so it is accepted, and in ministack 1 C–B
is 1.6, so it is accepted too. A ±5% date-to-date ripple keeps every
amplitude dispersion off zero, which dolphin's rasters read as nodata.

Run inside the v0.42.0 env:
    oracle/.venv-v042/bin/python oracle/gen_shp_fullstack_v042.py
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from osgeo import gdal

from dolphin import ps
from dolphin.io import VRTStack
from dolphin.workflows import ShpMethod
from dolphin.workflows.sequential import run_wrapped_phase_sequential

N, ROWS, COLS = 30, 16, 16
MINISTACK = 10
HALF = {"y": 2, "x": 2}
ALPHA = 0.001
OUT = Path(__file__).resolve().parent / "fixtures"


def synth_stack() -> np.ndarray:
    rng = np.random.default_rng(165)
    amplitude = np.ones((N, ROWS, COLS))
    amplitude[:, :, COLS // 2 :] = 3.0  # class B
    amplitude[MINISTACK : 2 * MINISTACK, 5:11, 4:8] *= 2.0  # class C, ministack 1 only
    ripple = 1.0 + 0.05 * (-1.0) ** np.arange(N)
    phase = rng.uniform(-np.pi, np.pi, (N, ROWS, COLS))
    return (amplitude * ripple[:, None, None] * np.exp(1j * phase)).astype(np.complex64)


def write_stack(stack: np.ndarray, folder: Path) -> list[Path]:
    driver = gdal.GetDriverByName("GTiff")
    files = []
    for index, image in enumerate(stack):
        date = datetime(2020, 1, 1) + timedelta(days=12 * index)
        path = folder / f"{date:%Y%m%d}.tif"
        ds = driver.Create(str(path), COLS, ROWS, 1, gdal.GDT_CFloat32)
        ds.SetGeoTransform([500000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0])
        ds.SetProjection("EPSG:32611")
        ds.GetRasterBand(1).WriteArray(image)
        ds = None
        files.append(path)
    return files


def read(path: Path) -> np.ndarray:
    return gdal.Open(str(path)).ReadAsArray()


def main() -> None:
    stack = synth_stack()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        vrt = VRTStack(write_stack(stack, work), outfile=work / "slc_stack.vrt")
        amp_mean, amp_dispersion = work / "amp_mean.tif", work / "amp_dispersion.tif"
        ps.create_ps(
            reader=vrt,
            like_filename=vrt.outfile,
            output_file=work / "ps.tif",
            output_amp_mean_file=amp_mean,
            output_amp_dispersion_file=amp_dispersion,
            # No PS in a pure-speckle scene; keep them out of the neighbour counts.
            amp_dispersion_threshold=0.0,
        )
        outputs = run_wrapped_phase_sequential(
            slc_vrt_stack=vrt,
            output_folder=work / "linked",
            ministack_size=MINISTACK,
            half_window=HALF,
            strides={"x": 1, "y": 1},
            ps_mask_file=work / "ps.tif",
            amp_mean_file=amp_mean,
            amp_dispersion_file=amp_dispersion,
            shp_method=ShpMethod.GLRT,
            shp_alpha=ALPHA,
            write_closure_phase=False,
            write_crlb=False,
        )
        shp_count_files = outputs[5]
        per_ministack = sorted(p for p in shp_count_files if "average" not in p.name)
        counts = np.stack([read(path) for path in per_ministack])
        mean, dispersion = read(amp_mean), read(amp_dispersion)

    np.save(OUT / "shp_fullstack_slc_stack.npy", stack)
    np.save(OUT / "shp_fullstack_counts.npy", counts.astype(np.uint16))
    print(f"wrote full-stack SHP fixtures to {OUT}")
    print(f"  shp_fullstack_slc_stack {stack.shape} {stack.dtype}")
    print(f"  shp_fullstack_counts    {counts.shape}, per-ministack means {counts.mean(axis=(1, 2))}")
    print(f"  ministacks identical: {all((counts[0] == c).all() for c in counts)}")
    print(f"  amp mean range {mean.min():.3f}..{mean.max():.3f}, dispersion {dispersion.min():.3f}..{dispersion.max():.3f}")


if __name__ == "__main__":
    main()
