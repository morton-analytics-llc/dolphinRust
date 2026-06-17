#!/usr/bin/env python
"""Generate Phase-8 (I/O) oracle fixtures.

Writes a GeoTIFF (GDAL) and an OPERA-style CSLC HDF5 (h5py) with known
geotransform / CRS / pixel values, plus the raw arrays as .npy, so the Rust
GDAL/HDF5 readers can be validated against GDAL/h5py.

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_io.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

OUT = Path(__file__).resolve().parent / "fixtures"
ROWS, COLS = 8, 10
GEOTRANSFORM = (500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0)
EPSG = 32611
CSLC_PATH = "/data/VV"


def write_geotiff() -> np.ndarray:
    arr = (np.arange(ROWS * COLS, dtype=np.float32).reshape(ROWS, COLS) * 0.5 - 3.0)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(OUT / "io_ref.tif"), COLS, ROWS, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(GEOTRANSFORM)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    ds.FlushCache()
    ds = None
    np.save(OUT / "io_ref_tif.npy", arr)
    return arr


def write_cslc() -> np.ndarray:
    import h5py

    rng = np.random.default_rng(3)
    cslc = (rng.standard_normal((ROWS, COLS)) + 1j * rng.standard_normal((ROWS, COLS))).astype(
        np.complex64
    )
    with h5py.File(OUT / "io_cslc.h5", "w") as f:
        f.create_dataset(CSLC_PATH, data=cslc)
    np.save(OUT / "io_cslc.npy", cslc)
    return cslc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tif = write_geotiff()
    cslc = write_cslc()
    print(f"wrote I/O fixtures to {OUT}")
    print(f"  io_ref.tif {tif.shape} gt={GEOTRANSFORM} epsg={EPSG}")
    print(f"  io_cslc.h5[{CSLC_PATH}] {cslc.shape} {cslc.dtype}")


if __name__ == "__main__":
    main()
