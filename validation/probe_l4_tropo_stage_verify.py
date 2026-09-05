#!/usr/bin/env python
"""Stage the probed L4 window and verify it against the remote read exactly.

Strategy doc steps 5-6: write the subset without changing dimension order,
coordinate values, band order, `_FillValue`/mask behaviour, units, scale/offset,
or the EPSG:4326 geotransform; then compare remote-window and staged-window
values and masks exactly.

"Exactly" here means bit-identical, not within a tolerance — this is a byte
extraction, so any difference would be a staging defect rather than numerical
drift.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_l4_tropo_transfer import (  # noqa: E402
    E,
    N,
    S,
    W,
    MeteredRangeReader,
    bracketing,
    signed_url,
)

OUT = Path(__file__).resolve().parent
STAGED = OUT / "l4_tropo_2018-01-06_window.nc"
VARS = ("hydrostatic_delay", "wet_delay")
COORDS = ("time", "height", "latitude", "longitude")


def main() -> None:
    token = os.environ.get("GP_EARTHDATA_TOKEN") or os.environ.get("EARTHDATA_TOKEN")
    if not token:
        sys.exit("no Earthdata token; run: source validation/creds.sh")
    dem = json.loads((OUT / "dem_range.json").read_text())
    lo, hi = dem["terrain_min_m"], dem["terrain_max_m"]

    url, size = signed_url(token)
    reader = MeteredRangeReader(url, size)
    remote: dict[str, np.ndarray] = {}
    attrs: dict[str, dict] = {}

    with h5py.File(reader, "r") as f:
        lats, lons = f["latitude"][:], f["longitude"][:]
        heights, times = f["height"][:], f["time"][:]
        rows = np.where((lats >= S) & (lats <= N))[0]
        cols = np.where((lons >= W) & (lons <= E))[0]
        r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 2, len(lats))
        c0, c1 = max(cols.min() - 1, 0), min(cols.max() + 2, len(lons))
        bands = bracketing(np.asarray(heights, dtype="float64"), lo, hi)
        b0, b1 = bands[0], bands[-1] + 1

        # The window must strictly contain the frame, and the height bands must
        # strictly bracket the terrain; a window that merely overlaps would
        # silently clip the frame edge during interpolation.
        lat_w, lon_w = lats[r0:r1], lons[c0:c1]
        hs = np.asarray(heights[b0:b1], dtype="float64")
        checks = {
            "lat covers frame": lat_w.min() <= S and lat_w.max() >= N,
            "lon covers frame": lon_w.min() <= W and lon_w.max() >= E,
            "heights bracket terrain": hs.min() <= lo and hs.max() >= hi,
        }
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not all(checks.values()):
            sys.exit("window does not strictly cover the frame")

        remote["latitude"], remote["longitude"] = lat_w, lon_w
        remote["height"], remote["time"] = np.asarray(heights[b0:b1]), np.asarray(times)
        for v in VARS:
            d = f[v]
            remote[v] = np.asarray(d[:, b0:b1, r0:r1, c0:c1])
            attrs[v] = {k: d.attrs[k] for k in d.attrs}
        for c in COORDS:
            attrs[c] = {k: f[c].attrs[k] for k in f[c].attrs}
        spatial_ref_attrs = {k: f["spatial_ref"].attrs[k] for k in f["spatial_ref"].attrs}
        spatial_ref_val = f["spatial_ref"][()]

    # --- write, preserving dtype, order, fill value, units, and the CRS ------
    with h5py.File(STAGED, "w") as g:
        for c in COORDS:
            ds = g.create_dataset(c, data=remote[c])
            for k, val in attrs[c].items():
                ds.attrs[k] = val
        for v in VARS:
            ds = g.create_dataset(v, data=remote[v])
            for k, val in attrs[v].items():
                ds.attrs[k] = val
        ds = g.create_dataset("spatial_ref", data=spatial_ref_val)
        for k, val in spatial_ref_attrs.items():
            ds.attrs[k] = val
        g.attrs["source_granule"] = (
            "OPERA_L4_TROPO-ZENITH_20180106T000000Z_20250922T192334Z_HRES_v1.0"
        )
        g.attrs["source_window"] = json.dumps(
            {"height": [int(b0), int(b1)], "lat": [int(r0), int(r1)], "lon": [int(c0), int(c1)]}
        )

    # --- verify staged == remote, bit for bit -------------------------------
    print("\nremote vs staged:")
    ok = True
    with h5py.File(STAGED, "r") as g:
        for key in (*COORDS, *VARS):
            a, b = remote[key], np.asarray(g[key])
            same_vals = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
                a, b, equal_nan=True
            )
            fill = attrs.get(key, {}).get("_FillValue")
            same_mask = True
            if fill is not None:
                same_mask = np.array_equal(a == fill[0], b == fill[0])
            units_a = attrs.get(key, {}).get("units")
            units_b = g[key].attrs.get("units")
            same_units = (units_a is None and units_b is None) or units_a == units_b
            good = same_vals and same_mask and same_units
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'}  {key:20s} {str(a.shape):14s} "
                  f"values={same_vals} mask={same_mask} units={same_units}")
        crs = g["spatial_ref"].attrs.get("crs_wkt", b"")
        crs = crs.decode() if isinstance(crs, bytes) else str(crs)
        has_4326 = "4326" in crs or "WGS 84" in crs
        ok &= has_4326
        print(f"  {'PASS' if has_4326 else 'FAIL'}  spatial_ref preserved (EPSG:4326)")

    print(f"\nstaged file: {STAGED} ({STAGED.stat().st_size:,} bytes)")
    print(f"verification transfer: {reader.transfer_bytes:,} bytes "
          f"in {reader.requests} requests")
    print("RESULT:", "exact match" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
