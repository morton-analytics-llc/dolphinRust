#!/usr/bin/env python
"""Stage the 2018 troposphere cohort as per-date netCDF-4 window subsets.

Authorized by the metered probe in
`md/research/gps-mmx1-2018-troposphere-fetch-strategy.md`
(`projected_total_transfer_bytes=68794097`): each L4 object is ~2.1 GB, but the
frame needs a handful of gzip chunks, so the cohort is fetched by byte range and
never lands whole.

Output is real netCDF-4 with dimension scales, not bare HDF5 — the Rust reader
opens these through GDAL's `NETCDF:` driver and needs `NETCDF_DIM_height` band
metadata; a file written with plain h5py segfaults that driver.

Run:
  source validation/creds.sh
  <venv>/bin/python validation/fetch_l4_tropo_cohort.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import requests
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent.parent
VARS = ("hydrostatic_delay", "wet_delay")
BLOCK = 16 * 1024


class MeteredRangeReader:
    """Seekable file object over HTTP byte ranges that counts every byte."""

    def __init__(self, url: str, size: int, session: requests.Session):
        self.url, self.size, self.session = url, size, session
        self._pos = 0
        self._cache: dict[int, bytes] = {}
        self.transfer_bytes = 0

    def _fetch(self, index: int) -> bytes:
        if index not in self._cache:
            start = index * BLOCK
            stop = min(start + BLOCK, self.size) - 1
            r = self.session.get(
                self.url, headers={"Range": f"bytes={start}-{stop}"}, timeout=180
            )
            if r.status_code != 206:
                raise RuntimeError(f"range not honored: HTTP {r.status_code}")
            self.transfer_bytes += len(r.content)
            self._cache[index] = r.content
        return self._cache[index]

    def read(self, length: int = -1) -> bytes:
        if length < 0:
            length = self.size - self._pos
        length = max(0, min(length, self.size - self._pos))
        out = bytearray()
        while len(out) < length:
            idx, off = divmod(self._pos + len(out), BLOCK)
            out.extend(self._fetch(idx)[off:][: length - len(out)])
        self._pos += len(out)
        return bytes(out)

    def seek(self, offset: int, whence: int = 0) -> int:
        self._pos = max(0, min({0: 0, 1: self._pos, 2: self.size}[whence] + offset, self.size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def signed(url: str, token: str, session: requests.Session) -> tuple[str, int]:
    r = session.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0"},
        allow_redirects=True,
        timeout=180,
    )
    r.raise_for_status()
    return r.url, int(r.headers["Content-Range"].split("/")[-1])


def bracketing(heights: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    order = np.argsort(heights)
    h = heights[order]
    first = max(int(np.searchsorted(h, lo, side="right") - 1), 0)
    last = min(int(np.searchsorted(h, hi, side="left")), len(h) - 1)
    return int(order[first]), int(order[last]) + 1


def stage_one(date: str, url: str, token: str, bounds, terrain, out_dir: Path) -> dict:
    W, E, S, N = bounds
    lo, hi = terrain
    session = requests.Session()
    signed_url, size = signed(url, token, session)
    reader = MeteredRangeReader(signed_url, size, session)

    with h5py.File(reader, "r") as f:
        lats, lons = f["latitude"][:], f["longitude"][:]
        heights, times = f["height"][:], f["time"][:]
        rows = np.where((lats >= S) & (lats <= N))[0]
        cols = np.where((lons >= W) & (lons <= E))[0]
        r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 2, len(lats))
        c0, c1 = max(cols.min() - 1, 0), min(cols.max() + 2, len(lons))
        b0, b1 = bracketing(np.asarray(heights, dtype="float64"), lo, hi)

        data = {v: np.asarray(f[v][:, b0:b1, r0:r1, c0:c1]) for v in VARS}
        fills = {v: f[v].attrs["_FillValue"][0] for v in VARS}
        units = {v: f[v].attrs.get("units", b"meters") for v in VARS}
        coords = {
            "time": np.asarray(times),
            "height": np.asarray(heights[b0:b1]),
            "latitude": np.asarray(lats[r0:r1]),
            "longitude": np.asarray(lons[c0:c1]),
        }
        coord_units = {c: f[c].attrs.get("units", b"") for c in coords}
        time_units = f["time"].attrs.get("units", b"hours since 1900-01-01")
        crs_wkt = f["spatial_ref"].attrs.get("crs_wkt", b"")

    path = out_dir / f"l4_tropo_{date.replace('-', '')}.nc"
    with Dataset(path, "w", format="NETCDF4") as nc:
        for name, arr in coords.items():
            nc.createDimension(name, len(arr))
        for name, arr in coords.items():
            var = nc.createVariable(name, arr.dtype, (name,))
            var[:] = arr
            u = coord_units[name] if name != "time" else time_units
            var.units = u.decode() if isinstance(u, bytes) else str(u)
        crs = nc.createVariable("spatial_ref", "i4", ())
        crs.crs_wkt = crs_wkt.decode() if isinstance(crs_wkt, bytes) else str(crs_wkt)
        crs.spatial_ref = crs.crs_wkt
        for v in VARS:
            var = nc.createVariable(
                v, "f4", ("time", "height", "latitude", "longitude"),
                fill_value=fills[v], zlib=True,
            )
            var[:] = data[v]
            u = units[v]
            var.units = u.decode() if isinstance(u, bytes) else str(u)
            var.grid_mapping = "spatial_ref"
        nc.source_date = date
        nc.source_granule = url.rsplit("/", 1)[-1]

    return {
        "date": date,
        "path": str(path),
        "transfer_bytes": reader.transfer_bytes,
        "object_bytes": size,
        "staged_bytes": path.stat().st_size,
        "shape": list(data[VARS[0]].shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--granules", type=Path,
                    default=Path("/private/tmp/claude-501/-Users-ryanemorton-Documents-GitHub-dolphinRust/"
                                 "ac76b8b7-1b9d-49cd-92de-c9c18f6b0ca8/scratchpad/tropo_probe/cohort_granules.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--terrain", nargs=2, type=float, default=[2217.646240234375, 2298.483154296875])
    ap.add_argument("--bounds", nargs=4, type=float,
                    default=[-99.17691, -98.97694, 19.40265, 19.48732])
    args = ap.parse_args()

    token = os.environ.get("GP_EARTHDATA_TOKEN") or os.environ.get("EARTHDATA_TOKEN")
    if not token:
        sys.exit("no Earthdata token; run: source validation/creds.sh")
    sel = json.loads(args.granules.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(stage_one, d, v["url"], token, args.bounds, args.terrain, args.out): d
            for d, v in sel.items()
        }
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                print(f"  {res['date']}  {res['transfer_bytes']:>9,} B -> "
                      f"{res['staged_bytes']:>6,} B  {res['shape']}")
            except Exception as e:  # noqa: BLE001
                failures.append((d, repr(e)))
                print(f"  {d}  FAILED {e!r}")

    total = sum(r["transfer_bytes"] for r in results)
    print(f"\nstaged {len(results)}/{len(sel)} epochs")
    print(f"total transfer: {total:,} bytes ({total / 1e6:.1f} MB)")
    print(f"vs projected  : 68,794,097 bytes (68.8 MB)")
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    (args.out / "transfer_receipt.json").write_text(
        json.dumps({"epochs": len(results), "total_transfer_bytes": total,
                    "per_epoch": sorted(results, key=lambda r: r["date"])}, indent=1)
    )


if __name__ == "__main__":
    main()
