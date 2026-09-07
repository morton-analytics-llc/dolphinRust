#!/usr/bin/env python
"""One-granule metered probe for the 2018 troposphere cohort transfer gate.

Implements md/research/gps-mmx1-2018-troposphere-fetch-strategy.md: open the L4
object remotely over byte ranges, extract only the 6x5 frame window for the
bracketing height bands of both delay variables, and meter every response body
byte so the cohort transfer can be projected from a measurement instead of the
whole-object fallback.

Metering models the production access path: reads are served from a 16 KiB block
cache, the same chunk size GDAL's /vsicurl uses by default, so the count reflects
real read amplification rather than the unattainable logical minimum.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import requests

GRANULE = "OPERA_L4_TROPO-ZENITH_20180106T000000Z_20250922T192334Z_HRES_v1.0"
URL = (
    "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L4_TROPO-ZENITH_V1/"
    f"{GRANULE}/{GRANULE}.nc"
)
OBJECT_BYTES = 2_147_849_917  # from the strategy doc, for this exact granule
MAX_OBJECT_BYTES = 2_223_500_438  # largest object in the cohort
EPOCHS = 52
BLOCK = 16 * 1024

# 2018 frame bounds (strategy doc) and the measured frame terrain range.
W, E, S, N = -99.17691, -98.97694, 19.40265, 19.48732
OUT = Path(__file__).resolve().parent


class MeteredRangeReader:
    """Minimal seekable file object over HTTP byte ranges, counting every byte."""

    def __init__(self, url: str, size: int, block: int = BLOCK):
        self.url, self.size, self.block = url, size, block
        self._pos = 0
        self._cache: dict[int, bytes] = {}
        self.transfer_bytes = 0
        self.requests = 0
        self.session = requests.Session()

    def _fetch(self, index: int) -> bytes:
        if index in self._cache:
            return self._cache[index]
        start = index * self.block
        stop = min(start + self.block, self.size) - 1
        r = self.session.get(
            self.url, headers={"Range": f"bytes={start}-{stop}"}, timeout=120
        )
        if r.status_code != 206:
            raise RuntimeError(f"range request not honored: HTTP {r.status_code}")
        body = r.content
        self.transfer_bytes += len(body)
        self.requests += 1
        self._cache[index] = body
        return body

    def read(self, length: int = -1) -> bytes:
        if length < 0:
            length = self.size - self._pos
        length = max(0, min(length, self.size - self._pos))
        out = bytearray()
        while len(out) < length:
            idx, off = divmod(self._pos + len(out), self.block)
            chunk = self._fetch(idx)[off:]
            out.extend(chunk[: length - len(out)])
        self._pos += len(out)
        return bytes(out)

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self.size}[whence]
        self._pos = max(0, min(base + offset, self.size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False


def signed_url(token: str) -> tuple[str, int]:
    """Resolve the redirect once with auth, then use the signed URL unauthenticated."""
    r = requests.get(
        URL,
        headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0"},
        allow_redirects=True,
        timeout=120,
    )
    r.raise_for_status()
    size = int(r.headers["Content-Range"].split("/")[-1])
    return r.url, size


def bracketing(heights: np.ndarray, lo: float, hi: float) -> list[int]:
    """Inclusive height indices bracketing [lo, hi]."""
    order = np.argsort(heights)
    h = heights[order]
    first = int(np.searchsorted(h, lo, side="right") - 1)
    last = int(np.searchsorted(h, hi, side="left"))
    first = max(first, 0)
    last = min(last, len(h) - 1)
    return [int(order[i]) for i in range(first, last + 1)]


def main() -> None:
    token = os.environ.get("GP_EARTHDATA_TOKEN") or os.environ.get("EARTHDATA_TOKEN")
    if not token:
        sys.exit("no Earthdata token; run: source validation/creds.sh")
    dem = json.loads((OUT / "dem_range.json").read_text())
    lo, hi = dem["terrain_min_m"], dem["terrain_max_m"]

    url, size = signed_url(token)
    if size != OBJECT_BYTES:
        print(f"note: object is {size} bytes, doc records {OBJECT_BYTES}")

    reader = MeteredRangeReader(url, size)
    staged: dict[str, np.ndarray] = {}
    with h5py.File(reader, "r") as f:
        print("datasets:", sorted(f.keys()))
        lats = f["latitude"][:]
        lons = f["longitude"][:]
        heights = f["height"][:]
        times = f["time"][:]

        rows = np.where((lats >= S) & (lats <= N))[0]
        cols = np.where((lons >= W) & (lons <= E))[0]
        # Widen by one cell each side so the frame is strictly covered.
        r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 2, len(lats))
        c0, c1 = max(cols.min() - 1, 0), min(cols.max() + 2, len(lons))
        bands = bracketing(np.asarray(heights, dtype="float64"), lo, hi)
        print(f"terrain {lo:.2f}..{hi:.2f} m -> height indices {bands} "
              f"= {[float(heights[b]) for b in bands]}")
        print(f"window rows {r0}:{r1} cols {c0}:{c1} "
              f"({r1 - r0}x{c1 - c0}), heights {len(bands)}, time {times.shape}")

        after_meta = reader.transfer_bytes
        print(f"metadata+coords so far: {after_meta:,} bytes in {reader.requests} requests")

        # Variables are (time, height, lat, lon); the bracketing bands are
        # contiguous, so one hyperslab per variable touches the fewest chunks.
        b0, b1 = bands[0], bands[-1] + 1
        for var in ("hydrostatic_delay", "wet_delay"):
            d = f[var]
            block = np.asarray(d[:, b0:b1, r0:r1, c0:c1])
            fill = d.attrs.get("_FillValue")
            valid = block != fill[0] if fill is not None else np.isfinite(block)
            staged[var] = block
            print(f"  {var}: shape {d.shape} chunks {d.chunks} -> staged "
                  f"{block.shape} valid {int(valid.sum())}/{block.size} "
                  f"range [{block[valid].min():.4f}, {block[valid].max():.4f}] m")

        staged["latitude"] = np.asarray(lats[r0:r1])
        staged["longitude"] = np.asarray(lons[c0:c1])
        staged["height"] = np.asarray(heights[bands[0]:bands[-1] + 1])
        staged["time"] = np.asarray(times)

    probe = reader.transfer_bytes
    projected = math.ceil(probe * EPOCHS * MAX_OBJECT_BYTES / OBJECT_BYTES)
    half = OBJECT_BYTES // 2
    verdict = "FALLBACK (no-go)" if probe >= half else "measured"

    print()
    print(f"probe_transfer_bytes={probe}  ({probe / 1e6:.2f} MB) "
          f"in {reader.requests} range requests")
    print(f"half-object no-go threshold = {half:,}  -> {verdict}")
    print(f"projected_total_transfer_bytes={projected}")
    print(f"  = {projected / 1e9:.3f} GB vs fallback 111.639 GB "
          f"({111_638_814_943 / projected:.1f}x smaller)")

    np.savez(OUT / "staged_subset.npz", **staged)
    (OUT / "probe_receipt.txt").write_text(
        f"granule={GRANULE}\n"
        f"object_bytes={size}\n"
        f"terrain_min_m={lo}\nterrain_max_m={hi}\n"
        f"height_indices={bands}\n"
        f"window_rows={r1 - r0}\nwindow_cols={c1 - c0}\n"
        f"block_bytes={BLOCK}\n"
        f"range_requests={reader.requests}\n"
        f"probe_transfer_bytes={probe}\n"
        f"projected_total_transfer_bytes={projected}\n"
    )
    print(f"\nwrote {OUT / 'probe_receipt.txt'}")


if __name__ == "__main__":
    main()
