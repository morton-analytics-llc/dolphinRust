"""Issue #230 acceptance, NISAR: in-house geocoded stack from raw L1 RSLC
against JPL's L2 GSLC on the same grid.

`dolphin nisar` geocodes each RSLC onto JPL's GSLC lattice (40 m pixel
edges, 5 m posting for a 40 MHz band, 40 m x 5 m for the 5 MHz band), so our
pixels are JPL's pixels. JPL's GSLC is the backtest reference only.

    fetch  range-reads the JPL GSLC window under our grid (frequencyA/B,
           polarisation, mask) into small local HDF5 files; nothing whole is
           downloaded. The GSLC is looked up in CMR by our file's source RSLC
           granule name, falling back to the same acquisition date.
    score  compares the two stacks.

Thresholds, from what two correct processors of the same RSLC may differ on:

* Registration. Both interpolate the same RSLC with an 8-tap sinc on the
  same zero-Doppler geometry; a residual mis-registration delta (in
  resolution cells) between two copies of the same speckle decorrelates them
  by sinc(delta) per axis. At the 0.1-pixel tolerance (NISAR 40 MHz: 1.2x
  oversampled in range, 1.2x in azimuth, so 0.1 px is 0.083 cell) that costs
  sinc(0.083)^2 = 0.98. Pass: median 11x11 |gamma| >= 0.95 over pixels JPL
  itself finds coherent between consecutive dates (>= 0.7), and median
  amplitude offset (128x128 patches, 1/16 px) <= 0.1 px in rows and cols.
  The products are flattened by exp(+j4piR/lambda) with each processor's
  DEM, and JPL publishes no flattening layer, so |gamma| here also carries
  any DEM difference (52 rad per metre of range at L band); within an 11x11
  window a smooth DEM difference only rotates the phase and does not lower
  |gamma|.
* Phase. Per consecutive pair, Delta = arg(I_ours * conj(I_jpl)). The
  flattening cancels between dates except through the perpendicular baseline:
  d(Delta)/dh = (4pi/lambda) B_perp / (R sin theta), 0.02 rad per metre of DEM
  difference for B_perp = 200 m at R = 950 km. The ionosphere and tide enter
  both products as sampling shifts, not phase. After a best-fit plane, the
  circular std of Delta over pixels with JPL coherence >= 0.9 must be
  <= 0.1 rad: the phase noise of a gamma ~ 0.995 pair, what two sinc
  interpolations of the same high-SNR pixel give.

    python -I validation/issue230_nisar_front_end.py fetch --ours DIR --jpl DIR
    python -I validation/issue230_nisar_front_end.py score --ours DIR --jpl DIR \\
        --label T005-A-F020 --out results/...json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edl_token import earthdata_token  # noqa: E402
from issue230_raw_front_end import (  # noqa: E402
    circular_std,
    offsets,
    remove_plane,
    windowed_correlation,
)

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
GSLC_COLLECTION = "C2854332392-ASF"
GRIDS = "/science/LSAR/GSLC/grids"
DATE = re.compile(r"_(\d{8})T\d{6}Z")
BLOCK = 1 << 20


class RangeFile(io.RawIOBase):
    """Read-only file over HTTP range requests with a 1 MiB block cache."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.session = requests.Session()
        head = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Range"].rsplit("/", 1)[1])
        self.pos = 0
        self.blocks: dict[int, bytes] = {}
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def block(self, index: int) -> bytes:
        if index not in self.blocks:
            lo = index * BLOCK
            hi = min(lo + BLOCK, self.size) - 1
            r = self.session.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            r.raise_for_status()
            self.blocks[index] = r.content
            self.bytes_read += len(r.content)
        return self.blocks[index]

    def readinto(self, buffer) -> int:
        end = min(self.pos + len(buffer), self.size)
        out = bytearray()
        while self.pos + len(out) < end:
            at = self.pos + len(out)
            data = self.block(at // BLOCK)
            out += data[at % BLOCK : min(len(data), at % BLOCK + end - at)]
        buffer[: len(out)] = out
        self.pos = end
        return len(out)


def signed(url: str, token: str) -> str:
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0"},
        allow_redirects=False,
        timeout=60,
    )
    location = r.headers.get("Location", "")
    if not location.startswith("https://"):
        raise SystemExit(f"ASF did not redirect {url}: HTTP {r.status_code}")
    return location


def date_of(path: Path) -> str:
    match = DATE.search(path.name)
    if not match:
        raise ValueError(f"no acquisition date in {path.name}")
    return match.group(1)


def cmr_gslc(rslc_name: str, day: str) -> dict:
    exact = rslc_name.replace("_L1_", "_L2_").replace("_RSLC_", "_GSLC_")
    query = {"collection_concept_id": GSLC_COLLECTION, "readable_granule_name": exact}
    entries = requests.get(CMR, params=query, timeout=60).json()["feed"]["entry"]
    match = "exact"
    if not entries:
        fields = rslc_name.split("_")
        pattern = "_".join(["NISAR_L2_*_GSLC_*", *fields[5:8], "*", f"{day}T*"])
        query.update(
            {"readable_granule_name": pattern, "options[readable_granule_name][pattern]": "true"}
        )
        entries = requests.get(CMR, params=query, timeout=60).json()["feed"]["entry"]
        match = "same date"
    if not entries:
        raise SystemExit(f"no JPL GSLC for {rslc_name}")
    entry = sorted(entries, key=lambda e: e["title"])[-1]
    url = next(
        l["href"] for l in entry["links"] if l["href"].startswith("https") and l["href"].endswith(".h5")
    )
    return {"title": entry["title"], "url": url, "match": match}


def window(coords: np.ndarray, wanted: np.ndarray) -> slice:
    start = int(np.argmin(np.abs(coords - wanted[0])))
    if abs(coords[start] - wanted[0]) > 1e-6:
        raise SystemExit("our grid is not on JPL's lattice")
    return slice(start, start + len(wanted))


def fetch(args: argparse.Namespace) -> None:
    token = earthdata_token(env_file=args.env_file)
    args.jpl.mkdir(parents=True, exist_ok=True)
    receipt = []
    for ours in sorted(args.ours.glob("CSLC-NISAR_*.h5")):
        day = date_of(ours)
        with h5py.File(ours, "r") as f:
            rslc = f["science/LSAR/GSLC/metadata/processingInformation/inputs/l1SlcGranules"][0]
            rslc = rslc.decode().strip() if isinstance(rslc, bytes) else str(rslc).strip()
            bands = {
                k: (f[f"{GRIDS}/{k}/xCoordinates"][()], f[f"{GRIDS}/{k}/yCoordinates"][()])
                for k in f[GRIDS]
            }
        gslc = cmr_gslc(rslc, day)
        remote = RangeFile(signed(gslc["url"], token))
        out = args.jpl / f"JPL-GSLC_{day}.h5"
        with h5py.File(remote, "r") as src, h5py.File(out, "w") as dst:
            used = src["science/LSAR/GSLC/metadata/processingInformation/inputs/l1SlcGranules"][0]
            for band, (x, y) in bands.items():
                group = f"{GRIDS}/{band}"
                cols = window(src[f"{group}/xCoordinates"][()], x)
                rows = window(src[f"{group}/yCoordinates"][()], y)
                for name in (args.polarization, "mask"):
                    dst[f"{group}/{name}"] = src[f"{group}/{name}"][rows, cols]
                dst[f"{group}/xCoordinates"] = x
                dst[f"{group}/yCoordinates"] = y
        receipt.append(
            {
                "date": day,
                "ours_rslc": rslc,
                "jpl_gslc": gslc["title"],
                "jpl_gslc_rslc": used.decode().rsplit("/", 1)[-1] if isinstance(used, bytes) else str(used),
                "lookup": gslc["match"],
                "bytes_range_read": remote.bytes_read,
            }
        )
        print(json.dumps(receipt[-1]))
    (args.jpl / "fetch.json").write_text(json.dumps(receipt, indent=2) + "\n")


def valid(mask: np.ndarray) -> np.ndarray:
    return (mask > 0) & (mask < 255)


def read_band(path: Path, band: str, pol: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        data = f[f"{GRIDS}/{band}/{pol}"][()].astype(np.complex64)
        mask = f[f"{GRIDS}/{band}/mask"][()]
    return np.where(valid(mask), data, np.nan), mask


def score_band(stacks: list, band: str) -> dict:
    dates = []
    for i, (day, ours, jpl) in enumerate(stacks):
        partner = stacks[i + 1][2] if i + 1 < len(stacks) else stacks[i - 1][2]
        stable = windowed_correlation(jpl, partner) >= 0.7
        gamma = windowed_correlation(ours, jpl)
        d_row, d_col, patches = offsets(jpl, ours)
        dates.append(
            {
                "date": day,
                "valid_fraction_ours": float(np.isfinite(ours).mean()),
                "valid_fraction_jpl": float(np.isfinite(jpl).mean()),
                "median_correlation_stable": float(np.nanmedian(gamma[stable])),
                "median_correlation_all": float(np.nanmedian(gamma)),
                "stable_pixels": int(np.isfinite(gamma[stable]).sum()),
                "offset_rows_px": d_row,
                "offset_cols_px": d_col,
                "offset_patches": patches,
            }
        )
    phase = []
    for (d1, o1, r1), (d2, o2, r2) in zip(stacks, stacks[1:]):
        ours_ifg, jpl_ifg = o1 * np.conj(o2), r1 * np.conj(r2)
        coherence = windowed_correlation(r1, r2)
        mask = (coherence >= 0.9) & np.isfinite(ours_ifg) & np.isfinite(jpl_ifg)
        entry = {"pair": f"{d1}_{d2}", "pixels": int(mask.sum())}
        if mask.sum() >= 100:
            delta = np.angle(np.where(mask, ours_ifg * np.conj(jpl_ifg), 1))
            entry["circular_std_raw_rad"] = circular_std(delta[mask])
            entry["circular_std_rad"] = circular_std(remove_plane(delta, mask)[mask])
        phase.append(entry)
    scored = [p["circular_std_rad"] for p in phase if "circular_std_rad" in p]
    summary = {
        "dates": len(dates),
        "pairs_scored": len(scored),
        "worst_median_correlation": min(d["median_correlation_stable"] for d in dates),
        "worst_abs_offset_rows_px": max(abs(d["offset_rows_px"]) for d in dates),
        "worst_abs_offset_cols_px": max(abs(d["offset_cols_px"]) for d in dates),
        "worst_phase_circular_std_rad": max(scored) if scored else None,
        "median_phase_circular_std_rad": float(np.median(scored)) if scored else None,
    }
    summary["pass_registration"] = bool(
        summary["worst_median_correlation"] >= 0.95
        and summary["worst_abs_offset_rows_px"] <= 0.1
        and summary["worst_abs_offset_cols_px"] <= 0.1
    )
    summary["pass_phase"] = bool(scored) and summary["worst_phase_circular_std_rad"] <= 0.1
    return {"summary": summary, "dates": dates, "phase_pairs": phase}


def score(args: argparse.Namespace) -> None:
    ours = {date_of(p): p for p in args.ours.glob("CSLC-NISAR_*.h5")}
    jpl = {p.stem.split("_")[-1]: p for p in args.jpl.glob("JPL-GSLC_*.h5")}
    days = sorted(set(ours) & set(jpl))
    if len(days) < 2:
        raise SystemExit(f"need >= 2 common dates, found {len(days)}")
    with h5py.File(ours[days[0]], "r") as f:
        bands = sorted(f[GRIDS])
    report = {"label": args.label, "polarization": args.polarization, "bands": {}}
    for band in bands:
        stacks = [
            (d, read_band(ours[d], band, args.polarization)[0], read_band(jpl[d], band, args.polarization)[0])
            for d in days
        ]
        report["bands"][band] = score_band(stacks, band)
        print(band, json.dumps(report["bands"][band]["summary"], indent=2))
    fetched = args.jpl / "fetch.json"
    if fetched.is_file():
        report["sources"] = json.loads(fetched.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "score"):
        p = sub.add_parser(name)
        p.add_argument("--ours", type=Path, required=True)
        p.add_argument("--jpl", type=Path, required=True)
        p.add_argument("--polarization", default="HH")
    sub.choices["fetch"].add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    sub.choices["score"].add_argument("--label", required=True)
    sub.choices["score"].add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"fetch": fetch, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
