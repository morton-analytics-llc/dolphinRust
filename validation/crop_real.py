#!/usr/bin/env python
"""Crop the downloaded real OPERA CSLC bursts to a small window for validation.

Both engines must read the same pixels, so we write minimal OPERA-layout HDF5s
(``/data/VV`` + ``/data/x_coordinates`` + ``/data/y_coordinates`` +
``/data/projection``), windowed and georeferenced consistently with the source.
Filenames (burst id + acquisition date) are preserved so both engines parse the
dates. Run after fetch_real.py:

    oracle/.venv/bin/python validation/crop_real.py [--row0 2200 --col0 9700 --size 384]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

SRC = Path(__file__).resolve().parent / "real_data"
OUT = SRC / "cropped"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row0", type=int, default=2200)
    ap.add_argument("--col0", type=int, default=9700)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--burst", default="", help="substring filter (e.g. T005) when real_data holds multiple bursts")
    ap.add_argument("--out", type=Path, default=OUT, help="output dir for cropped granules")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in SRC.glob("OPERA_*.h5") if args.burst in p.name)
    if not files:
        raise SystemExit("no source granules — run validation/fetch_real.py first")
    r0, c0, n = args.row0, args.col0, args.size

    for src in files:
        dst = out / src.name
        with h5py.File(src, "r") as f:
            vv = f["/data/VV"][r0 : r0 + n, c0 : c0 + n]
            xc = f["/data/x_coordinates"][c0 : c0 + n]
            yc = f["/data/y_coordinates"][r0 : r0 + n]
            epsg = int(f["/data/projection"][()])
            wavelength = float(
                f["/metadata/processing_information/input_burst_metadata/wavelength"][()]
            )
        with h5py.File(dst, "w") as g:
            grp = g.create_group("data")
            grp.create_dataset("VV", data=vv.astype(np.complex64))
            grp.create_dataset("x_coordinates", data=xc)
            grp.create_dataset("y_coordinates", data=yc)
            grp.create_dataset("projection", data=np.int32(epsg))
        print(f"{src.name}: {vv.shape} epsg={epsg} -> {dst}")

    print(f"\nwavelength (m): {wavelength}")
    print(f"cropped {len(files)} files to {out}")


if __name__ == "__main__":
    main()
