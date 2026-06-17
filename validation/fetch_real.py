#!/usr/bin/env python
"""Fetch a small real OPERA CSLC-S1 single-burst time series for validation.

Authenticates with the Earthdata bearer token (see validation/creds.sh), pulls
the first N acquisitions of one burst, and writes them to validation/real_data/.
Idempotent: skips files already present and HDF5-valid.

    source validation/creds.sh
    oracle/.venv/bin/python validation/fetch_real.py [--burst T063_133231_IW1] [--n 5]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import asf_search as asf

OUT = Path(__file__).resolve().parent / "real_data"


def is_valid_h5(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1_000_000 and path.read_bytes()[:4] == b"\x89HDF"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", default="T063_133231_IW1")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", default="2023-06-01")
    ap.add_argument("--end", default="2023-10-01")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    token = os.environ["GP_EARTHDATA_TOKEN"]
    session = asf.ASFSession().auth_with_token(token)

    results = asf.search(
        dataset="OPERA-S1",
        processingLevel="CSLC",
        operaBurstID=args.burst,
        start=args.start,
        end=args.end,
    )
    chosen = sorted(results, key=lambda r: r.properties["startTime"])[: args.n]
    print(f"burst {args.burst}: {len(results)} granules, taking {len(chosen)}")

    for r in chosen:
        name = r.properties["fileName"]
        dest = OUT / name
        if is_valid_h5(dest):
            print(f"  have  {name}")
            continue
        print(f"  fetch {name} ...", flush=True)
        r.download(path=str(OUT), session=session)
        size = dest.stat().st_size / 1e6 if dest.exists() else 0
        ok = "ok" if is_valid_h5(dest) else "INVALID"
        print(f"        {size:.1f} MB [{ok}]")

    files = sorted(p.name for p in OUT.glob("OPERA_*.h5") if is_valid_h5(p))
    print(f"\nready: {len(files)} CSLC files in {OUT}")
    for f in files:
        print(" ", f)


if __name__ == "__main__":
    main()
