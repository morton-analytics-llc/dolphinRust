#!/usr/bin/env python
"""Synthesize a date-named single-burst CSLC stack both engines can ingest.

Emits one HDF5 per acquisition (complex64 at ``/data/VV``) named ``cslc_YYYYMMDD.h5``
so dolphin's date parser accepts it, plus a deterministic smooth deformation signal
(a linear range ramp growing linearly in time). Fixed seed; ``--speckle`` controls the
per-SLC complex noise so the speckle-vs-agreement hypothesis can be probed.

The signal is small (|ifg phase| < ~1.2 rad) so SNAPHU unwrapping is cycle-free and the
end-to-end comparison isolates the estimators rather than integer-cycle disagreements.

Run inside the pinned oracle env:
  oracle/.venv/bin/python validation/gen_stack.py --outdir validation/data --speckle 0.05
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np

DATASET = "/data/VV"
N, ROWS, COLS = 5, 48, 64
SEED = 21
START = date(2022, 11, 19)
CADENCE_DAYS = 1  # acquisition spacing encoded in the filenames dolphin parses


def synth_stack(speckle: float) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    _, xx = np.mgrid[0:ROWS, 0:COLS].astype(np.float64)
    ramp = xx / COLS  # in [0, 1)
    phases = np.stack([0.3 * t * ramp for t in range(N)])  # smooth, small, grows in time
    noise = rng.standard_normal((N, ROWS, COLS)) + 1j * rng.standard_normal((N, ROWS, COLS))
    return (np.exp(1j * phases) + speckle * noise / np.sqrt(2)).astype(np.complex64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--speckle", type=float, default=0.05)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    stack = synth_stack(args.speckle)
    files = []
    for t in range(N):
        d = START + timedelta(days=t * CADENCE_DAYS)
        path = args.outdir / f"cslc_{d:%Y%m%d}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(DATASET, data=stack[t])
        files.append(path)
    print(f"wrote {N} CSLC files to {args.outdir} (speckle={args.speckle}, cadence={CADENCE_DAYS}d)")
    for f in files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
