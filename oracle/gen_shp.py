#!/usr/bin/env python
"""Generate Phase-2 (SHP) oracle fixtures from pinned dolphin v0.35.0.

Reuses the Phase-1 synthetic stack: emits dolphin's GLRT and KS neighbor masks
and the covariance estimated *with* the GLRT SHP weighting, so the Rust SHP
kernels and the SHP-weighted covariance can be validated against the oracle.

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_shp.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import covariance
from dolphin.shp import estimate_neighbors
from dolphin.workflows import ShpMethod

HALF = HalfWindow(2, 2)
STRIDES = Strides(1, 1)
ALPHA = 0.001

OUT = Path(__file__).resolve().parent / "fixtures"


def main() -> None:
    stack = np.load(OUT / "slc_stack.npy")
    nslc = stack.shape[0]
    amp = np.abs(stack)

    glrt = estimate_neighbors(
        halfwin_rowcol=(HALF.y, HALF.x),
        alpha=ALPHA,
        strides={"y": STRIDES.y, "x": STRIDES.x},
        amp_stack=amp,
        nslc=nslc,
        method=ShpMethod.GLRT,
    )
    ks = estimate_neighbors(
        halfwin_rowcol=(HALF.y, HALF.x),
        alpha=ALPHA,
        strides={"y": STRIDES.y, "x": STRIDES.x},
        amp_stack=amp,
        method=ShpMethod.KS,
    )
    c_shp = np.asarray(
        covariance.estimate_stack_covariance(stack, HALF, STRIDES, neighbor_arrays=glrt)
    )

    np.save(OUT / "glrt_neighbors.npy", glrt)
    np.save(OUT / "ks_neighbors.npy", ks)
    np.save(OUT / "cov_C_shp.npy", c_shp)

    print(f"wrote SHP fixtures to {OUT}")
    print(f"  glrt_neighbors {glrt.shape} {glrt.dtype}  true frac={glrt.mean():.3f}")
    print(f"  ks_neighbors   {ks.shape} {ks.dtype}  true frac={ks.mean():.3f}")
    print(f"  cov_C_shp      {c_shp.shape} {c_shp.dtype}")


if __name__ == "__main__":
    main()
