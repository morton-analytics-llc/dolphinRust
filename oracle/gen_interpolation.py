#!/usr/bin/env python
"""Generate the pre-unwrap interpolation oracle fixtures from pinned dolphin v0.35.0.

`dolphin.interpolation.interpolate` is what `unwrap_options.run_interpolation`
runs on each wrapped interferogram before the unwrapper (v0.35.0
`unwrap/_unwrap.py`); v0.42.0's module is byte-identical bar the logger name.

Committed and small (~60 KB), like the similarity fixtures, so the contract runs
in CI instead of skipping.

Cases:

* `interpolation_cont_out` — continuous weights, `weight_cutoff=0.5`, defaults
  otherwise (`max_radius=51`, `num_neighbors=20`, `alpha=0.75`).
* `interpolation_mask_out` — the unwrap stage's own call: boolean coherent-pixel
  mask cast to weights, `weight_cutoff=0.3`, `max_radius=4`. The small radius
  leaves some low-weight pixels with fewer than 20 qualifying neighbours and one
  isolated block with none, exercising both of dolphin's edge paths.

The ifg holds exact zeros (nodata) both at low- and high-weight pixels: dolphin
skips zero pixels but still uses a zero neighbour's phase (angle 0) when its
weight qualifies, and the port must do the same.

Run inside the pinned env:
  oracle/.venv/bin/python oracle/gen_interpolation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dolphin
from dolphin.interpolation import interpolate

OUT = Path(__file__).resolve().parent / "fixtures"
ROWS, COLS = 36, 44
SEED = 25  # the issue number


def build_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    phase = 0.35 * rr - 0.22 * cc + rng.normal(scale=0.6, size=(ROWS, COLS))
    amplitude = rng.uniform(0.5, 2.0, size=(ROWS, COLS))
    ifg = (amplitude * np.exp(1j * phase)).astype(np.complex64)
    weights = rng.uniform(0.0, 1.0, size=(ROWS, COLS)).astype(np.float32)
    # A decorrelated block wider than max_radius=4 in the mask case.
    weights[10:22, 12:26] = 0.05
    ifg[3, 5] = 0          # nodata at a low-weight pixel
    weights[3, 5] = 0.1
    ifg[15, 30] = 0        # nodata at a high-weight pixel (still a neighbour)
    weights[15, 30] = 0.95
    return ifg, weights


def main() -> None:
    assert dolphin.__version__ == "0.35.0", dolphin.__version__
    OUT.mkdir(exist_ok=True)
    ifg, weights = build_inputs()
    cont = interpolate(ifg, weights, weight_cutoff=0.5)
    mask = weights >= 0.3
    masked = interpolate(ifg, mask, weight_cutoff=0.3, max_radius=4)
    np.save(OUT / "interpolation_ifg.npy", ifg)
    np.save(OUT / "interpolation_weights.npy", weights)
    np.save(OUT / "interpolation_cont_out.npy", cont.astype(np.complex64))
    np.save(OUT / "interpolation_mask_out.npy", masked.astype(np.complex64))
    print(f"dolphin {dolphin.__version__}: wrote interpolation_* to {OUT}")


if __name__ == "__main__":
    main()
