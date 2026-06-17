#!/usr/bin/env python
"""Generate Phase-3 (PS) oracle fixtures from pinned dolphin v0.35.0.

Emits the amplitude mean / dispersion / PS mask from `calc_ps_block` (with the
`create_ps` nodata rule), and a PS-fill case (`fill_ps_pixels`) at strides (2,2)
with a fixed PS pattern so the brightest-in-window selection is exercised.

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_ps.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dolphin._types import Strides
from dolphin.ps import calc_ps_block
from dolphin.phase_link._ps_filling import fill_ps_pixels

THRESHOLD = 0.25
OUT = Path(__file__).resolve().parent / "fixtures"


def main() -> None:
    stack = np.load(OUT / "slc_stack.npy")  # (nslc, rows, cols) complex64
    nslc, rows, cols = stack.shape
    amp = np.abs(stack)

    mean, amp_disp, ps = calc_ps_block(amp, THRESHOLD, min_count=nslc)
    ps_u8 = ps.astype(np.uint8)
    ps_u8[amp_disp == 0] = 255  # create_ps nodata rule

    np.save(OUT / "amp_mean.npy", mean.astype(np.float32))
    np.save(OUT / "amp_disp.npy", amp_disp.astype(np.float32))
    np.save(OUT / "ps_mask.npy", ps_u8)

    # ---- PS-fill case (strides 2,2), fixed PS pattern, unit-magnitude phase ----
    strides = Strides(2, 2)
    ps_bool = ((np.arange(rows)[:, None] * np.arange(cols)[None, :]) % 7 == 0)
    out_rows, out_cols = rows // strides.y, cols // strides.x
    cpx_phase = np.ones((nslc, out_rows, out_cols), dtype=np.complex64)
    temp_coh = np.zeros((out_rows, out_cols), dtype=np.float32)

    fill_ps_pixels(cpx_phase, temp_coh, stack, ps_bool, strides, None, 0, use_max_ps=True)

    np.save(OUT / "ps_fill_mask.npy", ps_bool)
    np.save(OUT / "ps_fill_cpx_phase.npy", cpx_phase)
    np.save(OUT / "ps_fill_temp_coh.npy", temp_coh)

    print(f"wrote PS fixtures to {OUT}")
    print(f"  amp_mean/amp_disp {mean.shape} {mean.dtype}")
    print(f"  ps_mask values = {np.unique(ps_u8)}  PS frac={(ps_u8 == 1).mean():.3f}")
    print(f"  ps_fill: strides=2 PS frac={ps_bool.mean():.3f} temp_coh max={temp_coh.max()}")


if __name__ == "__main__":
    main()
