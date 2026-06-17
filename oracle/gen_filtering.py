#!/usr/bin/env python
"""Generate Phase-7 (filtering) oracle fixtures from dolphin v0.35.0.

Emits the long-wavelength FFT Gaussian high-pass filter output (no bad pixels,
so GDAL fill is identity) and the Goldstein adaptive filter output for synthetic
inputs.

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_filtering.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dolphin.filtering import filter_long_wavelength
from dolphin.goldstein import goldstein

OUT = Path(__file__).resolve().parent / "fixtures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)

    # --- long-wavelength high-pass: smooth ramp + short-scale signal, no nodata ---
    rows, cols = 64, 64
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    ramp = 0.01 * xx + 0.005 * yy  # long-wavelength component
    fine = np.sin(2 * np.pi * xx / 6.0) + np.cos(2 * np.pi * yy / 5.0)
    unw = (ramp + fine + 10.0).astype(np.float64)  # +10 so no zeros (all in-bounds)
    bad = np.zeros((rows, cols), dtype=bool)
    lw = filter_long_wavelength(unw, bad, wavelength_cutoff=5_000, pixel_spacing=30)

    np.save(OUT / "filt_lw_input.npy", unw)
    np.save(OUT / "filt_lw_output.npy", np.asarray(lw).astype(np.float64))

    # --- Goldstein adaptive filter on a complex phase field ---
    g_rows, g_cols = 48, 48
    phase = np.exp(1j * (0.2 * xx[:g_rows, :g_cols] + rng.standard_normal((g_rows, g_cols))))
    phase = phase.astype(np.complex64)
    gout = goldstein(phase, alpha=0.5, psize=16)

    np.save(OUT / "filt_gold_input.npy", phase)
    np.save(OUT / "filt_gold_output.npy", np.asarray(gout).astype(np.complex64))

    print(f"wrote filtering fixtures to {OUT}")
    print(f"  long-wavelength {unw.shape}  out range=[{lw.min():.3f},{lw.max():.3f}]")
    print(f"  goldstein {phase.shape}  out |.| max={np.abs(gout).max():.3f}")


if __name__ == "__main__":
    main()
