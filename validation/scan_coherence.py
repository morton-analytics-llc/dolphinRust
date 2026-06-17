#!/usr/bin/env python
"""Locate the most coherent window in the real burst to crop for validation.

A random crop can land on decorrelated terrain where the displacement signal is
below the cross-engine noise floor. This scans a decimated version of the full
stack, computes an average adjacent-pair interferometric coherence (boxcar), and
prints the full-resolution (row0, col0) of the most coherent `size`-window.

    oracle/.venv/bin/python validation/scan_coherence.py [--size 384 --dec 4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter

SRC = Path(__file__).resolve().parent / "real_data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--dec", type=int, default=4)
    ap.add_argument("--box", type=int, default=5)
    args = ap.parse_args()
    d = args.dec

    files = sorted(SRC.glob("OPERA_*.h5"))
    stack = [h5py.File(f, "r")["/data/VV"][::d, ::d] for f in files]
    stack = np.nan_to_num(np.stack(stack).astype(np.complex64))  # (n, R/d, C/d)

    # Average adjacent-pair coherence with a spatial boxcar.
    coh = np.zeros(stack.shape[1:], dtype=np.float64)
    for a, b in zip(stack[:-1], stack[1:]):
        ifg = a * np.conj(b)
        num = np.abs(uniform_filter(ifg.real, args.box) + 1j * uniform_filter(ifg.imag, args.box))
        den = np.sqrt(
            uniform_filter(np.abs(a) ** 2, args.box) * uniform_filter(np.abs(b) ** 2, args.box)
        )
        coh += num / (den + 1e-9)
    coh /= len(stack) - 1

    # Mean coherence over each candidate window (decimated size), find the best.
    w = max(1, args.size // d)
    win_mean = uniform_filter(coh, w)  # value at center ~= window mean
    # restrict to centers where the full window fits
    half = w // 2
    valid = np.full(coh.shape, -1.0)
    valid[half : coh.shape[0] - half, half : coh.shape[1] - half] = win_mean[
        half : coh.shape[0] - half, half : coh.shape[1] - half
    ]
    ci, cj = np.unravel_index(np.argmax(valid), valid.shape)
    row0 = max(0, (ci - half) * d)
    col0 = max(0, (cj - half) * d)
    print(f"overall mean coherence: {coh.mean():.3f}")
    print(f"best window mean coherence: {valid[ci, cj]:.3f}")
    print(f"crop here:  --row0 {row0} --col0 {col0} --size {args.size}")


if __name__ == "__main__":
    main()
