#!/usr/bin/env python
"""Real-stack phase-linking oracle for the GPU spike (dolphin v0.35.0).

Loads the already-fetched cropped Mexico OPERA CSLC-S1 burst stack (13 acqs,
384x384), computes dolphin's covariance and EMI phase exactly as the production
config does (half_window (11,5), strides (1,1), EMI, ref_idx 0), and saves them
as npy so the Rust GPU/CPU phase-linkers can be validated against the oracle on
a *real* stack without linking GDAL/HDF5 into the phaselink crate.

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_phaselink_real.py
Fixtures land in oracle/fixtures/ (git-ignored).
"""

from __future__ import annotations

import glob
from pathlib import Path

import h5py
import numpy as np

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import covariance
from dolphin.phase_link._core import process_coherence_matrices

DATA = "validation/real_data/cropped_mexico"
DATASET = "/data/VV"
HALF = HalfWindow(5, 11)  # dolphin HalfWindow(y, x); config x=11, y=5
STRIDES = Strides(1, 1)
REF_IDX = 0

OUT = Path(__file__).resolve().parent / "fixtures"


def load_stack() -> np.ndarray:
    files = sorted(glob.glob(f"{DATA}/*.h5"))
    if not files:
        raise SystemExit(f"no CSLCs under {DATA}")
    arrs = [h5py.File(f, "r")[DATASET][:] for f in files]
    return np.stack(arrs).astype(np.complex64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stack = load_stack()
    c_arrays = np.asarray(covariance.estimate_stack_covariance(stack, HALF, STRIDES))
    phase_emi, _eig, est_emi = process_coherence_matrices(
        c_arrays, use_evd=False, reference_idx=REF_IDX
    )

    np.save(OUT / "real_cov_C.npy", c_arrays.astype(np.complex64))
    np.save(OUT / "real_phase_emi.npy", np.asarray(phase_emi).astype(np.complex64))

    print(f"wrote real fixtures to {OUT}")
    print(f"  stack       {stack.shape} {stack.dtype}")
    print(f"  real_cov_C  {c_arrays.shape} {c_arrays.dtype}")
    print(f"  real_phase  {np.asarray(phase_emi).shape}")
    print(f"  estimator unique = {np.unique(np.asarray(est_emi))}")


if __name__ == "__main__":
    main()
