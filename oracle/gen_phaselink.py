#!/usr/bin/env python
"""Generate Phase-1 oracle fixtures from pinned dolphin v0.35.0.

Reference oracle (PLAYBOOK §Correctness, secondary check): emit dolphin's
covariance and EVD/EMI estimator outputs for a fixed synthetic SLC stack so the
Rust kernels can be validated to physical tolerances (not bit-exactness).

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_phaselink.py
Fixtures land in oracle/fixtures/ (git-ignored, not committed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import covariance
from dolphin.phase_link._core import process_coherence_matrices

SEED = 1234
NSLC, ROWS, COLS = 8, 12, 12
HALF = HalfWindow(2, 2)
STRIDES = Strides(1, 1)
REF_IDX = 0

OUT = Path(__file__).resolve().parent / "fixtures"


def make_stack() -> np.ndarray:
    """Coherent DS: shared temporal ramp + moderate speckle (so |C| < 1, PD)."""
    rng = np.random.default_rng(SEED)
    ramp = np.exp(1j * 0.5 * np.arange(NSLC))  # true temporal phase
    speckle = (rng.standard_normal((NSLC, ROWS, COLS))
               + 1j * rng.standard_normal((NSLC, ROWS, COLS))) / np.sqrt(2)
    stack = ramp[:, None, None] + 0.3 * speckle
    return stack.astype(np.complex64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stack = make_stack()

    c_arrays = np.asarray(
        covariance.estimate_stack_covariance(stack, HALF, STRIDES)
    )

    phase_evd, eigvals_evd, est_evd = process_coherence_matrices(
        c_arrays, use_evd=True, reference_idx=REF_IDX
    )
    phase_emi, eigvals_emi, est_emi = process_coherence_matrices(
        c_arrays, use_evd=False, reference_idx=REF_IDX
    )

    np.save(OUT / "slc_stack.npy", stack)
    np.save(OUT / "cov_C.npy", c_arrays)
    np.save(OUT / "phase_evd.npy", np.asarray(phase_evd))
    np.save(OUT / "phase_emi.npy", np.asarray(phase_emi))
    np.save(OUT / "estimator_emi.npy", np.asarray(est_emi))

    print(f"wrote fixtures to {OUT}")
    print(f"  slc_stack   {stack.shape} {stack.dtype}")
    print(f"  cov_C       {c_arrays.shape} {c_arrays.dtype}")
    print(f"  phase_evd   {np.asarray(phase_evd).shape}")
    print(f"  phase_emi   {np.asarray(phase_emi).shape}")
    print(f"  estimator_emi unique = {np.unique(np.asarray(est_emi))}")


if __name__ == "__main__":
    main()
