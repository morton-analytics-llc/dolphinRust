#!/usr/bin/env python
"""Generate the v1.2.0 quality-layer oracle fixtures from dolphin v0.42.0.

These two layers (CRLB uncertainty, sequential closure phase) do not exist in
the pinned v0.35.0 oracle, so they are validated against a *second*, forward
oracle env at v0.42.0 (CRLB landed v0.40, closure v0.41, plus the v0.42
singular-matrix CRLB fix). This script touches ONLY the two new layers; the
existing kernels stay validated at v0.35.0 (see VALIDATION.md).

Inputs reuse the committed `cov_C.npy` (a complex coherence stack — version
independent data), so standing up this oracle does not re-tune any existing
kernel.

Run inside the v0.42.0 env:  oracle/.venv-v042/bin/python oracle/gen_quality_v042.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from dolphin.phase_link._closure_phase import compute_nearest_closure_phases_batch
from dolphin.phase_link._core import process_coherence_matrices

OUT = Path(__file__).resolve().parent / "fixtures"

# dolphin's conservative look count for CRLB: sqrt(half_y * half_x). cov_C.npy
# was formed with HalfWindow(2, 2) (see gen_phaselink.py), so num_looks = 2.0.
HALF = (2, 2)
NUM_LOOKS = math.sqrt(HALF[0] * HALF[1])
# CRLB / EMI regularization used for the contract — dolphin defaults (no reg).
BETA = 0.0
ZERO_THRESH = 0.0
REF_IDX = 0


def main() -> None:
    c_arrays = np.load(OUT / "cov_C.npy")  # (rows, cols, nslc, nslc) complex

    # --- CRLB via the production path (process_coherence_matrices) -----------
    # Returns (cpx_phase, eig_vals, estimator, crlb_std_dev); we keep the 4th.
    *_, crlb_std_dev = process_coherence_matrices(
        c_arrays,
        use_evd=False,
        beta=BETA,
        zero_correlation_threshold=ZERO_THRESH,
        reference_idx=REF_IDX,
        num_looks=NUM_LOOKS,
        first_real_slc_idx=0,
        compute_crlb=True,
    )
    crlb_std_dev = np.asarray(crlb_std_dev)  # (rows, cols, nslc)

    # --- sequential closure phase (nearest-neighbour triplets) ---------------
    closure = np.asarray(compute_nearest_closure_phases_batch(c_arrays))  # (r,c,N-2)

    # --- singular / ill-conditioned Γ fixtures (the v0.42 fix) ---------------
    # Pathological per-pixel coherence blocks, run through the same path so the
    # Rust kernel must reproduce dolphin's exact NaN / large-sigma behaviour.
    n = c_arrays.shape[-1]
    zero_block = np.zeros((n, n), dtype=c_arrays.dtype)  # fully decorrelated
    identity_block = np.eye(n, dtype=c_arrays.dtype)  # no cross-coherence
    rank1 = np.ones((n, n), dtype=c_arrays.dtype)  # |C|=1 everywhere (singular Γ)
    singular_C = np.stack([zero_block, identity_block, rank1])[None, :, :, :]
    singular_C = singular_C.reshape(1, 3, n, n)  # (rows=1, cols=3, n, n)
    *_, singular_sigma = process_coherence_matrices(
        singular_C,
        use_evd=False,
        beta=BETA,
        zero_correlation_threshold=ZERO_THRESH,
        reference_idx=REF_IDX,
        num_looks=NUM_LOOKS,
        first_real_slc_idx=0,
        compute_crlb=True,
    )
    singular_sigma = np.asarray(singular_sigma)  # (1, 3, n)

    np.save(OUT / "crlb_sigma_v042.npy", crlb_std_dev.astype(np.float32))
    np.save(OUT / "closure_phase_v042.npy", closure.astype(np.float32))
    np.save(OUT / "crlb_singular_C_v042.npy", singular_C.astype(np.complex64))
    np.save(OUT / "crlb_singular_sigma_v042.npy", singular_sigma.astype(np.float32))

    print(f"wrote v0.42.0 quality fixtures to {OUT}")
    print(f"  num_looks = {NUM_LOOKS}")
    with np.errstate(invalid="ignore"):
        print(
            f"  crlb_sigma     {crlb_std_dev.shape}  "
            f"range=[{np.nanmin(crlb_std_dev):.4f},{np.nanmax(crlb_std_dev):.4f}]"
        )
    print(
        f"  closure_phase  {closure.shape}  "
        f"range=[{closure.min():.4f},{closure.max():.4f}]"
    )
    print(f"  singular_sigma {singular_sigma.shape} (zero|identity|rank1):")
    print(f"    {singular_sigma.reshape(3, n)}")


if __name__ == "__main__":
    main()
