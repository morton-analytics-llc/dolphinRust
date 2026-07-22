#!/usr/bin/env python
"""Generate Phase-4 (quality) oracle fixtures from pinned dolphin v0.35.0.

Covers the two quality kernels present in v0.35.0: temporal coherence
(`metrics.estimate_temp_coh`) and the compressed SLC (`_compress.compress`).
NOTE: v0.35.0 has no CRLB or closure-phase modules (those are `main`-only), so
they are deferred — see STATUS.md / PLAYBOOK.md.

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_quality.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dolphin.phase_link._compress import compress
from dolphin.phase_link.metrics import estimate_temp_coh

OUT = Path(__file__).resolve().parent / "fixtures"


def main() -> None:
    stack = np.load(OUT / "slc_stack.npy")  # (nslc, rows, cols)
    c_arrays = np.load(OUT / "cov_C.npy")  # (rows, cols, nslc, nslc)
    phase = np.load(OUT / "phase_emi.npy")  # (rows, cols, nslc)

    # estimate_temp_coh vmaps over (rows, cols) -> expects (rows, cols, nslc).
    temp_coh = np.asarray(estimate_temp_coh(phase, c_arrays))

    # Perturbed phase: imperfect fit so temp_coh spreads below 1 (exercises the
    # full formula, not the degenerate all-ones case).
    rng = np.random.default_rng(99)
    noisy = np.exp(1j * (np.angle(phase) + 0.5 * rng.standard_normal(phase.shape)))
    noisy = noisy.astype(np.complex64)
    temp_coh_noisy = np.asarray(estimate_temp_coh(noisy, c_arrays))

    # Dolphin computes these bounded per-date row means internally, then returns
    # only their argmax as the public `avg_coh` field. Preserve both so Rust can
    # validate the scientific value without misnaming the integer index.
    avg_coh_per_date = np.abs(c_arrays).mean(axis=3)
    avg_coh_reference_idx = np.argmax(avg_coh_per_date, axis=2)

    # compress expects (nslc, rows, cols).
    compressed = compress(stack, np.moveaxis(phase, -1, 0))  # strides (1,1): no upsample

    np.save(OUT / "temp_coh_emi.npy", temp_coh.astype(np.float32))
    np.save(OUT / "phase_noisy.npy", noisy)
    np.save(OUT / "temp_coh_noisy.npy", temp_coh_noisy.astype(np.float32))
    np.save(OUT / "compressed_slc.npy", compressed.astype(np.complex64))
    np.save(OUT / "avg_coh_per_date.npy", np.moveaxis(avg_coh_per_date, -1, 0).astype(np.float32))
    np.save(OUT / "avg_coh_reference_idx.npy", avg_coh_reference_idx.astype(np.int64))

    print(f"wrote quality fixtures to {OUT}")
    print(f"  temp_coh      {temp_coh.shape}  range=[{temp_coh.min():.3f},{temp_coh.max():.3f}]")
    print(f"  temp_coh_noisy range=[{temp_coh_noisy.min():.3f},{temp_coh_noisy.max():.3f}]")
    print(f"  compressed    {compressed.shape} {compressed.dtype}")
    print(
        f"  avg_coh/date  {avg_coh_per_date.shape} "
        f"range=[{avg_coh_per_date.min():.3f},{avg_coh_per_date.max():.3f}]"
    )
    print(f"  avg_coh public argmax {avg_coh_reference_idx.shape} {avg_coh_reference_idx.dtype}")


if __name__ == "__main__":
    main()
