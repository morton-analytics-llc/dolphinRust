#!/usr/bin/env python
"""Generate the phase-similarity oracle fixtures from dolphin v0.42.0 (issue #100).

`dolphin.similarity` (Wang et al. 2022 eq. 5 median / eq. 6 max) landed after the
pinned v0.35.0 oracle, so it is validated against the same forward v0.42.0 env
already used for CRLB and closure phase (see `gen_quality_v042.py`,
VALIDATION.md).

The fixtures are deliberately small and **committed**, so the contract runs in CI
instead of skipping: an oracle gate that only ever runs locally is a gate that
silently stops gating.

Two things are pinned here, not just the metric:

* `circle_idxs_r*.npy` — the neighbour offsets from `get_circle_idxs`. That
  midpoint-circle enumeration defines *which* pixels are compared, so the Rust
  port has to reproduce the set exactly or every downstream number drifts for a
  reason no tolerance would explain.
* the similarity rasters themselves, unmasked and masked.

Run inside the v0.42.0 env:
  oracle/.venv-v042/bin/python oracle/gen_similarity_v042.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dolphin.similarity import get_circle_idxs, max_similarity, median_similarity

OUT = Path(__file__).resolve().parent / "fixtures"

ROWS, COLS, N_IFG = 24, 28, 5
SEARCH_RADIUS = 5
RADII = (3, 5, 8)
SEED = 100  # the issue number, so the fixture provenance is self-evident


def build_stack() -> np.ndarray:
    """A smooth phase ramp with a planted vertical discontinuity, plus speckle.

    The left half carries one ramp, the right half the same ramp offset by pi.
    Neighbour agreement is therefore high inside each half and drops sharply for
    pixels straddling the seam at `COLS // 2` — the analytic behaviour the metric
    exists to detect.
    """
    rng = np.random.default_rng(SEED)
    rows = np.arange(ROWS)[None, :, None]
    cols = np.arange(COLS)[None, None, :]
    scale = np.arange(1, N_IFG + 1)[:, None, None]

    ramp = 0.15 * scale * (0.5 * rows + 0.25 * cols)
    ramp = ramp + np.where(cols >= COLS // 2, np.pi, 0.0)
    noise = 0.05 * rng.standard_normal((N_IFG, ROWS, COLS))
    return np.exp(1j * (ramp + noise)).astype("complex64")


def build_mask(rng: np.random.Generator) -> np.ndarray:
    """A mostly-valid mask with a dead block and scattered dropouts."""
    mask = rng.random((ROWS, COLS)) > 0.15
    mask[4:9, 6:11] = False
    return mask


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1)

    stack = build_stack()
    np.save(OUT / "similarity_stack.npy", stack)

    for radius in RADII:
        idxs = np.asarray(get_circle_idxs(radius), dtype="int32")
        np.save(OUT / f"similarity_circle_idxs_r{radius}.npy", idxs)
        print(f"circle_idxs r={radius}: {len(idxs)} offsets")

    # median_similarity/max_similarity mutate the mask they are handed
    # (`mask[invalid_mask] = False`), so each call gets its own copy.
    median = median_similarity(stack, SEARCH_RADIUS, mask=None)
    np.save(OUT / "similarity_median.npy", np.asarray(median, dtype="float32"))

    maximum = max_similarity(stack, SEARCH_RADIUS, mask=None)
    np.save(OUT / "similarity_max.npy", np.asarray(maximum, dtype="float32"))

    mask = build_mask(rng)
    np.save(OUT / "similarity_mask.npy", mask)
    median_masked = median_similarity(stack, SEARCH_RADIUS, mask=mask.copy())
    np.save(
        OUT / "similarity_median_masked.npy",
        np.asarray(median_masked, dtype="float32"),
    )

    left = median[:, : COLS // 2 - SEARCH_RADIUS]
    seam = median[:, COLS // 2 - 1 : COLS // 2 + 1]
    print(f"stack {stack.shape}, search_radius {SEARCH_RADIUS}")
    print(f"median: interior {np.nanmean(left):.4f}, seam {np.nanmean(seam):.4f}")
    print(f"max:    interior {np.nanmean(maximum):.4f}")
    print(f"masked median finite: {np.isfinite(median_masked).sum()} / {ROWS * COLS}")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
