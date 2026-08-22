#!/usr/bin/env python
"""Generate the issue-50 layover/shadow-mask oracle from dolphin v0.35.0.

The fixture pins dolphin's ZERO_IS_NODATA polarity and its looked-validity
rule: an output cell is invalid only when every native pixel in that stride
cell is invalid.

Run with the pinned environment:
    oracle/.venv/bin/python oracle/gen_layover_shadow_mask.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import dolphin
import numpy as np
from dolphin import io, masking
from dolphin._types import HalfWindow, Strides
from dolphin.phase_link._core import run_phase_linking

VERSION = "0.35.0"
SOURCE_COMMIT = "e567e554300f9bb2c6c4c49358d41876ce81e5a7"
NSLC = 6
OUT = Path(__file__).resolve().parent / "fixtures"


def make_stack(shape: tuple[int, int]) -> np.ndarray:
    """Deterministic coherent stack with spatial and temporal phase variation."""
    rows, cols = shape
    dates = np.arange(NSLC)[:, None, None]
    row, col = np.indices((rows, cols))
    phase = 0.25 * dates + 0.03 * row + 0.02 * col + 0.01 * dates * row
    return np.exp(1j * phase).astype(np.complex64)


def phase_link(
    stack: np.ndarray, nodata_mask: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    result = run_phase_linking(
        stack,
        half_window=HalfWindow(1, 1),
        strides=Strides(stride, stride),
        use_evd=True,
        reference_idx=0,
        nodata_mask=nodata_mask,
    )
    return np.asarray(result.cpx_phase), np.asarray(result.temp_coh)


def main() -> None:
    if dolphin.__version__ != VERSION:
        raise RuntimeError(
            f"layover/shadow oracle requires dolphin {VERSION}, got {dolphin.__version__}"
        )

    # Nonzero values are deliberately not limited to 1. The upper-right 2x2
    # stride cell is wholly invalid; the upper-left and lower-right cells are
    # only partially invalid.
    mask_values = np.array(
        [
            [0, 2, 0, 0],
            [2, 2, 0, 0],
            [255, 1, 3, 3],
            [1, 1, 3, 0],
        ],
        dtype=np.uint8,
    )
    stack = make_stack(mask_values.shape)

    with TemporaryDirectory() as directory:
        mask_file = Path(directory) / "layover_shadow_mask.tif"
        io.write_arr(
            arr=mask_values,
            output_name=mask_file,
            geotransform=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
            projection="EPSG:4326",
            nodata=255,
        )
        nodata_mask = masking.load_mask_as_numpy(mask_file)

    expected_nodata = (mask_values == 0) | (mask_values == 255)
    if not np.array_equal(nodata_mask, expected_nodata):
        raise AssertionError("dolphin v0.35.0 mask polarity changed")

    phase_stride1, quality_stride1 = phase_link(stack, nodata_mask, 1)
    phase_stride2, quality_stride2 = phase_link(stack, nodata_mask, 2)

    expected_stride2_validity = np.array(
        [[True, False], [True, True]], dtype=bool
    )
    if not np.array_equal(np.isfinite(quality_stride1), ~nodata_mask):
        raise AssertionError("stride-1 validity no longer follows zero-is-invalid")
    if not np.array_equal(
        np.isfinite(quality_stride2), expected_stride2_validity
    ):
        raise AssertionError("stride-2 validity no longer uses all-invalid reduction")

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "layover_shadow_mask_values.npy", mask_values)
    np.save(
        OUT / "layover_shadow_mask_validity.npy",
        (~nodata_mask).astype(np.uint8),
    )
    np.save(OUT / "layover_shadow_mask_stack.npy", stack)
    np.save(OUT / "layover_shadow_mask_phase_stride1.npy", phase_stride1)
    np.save(
        OUT / "layover_shadow_mask_temporal_coherence_stride1.npy",
        quality_stride1,
    )
    np.save(OUT / "layover_shadow_mask_phase_stride2.npy", phase_stride2)
    np.save(
        OUT / "layover_shadow_mask_temporal_coherence_stride2.npy",
        quality_stride2,
    )

    print(
        f"dolphin {VERSION} ({SOURCE_COMMIT}): wrote layover/shadow fixtures to {OUT}"
    )
    print(f"  native valid:\n{(~nodata_mask).astype(np.uint8)}")
    print(f"  stride-2 valid:\n{expected_stride2_validity.astype(np.uint8)}")


if __name__ == "__main__":
    main()
