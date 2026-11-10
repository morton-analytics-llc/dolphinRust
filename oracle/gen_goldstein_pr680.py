#!/usr/bin/env python
"""Generate Goldstein contracts from dolphin's post-PR-#680 filter.

The pinned oracle env carries dolphin 0.35.0, which predates PR #680, so the
reference implementation below is upstream `src/dolphin/goldstein.py` at commit
47052b4ec7c490e8ec4f3554be8d1fc6a8a74ef4 (isce-framework/dolphin), reproduced
without change apart from removing the docstrings.

Run: oracle/.venv/bin/python oracle/gen_goldstein_pr680.py
"""

from pathlib import Path
import json

import numpy as np

OUT = (
    Path(__file__).resolve().parents[1]
    / "crates/dolphin-filtering/tests/fixtures/goldstein_pr680.json"
)


def goldstein(phase, alpha, psize=32):
    def apply_pspec(data):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha = }")
        weight = np.power(np.abs(data) ** 2, alpha / 2)
        return weight * data

    def make_weight(nxp, nyp):
        wx = 1.0 - np.abs(np.arange(nxp // 2) - (nxp / 2.0 - 1.0)) / (nxp / 2.0 - 1.0)
        wy = 1.0 - np.abs(np.arange(nyp // 2) - (nyp / 2.0 - 1.0)) / (nyp / 2.0 - 1.0)
        quadrant = np.outer(wy, wx)
        return np.block(
            [
                [quadrant, np.flip(quadrant, axis=1)],
                [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)],
            ]
        )

    def patch_goldstein_filter(data, weight, psize):
        data = np.fft.fft2(data, s=(psize, psize))
        data = apply_pspec(data)
        data = np.fft.ifft2(data, s=(psize, psize))
        return weight * data

    def apply_goldstein_filter(data):
        empty_mask = np.isnan(data) | (data == 0)
        if np.all(empty_mask):
            return data
        nrows, ncols = data.shape
        step = psize // 2
        pad_top = step
        pad_left = step
        pad_bottom = step + (step - (nrows % step)) % step
        pad_right = step + (step - (ncols % step)) % step
        data_padded = np.pad(
            data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect"
        )
        out = np.zeros(data_padded.shape, dtype=np.complex64)
        weight_sum = np.zeros(data_padded.shape, dtype=np.float64)
        weight_matrix = make_weight(psize, psize)
        padded_rows, padded_cols = data_padded.shape
        for i in range(0, padded_rows - psize + 1, step):
            for j in range(0, padded_cols - psize + 1, step):
                data_window = data_padded[i : i + psize, j : j + psize]
                filtered_window = patch_goldstein_filter(data_window, weight_matrix, psize)
                out[i : i + psize, j : j + psize] += filtered_window
                weight_sum[i : i + psize, j : j + psize] += weight_matrix
        valid = weight_sum > 0
        out[valid] /= weight_sum[valid]
        out = out[pad_top : pad_top + nrows, pad_left : pad_left + ncols]
        out[empty_mask] = 0
        return out

    if np.iscomplexobj(phase):
        return apply_goldstein_filter(phase)
    return apply_goldstein_filter(np.exp(1j * phase))


def finite_or_none(values):
    return [None if not np.isfinite(v) else float(v) for v in values]


def main():
    rng = np.random.default_rng(680)
    cases = []
    for name, shape, psize, alpha in [
        ("divisible", (48, 48), 16, 0.5),
        ("trailing_edges", (37, 45), 16, 0.5),
        ("small_patch_zero_phase_and_empty", (21, 27), 8, 0.8),
    ]:
        yy, xx = np.indices(shape, dtype=np.float64)
        phase = np.exp(1j * (0.2 * xx - 0.1 * yy + rng.standard_normal(shape)))
        if name == "small_patch_zero_phase_and_empty":
            phase[3:9, 4:12] = 1.0 + 0.0j  # valid, zero phase: not empty
            phase[14:17, 20:25] = 0.0  # zero magnitude: empty
            phase[0, 0] = complex(np.nan, np.nan)  # NaN: empty
        phase = phase.astype(np.complex64)
        out = goldstein(phase, alpha=alpha, psize=psize)
        cases.append(
            {
                "name": name,
                "shape": list(shape),
                "psize": psize,
                "alpha": alpha,
                "input_re": finite_or_none(phase.real.ravel()),
                "input_im": finite_or_none(phase.imag.ravel()),
                "expected_re": finite_or_none(out.real.ravel()),
                "expected_im": finite_or_none(out.imag.ravel()),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source_commit": "47052b4ec7c490e8ec4f3554be8d1fc6a8a74ef4", "cases": cases}))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
