#!/usr/bin/env python
"""Generate the v1.2.0 multi-ministack stitching oracle from dolphin v0.42.0.

Mirrors crates/dolphin-workflows/src/sequential.rs over a >=2-ministack stack:
per ministack form the sample covariance, phase-link (EMI), then compute the
three quality layers the way sequential.rs composes them --

  * temporal coherence  : metrics.estimate_temp_coh(cpx, C)         -> (r, c)
  * CRLB sigma (real)   : process_coherence_matrices(..., crlb)[:, real:]
  * closure phase       : compute_nearest_closure_phases_batch(C)   (full)

and stitch across ministacks exactly as sequential.rs does: temporal coherence
by NaN-aware mean (dolphin's `temporal_coherence_average` = numpy.nanmean),
CRLB/closure by band-major concatenation. Reuses the committed
`sequential_slc_stack.npy` so this stands up no new input data.

Run inside the v0.42.0 env:  oracle/.venv-v042/bin/python oracle/gen_stitch_v042.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link._closure_phase import compute_nearest_closure_phases_batch
from dolphin.phase_link._compress import compress
from dolphin.phase_link._core import process_coherence_matrices
from dolphin.phase_link.covariance import estimate_stack_covariance
from dolphin.phase_link.metrics import estimate_temp_coh
from dolphin.stack import CompressedSlcPlan, MiniStackPlanner

OUT = Path(__file__).resolve().parent / "fixtures"

N, SIZE, MAXC = 10, 5, 10  # 10 SLCs, ministack 5 -> two ministacks
HALF = HalfWindow(1, 1)
STRIDES = Strides(1, 1)
BETA, ZERO_THRESH = 0.0, 0.0
NUM_LOOKS = math.sqrt(HALF.y * HALF.x)  # sequential.rs crlb_real_dates


def main() -> None:
    stack = np.load(OUT / "sequential_slc_stack.npy")  # (N, rows, cols) complex64
    planner = MiniStackPlanner(
        file_list=[f"{i}.h5" for i in range(N)],
        dates=[[__import__("datetime").datetime(2020, 1, 1)] for _ in range(N)],
        is_compressed=[False] * N,
        max_num_compressed=MAXC,
        output_reference_idx=0,
        compressed_slc_plan=CompressedSlcPlan.ALWAYS_FIRST,
    )

    comp_slcs: list[np.ndarray] = []
    temp_cohs: list[np.ndarray] = []
    crlbs: list[np.ndarray] = []
    closures: list[np.ndarray] = []

    for ms in planner.plan(SIZE):
        num_comp = len(comp_slcs[-MAXC:])
        real_idx = [int(Path(f).stem) for f in ms.real_slc_file_list]
        cur_comp = comp_slcs[-MAXC:]
        combined = np.concatenate(
            [np.stack(cur_comp), stack[real_idx]] if cur_comp else [stack[real_idx]],
            axis=0,
        ).astype(np.complex64)

        c_arrays = np.asarray(estimate_stack_covariance(combined, HALF, STRIDES))

        # Phase-link at the output reference for temporal coherence.
        cpx, *_ = process_coherence_matrices(
            c_arrays,
            use_evd=False,
            beta=BETA,
            zero_correlation_threshold=ZERO_THRESH,
            reference_idx=ms.output_reference_idx,
            compute_crlb=False,
        )
        cpx = np.exp(1j * np.angle(np.asarray(cpx)))  # unit phasor (r, c, nslc)
        tcoh = np.asarray(estimate_temp_coh(cpx, c_arrays))
        temp_cohs.append(tcoh)

        # CRLB at the last-compressed reference (max(first_real-1, 0)); real bands.
        crlb_ref = max(num_comp - 1, 0)
        *_, crlb = process_coherence_matrices(
            c_arrays,
            use_evd=False,
            beta=BETA,
            zero_correlation_threshold=ZERO_THRESH,
            reference_idx=crlb_ref,
            num_looks=NUM_LOOKS,
            first_real_slc_idx=num_comp,
            compute_crlb=True,
        )
        crlb = np.moveaxis(np.asarray(crlb), -1, 0)[num_comp:]  # (real, r, c)
        crlbs.append(crlb)

        closures.append(np.moveaxis(np.asarray(compute_nearest_closure_phases_batch(c_arrays)), -1, 0))

        comp_slcs.append(
            compress(
                combined,
                np.moveaxis(cpx, -1, 0),  # compress wants (nslc, r, c)
                first_real_slc_idx=num_comp,
                reference_idx=ms.compressed_reference_idx,
            )
        )

    full_tcoh = np.nanmean(np.stack(temp_cohs, axis=0), axis=0)
    full_crlb = np.concatenate(crlbs, axis=0)
    full_closure = np.concatenate(closures, axis=0)

    np.save(OUT / "stitch_temp_coh_full.npy", full_tcoh.astype(np.float32))
    np.save(OUT / "stitch_per_ministack_temp_coh.npy", np.stack(temp_cohs).astype(np.float32))
    np.save(OUT / "stitch_crlb.npy", full_crlb.astype(np.float32))
    np.save(OUT / "stitch_closure.npy", full_closure.astype(np.float32))

    print(f"wrote stitching oracle to {OUT}")
    print(f"  per-ministack temp_coh {np.stack(temp_cohs).shape} -> full {full_tcoh.shape}")
    print(f"  full_tcoh range=[{np.nanmin(full_tcoh):.4f},{np.nanmax(full_tcoh):.4f}]")
    print(f"  full_crlb {full_crlb.shape}  full_closure {full_closure.shape}")


if __name__ == "__main__":
    main()
