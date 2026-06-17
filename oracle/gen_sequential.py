#!/usr/bin/env python
"""Generate Phase-5 sequential-loop oracle fixtures from dolphin v0.35.0.

Runs the Ansari (2017) sequential estimator over a multi-ministack synthetic
stack using dolphin primitives (run_phase_linking + compress per ministack,
compressed-SLC carry-forward driven by MiniStackPlanner) and dumps the end-to-end
per-date linked-phase history and the compressed SLCs.

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_sequential.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link._compress import compress
from dolphin.phase_link._core import run_phase_linking
from dolphin.stack import CompressedSlcPlan, MiniStackPlanner

N, ROWS, COLS = 10, 6, 6
HALF = HalfWindow(1, 1)
STRIDES = Strides(1, 1)
SIZE, MAXC = 5, 10
OUT = Path(__file__).resolve().parent / "fixtures"


def synth_stack() -> np.ndarray:
    rng = np.random.default_rng(7)
    ramp = np.exp(1j * 0.3 * np.arange(N))
    speckle = (rng.standard_normal((N, ROWS, COLS)) + 1j * rng.standard_normal((N, ROWS, COLS)))
    return (ramp[:, None, None] + 0.3 * speckle / np.sqrt(2)).astype(np.complex64)


def main() -> None:
    stack = synth_stack()
    planner = MiniStackPlanner(
        file_list=[f"{i}.h5" for i in range(N)],
        dates=[[datetime(2020, 1, 1) + timedelta(days=12 * i)] for i in range(N)],
        is_compressed=[False] * N,
        max_num_compressed=MAXC,
        output_reference_idx=0,
        compressed_slc_plan=CompressedSlcPlan.ALWAYS_FIRST,
    )

    comp_slcs: list[np.ndarray] = []
    real_phases: list[np.ndarray] = []
    for ms in planner.plan(SIZE):
        num_comp = int(sum(ms.is_compressed))
        real_idx = [int(Path(f).stem) for f in ms.real_slc_file_list]
        cur_comp = comp_slcs[-MAXC:]
        combined = np.concatenate(
            [np.stack(cur_comp), stack[real_idx]] if cur_comp else [stack[real_idx]], axis=0
        ).astype(np.complex64)

        pl = run_phase_linking(
            combined, HALF, STRIDES, use_evd=False, reference_idx=ms.output_reference_idx
        )
        cpx = np.asarray(pl.cpx_phase)  # (combined_n, out_rows, out_cols)
        comp_slcs.append(
            compress(combined, cpx, first_real_slc_idx=num_comp, reference_idx=ms.compressed_reference_idx)
        )
        real_phases.append(cpx[num_comp:])

    final = np.concatenate(real_phases, axis=0)  # (N, out_rows, out_cols)
    np.save(OUT / "sequential_slc_stack.npy", stack)
    np.save(OUT / "sequential_phase.npy", final.astype(np.complex64))
    np.save(OUT / "sequential_compressed.npy", np.stack(comp_slcs).astype(np.complex64))

    print(f"wrote sequential fixtures to {OUT}")
    print(f"  sequential_phase      {final.shape}")
    print(f"  sequential_compressed {np.stack(comp_slcs).shape}")


if __name__ == "__main__":
    main()
