#!/usr/bin/env python
"""Generate the multi-ministack sequential oracle fixtures (issue #192).

`gen_sequential.py`'s stack has two ministacks, so at most one compressed SLC is
ever carried and dolphin's old and new `always_first` reference rules cannot
differ on it. This stack has four ministacks and decorrelates (exponential, 90
days, no floor), which is where the two rules part from the third ministack on.

The sequential estimator is run from dolphin primitives (`run_phase_linking` +
`compress` per ministack) with the reference indices taken from dolphin's own
`MiniStackPlanner`, so each release's fixture carries that release's rule:

- v0.42.9: both indices on the newest carried compressed SLC
  (`CompressedSlcPlan::AlwaysFirst` in dolphinRust);
- v0.35.0: both on index 0, the oldest (`AlwaysFirstLegacy`).

The dolphin version is part of every file name and the script refuses any other
release.

    oracle/.venv/bin/python oracle/gen_sequential_multi.py              # v0.35.0
    PYTHONPATH=<dolphin v0.42.9> oracle/.venv-v042/bin/python oracle/gen_sequential_multi.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import dolphin
from dolphin._types import HalfWindow, Strides
from dolphin.phase_link._compress import compress
from dolphin.phase_link._core import run_phase_linking
from dolphin.stack import CompressedSlcPlan, MiniStackPlanner

N, ROWS, COLS = 20, 8, 8
HALF = HalfWindow(2, 2)
STRIDES = Strides(1, 1)
SIZE, MAXC = 5, 10
TAU_DAYS, REPEAT = 90.0, 12.0
# The planner's `output_reference_idx` for the default rule of each release.
DEFAULT_REFERENCE = {"0.42.9": None, "0.35.0": 0}
OUT = Path(__file__).resolve().parent / "fixtures"


def synth_stack() -> np.ndarray:
    """Homogeneous distributed scatterers, exponential decorrelation, one true
    phase history (the model of `ref_idx_experiment.py`)."""
    rng = np.random.default_rng(192)
    days = REPEAT * np.arange(N)
    truth = 3.0 * days / 365.25 + 0.8 * np.sin(2 * np.pi * days / 365.25)
    lag = np.abs(days[:, None] - days[None, :])
    covariance = np.exp(-lag / TAU_DAYS) * np.exp(1j * (truth[:, None] - truth[None, :]))
    chol = np.linalg.cholesky(covariance + 1e-10 * np.eye(N))
    white = (rng.standard_normal((N, ROWS * COLS)) + 1j * rng.standard_normal((N, ROWS * COLS))) / np.sqrt(2)
    return (chol @ white).reshape(N, ROWS, COLS).astype(np.complex64)


def main() -> None:
    version = dolphin.__version__
    if version not in DEFAULT_REFERENCE:
        raise SystemExit(f"no sequential oracle pinned for dolphin {version}; expected one of {sorted(DEFAULT_REFERENCE)}")
    tag = "v" + version.replace(".", "")
    stack = synth_stack()
    planner = MiniStackPlanner(
        file_list=[f"{i}.h5" for i in range(N)],
        dates=[[datetime(2020, 1, 1) + timedelta(days=12 * i)] for i in range(N)],
        is_compressed=[False] * N,
        max_num_compressed=MAXC,
        output_reference_idx=DEFAULT_REFERENCE[version],
        compressed_slc_plan=CompressedSlcPlan.ALWAYS_FIRST,
    )

    comp_slcs: list[np.ndarray] = []
    real_phases: list[np.ndarray] = []
    references = []
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
        cpx = np.asarray(pl.cpx_phase)
        comp_slcs.append(
            compress(combined, cpx, first_real_slc_idx=num_comp, reference_idx=ms.compressed_reference_idx)
        )
        real_phases.append(cpx[num_comp:])
        references.append((ms.output_reference_idx, ms.compressed_reference_idx))

    OUT.mkdir(parents=True, exist_ok=True)
    stack_path = OUT / "sequential_multi_slc_stack.npy"
    if stack_path.exists() and not np.array_equal(np.load(stack_path), stack):
        raise SystemExit(f"{stack_path} differs from this env's stack; the two releases must share one input")
    np.save(stack_path, stack)
    np.save(OUT / f"sequential_{tag}_phase.npy", np.concatenate(real_phases, axis=0).astype(np.complex64))
    np.save(OUT / f"sequential_{tag}_compressed.npy", np.stack(comp_slcs).astype(np.complex64))
    print(f"dolphin {version}: (output, compressed) reference per ministack {references}")
    print(f"wrote sequential_multi_slc_stack.npy and sequential_{tag}_{{phase,compressed}}.npy to {OUT}")


if __name__ == "__main__":
    main()
